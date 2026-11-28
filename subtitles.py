import gc, json, os, re, subprocess, sys, tempfile, time, wave
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from config import (
    APP_DIR,
    DIAR_WORD_SPLIT_GAP_SEC,
    N_THREADS,
    WHISPER_DIR,
    WHISPER_REPO_ID,
    WHISPER_MODEL_FILE,
    WHISPER_VAD_REPO_ID,
    WHISPER_VAD_MODEL_FILE,
)
from diarize import diarize_segments
from models import chat, close_model

SR = 16000             

CHUNK_CHARS = 4000       
NOTE_SENTENCES = "4-7"   
EVENT_SENTENCES = "3-6" 
EVENT_COUNT = "1-2"      
RECENT_NOTES = 3       
MAX_FAR_CHARS = 2500     
FAR_WORDS = 300          


def free():
    gc.collect()


def extract_audio(input_video, output_audio, track=0, sample_rate=16000, mono=True):
    Path(output_audio).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", input_video, "-map", f"0:a:{track}", "-vn",
           "-acodec", "pcm_s16le", "-ar", str(sample_rate)]
    if mono:
        cmd += ["-ac", "1"]
    cmd.append(output_audio)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if (os.name == "nt" and getattr(sys, "frozen", False)) else 0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка ffmpeg:\n{result.stderr}")
    print(f"Аудио сохранено в {output_audio}")
    return output_audio


def ts(t):
    h, rem = divmod(int(t * 1000), 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


_VAD_SEG_RE = re.compile(
    r"Speech segment\s+\d+:\s*start\s*=\s*([0-9.]+)\s*,\s*end\s*=\s*([0-9.]+)"
)
_LANG_RE = re.compile(r"auto-detected language:\s*([a-zA-Z-]+)")


def _find_whisper_bin(filename: str) -> Path:
    if not WHISPER_DIR:
        raise ValueError(f"Не указан путь WHISPER_DIR в config.py (папка с {filename})")
    exe = Path(WHISPER_DIR) / filename
    if not exe.exists():
        raise FileNotFoundError(f"Не найден {filename} по пути: {exe}")
    return exe


def _run_subprocess(
    cmd: list[str],
    cwd: Path,
    log_name: str,
    stop_event=None,
    capture_output: bool = False,
) -> str:
    popen_kwargs = {"cwd": str(cwd)}
    log_handle = None
    out_file = None

    if capture_output:
        out_file = tempfile.TemporaryFile(mode="w+b")
        popen_kwargs.update({
            "stdin": subprocess.DEVNULL,
            "stdout": out_file,
            "stderr": subprocess.STDOUT,
        })
    elif getattr(sys, "frozen", False):
        try:
            log_handle = open(APP_DIR / log_name, "a", encoding="utf-8", errors="replace")
        except OSError:
            log_handle = subprocess.DEVNULL
        popen_kwargs.update({
            "stdin": subprocess.DEVNULL,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
        })

    if os.name == "nt" and getattr(sys, "frozen", False):
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    finally:
        if log_handle is not None and log_handle is not subprocess.DEVNULL:
            log_handle.close()

    try:
        while proc.poll() is None:
            if stop_event is not None and stop_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise InterruptedError("Остановлено пользователем")
            time.sleep(0.5)

        stdout_text = ""
        if out_file is not None:
            out_file.seek(0)
            stdout_text = out_file.read().decode("utf-8", errors="replace")
    finally:
        if out_file is not None:
            out_file.close()

    if proc.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} завершился с кодом {proc.returncode}")

    return stdout_text


def _parse_vad_segments(output_text: str) -> list[tuple[float, float]]:
    # whisper-vad-speech-segments печатает границы в сотых долях секунды (centiseconds)
    intervals = []
    for m in _VAD_SEG_RE.finditer(output_text):
        start = float(m.group(1)) / 100.0
        end = float(m.group(2)) / 100.0
        if end > start:
            intervals.append((start, end))
    return intervals


def _detect_speech_intervals(
    audio_path: Path, vad_model_path: str, stop_event=None
) -> list[tuple[float, float]]:
    vad_exe = _find_whisper_bin("whisper-vad-speech-segments.exe")
    cmd = [
        str(vad_exe),
        "-vm", vad_model_path,
        "-f", str(Path(audio_path).resolve()),
        "-t", str(N_THREADS),
    ]
    out = _run_subprocess(
        cmd,
        cwd=vad_exe.parent,
        log_name="whisper.log",
        stop_event=stop_event,
        capture_output=True,
    )
    return _parse_vad_segments(out)


def _mute_non_speech(
    src_wav: Path,
    dst_wav: Path,
    intervals: list[tuple[float, float]],
    pad_sec: float = 0.2,
) -> None:
    with wave.open(str(src_wav), "rb") as wf:
        params = wf.getparams()
        if params.sampwidth != 2:
            raise ValueError(f"Ожидался 16-bit PCM WAV, получено {params.sampwidth * 8}-bit")
        raw = wf.readframes(params.nframes)

    samples = np.frombuffer(raw, dtype=np.int16)
    if params.nchannels > 1:
        samples = samples.reshape(-1, params.nchannels)
    out = np.zeros_like(samples)

    total = len(samples)
    sr = params.framerate
    for start_sec, end_sec in intervals:
        i = max(0, int((start_sec - pad_sec) * sr))
        j = min(total, int((end_sec + pad_sec) * sr))
        if j > i:
            out[i:j] = samples[i:j]

    with wave.open(str(dst_wav), "wb") as wf:
        wf.setparams(params)
        wf.writeframes(out.tobytes())


def _clamp_segments_to_vad(
    segments: list[dict],
    vad_intervals: list[tuple[float, float]],
    pad_sec: float = 0.25,
) -> list[dict]:
    clamped = []
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        overlapping = [
            (vs, ve) for vs, ve in vad_intervals
            if ve > seg_start and vs < seg_end
        ]
        if not overlapping:
            continue
        start = max(seg_start, overlapping[0][0] - pad_sec)
        end = min(seg_end, overlapping[-1][1] + pad_sec)
        if end > start:
            clamped.append({
                "start": start,
                "end": end,
                "text": seg["text"],
            })
    return clamped


def _split_item_by_token_gaps(item: dict, gap_sec: float = DIAR_WORD_SPLIT_GAP_SEC) -> list[dict]:
    """Если whisper -ojf вернул токены с таймкодами, делит склеенные реплики по внутренним паузам."""
    tokens = item.get("tokens")
    if not tokens:
        text = (item.get("text") or "").strip()
        if not text or text == "[BLANK_AUDIO]":
            return []
        offsets = item["offsets"]
        return [{"start": offsets["from"] / 1000.0, "end": offsets["to"] / 1000.0, "text": text}]

    chunks: list[dict] = []
    cur_text: list[str] = []
    cur_start: float | None = None
    cur_end: float | None = None

    for tok in tokens:
        t_str = tok.get("text") or ""
        if not t_str or (t_str.startswith("[") and t_str.endswith("]")):
            continue
        off = tok.get("offsets")
        if not off:
            cur_text.append(t_str)
            continue
        t_from = off["from"] / 1000.0
        t_to = max(t_from, off["to"] / 1000.0)
        if cur_end is not None and (t_from - cur_end) >= gap_sec and "".join(cur_text).strip():
            chunks.append({
                "start": cur_start if cur_start is not None else t_from,
                "end": cur_end,
                "text": "".join(cur_text).strip(),
            })
            cur_text = []
            cur_start = None
        if cur_start is None:
            cur_start = t_from
        cur_end = t_to
        cur_text.append(t_str)

    tail = "".join(cur_text).strip()
    if tail and cur_start is not None and cur_end is not None:
        chunks.append({"start": cur_start, "end": cur_end, "text": tail})

    if not chunks:
        text = (item.get("text") or "").strip()
        if text and text != "[BLANK_AUDIO]":
            offsets = item["offsets"]
            chunks.append({"start": offsets["from"] / 1000.0, "end": offsets["to"] / 1000.0, "text": text})
    return chunks


def _parse_whisper_json(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = []
    for item in data.get("transcription", []):
        segments.extend(_split_item_by_token_gaps(item))
    return segments


def _write_speech_sample(
    src_wav: Path,
    dst_wav: Path,
    intervals: list[tuple[float, float]],
    target_sec: float = 30.0,
) -> bool:
    if not intervals:
        return False

    mid = len(intervals) // 2
    ordered = intervals[mid:] + intervals[:mid]

    chunks = []
    collected = 0
    with wave.open(str(src_wav), "rb") as wf:
        params = wf.getparams()
        sr = params.framerate
        total = params.nframes
        target_frames = int(target_sec * sr)

        for start_sec, end_sec in ordered:
            if collected >= target_frames:
                break
            i = max(0, int(start_sec * sr))
            j = min(total, int(end_sec * sr))
            if j <= i:
                continue
            take = min(j - i, target_frames - collected)
            wf.setpos(i)
            chunks.append(wf.readframes(take))
            collected += take

    if collected < sr:
        return False

    with wave.open(str(dst_wav), "wb") as wf:
        wf.setparams(params)
        wf.writeframes(b"".join(chunks))
    return True


def _detect_language(
    whisper_exe: Path,
    model_path: str,
    src_wav: Path,
    vad_intervals: list[tuple[float, float]] | None,
    workdir: Path,
    stop_event=None,
) -> str:
    if not vad_intervals:
        return "auto"

    lang_wav = (workdir / "detect_lang.wav").resolve()
    try:
        if not _write_speech_sample(src_wav, lang_wav, vad_intervals):
            return "auto"
        cmd = [
            str(whisper_exe),
            "-m", model_path,
            "-f", str(lang_wav),
            "-l", "auto",
            "-dl",
            "-t", str(N_THREADS),
        ]
        out = _run_subprocess(
            cmd,
            cwd=whisper_exe.parent,
            log_name="whisper.log",
            stop_event=stop_event,
            capture_output=True,
        )
    finally:
        lang_wav.unlink(missing_ok=True)

    m = _LANG_RE.search(out)
    if m:
        lang = m.group(1)
        print(f"Определён язык: {lang}")
        return lang
    return "auto"


def _emit(on_progress, stage_frac: float, step_frac: float | None, text: str) -> None:
    if on_progress is not None:
        on_progress(stage_frac, step_frac, text)


def _check_stop(stop_event) -> None:
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Остановлено пользователем")


def transcribe(
    audio_path,
    workdir,
    model_file=WHISPER_MODEL_FILE,
    vad_model_file=WHISPER_VAD_MODEL_FILE,
    language=None,
    stop_event=None,
    on_progress=None,
) -> list[dict]:
    whisper_exe = _find_whisper_bin("whisper-cli.exe")

    _check_stop(stop_event)
    _emit(on_progress, 0.15, None, "Субтитры: проверка моделей Whisper...")
    model_path = hf_hub_download(repo_id=WHISPER_REPO_ID, filename=model_file)
    _check_stop(stop_event)
    vad_path = (
        hf_hub_download(repo_id=WHISPER_VAD_REPO_ID, filename=vad_model_file)
        if vad_model_file
        else None
    )
    _check_stop(stop_event)

    src_wav = Path(audio_path).resolve()
    vad_intervals = None
    vad_wav = None
    target_wav = src_wav

    if vad_path:
        _emit(on_progress, 0.25, None, "Субтитры: поиск речи (VAD)...")
        vad_intervals = _detect_speech_intervals(src_wav, vad_path, stop_event=stop_event)
        if not vad_intervals:
            return []
        if language is None:
            _emit(on_progress, 0.40, None, "Субтитры: определение языка...")
            language = _detect_language(
                whisper_exe,
                model_path,
                src_wav,
                vad_intervals,
                Path(workdir),
                stop_event=stop_event,
            )
        _emit(on_progress, 0.48, None, "Субтитры: подготовка аудио...")
        vad_wav = (Path(workdir) / "audio_16k_vad.wav").resolve()
        _mute_non_speech(src_wav, vad_wav, vad_intervals)
        target_wav = vad_wav

    out_prefix = Path(workdir) / "whisper_out"
    cmd = [
        str(whisper_exe),
        "-m", model_path,
        "-f", str(target_wav),
        "-l", language or "auto",
        "-mc", "0",
        "-sns",
        "-ojf",
        "-of", str(out_prefix.resolve()),
        "-t", str(N_THREADS),
    ]
    _emit(on_progress, 0.55, None, "Субтитры: распознавание речи (Whisper)...")
    try:
        _run_subprocess(cmd, cwd=whisper_exe.parent, log_name="whisper.log", stop_event=stop_event)
    finally:
        if vad_wav is not None:
            vad_wav.unlink(missing_ok=True)

    segments = _parse_whisper_json(Path(f"{out_prefix}.json"))
    _emit(on_progress, 0.95, 0.95, "Субтитры: обработка сегментов...")
    if vad_intervals is not None:
        segments = _clamp_segments_to_vad(segments, vad_intervals)
    return segments


def get_segments(
    input_video,
    workdir,
    model_file=WHISPER_MODEL_FILE,
    vad_model_file=WHISPER_VAD_MODEL_FILE,
    language=None,
    stop_event=None,
    on_progress=None,
):
    cache = workdir / "segments.json"
    if cache.exists():
        _emit(on_progress, 1.0, 1.0, "Субтитры загружены из кэша")
        return json.loads(cache.read_text(encoding="utf-8"))

    wav_path = workdir / "audio_16k.wav"
    if not wav_path.exists():
        _emit(on_progress, 0.05, None, "Субтитры: извлечение аудио (ffmpeg)...")
        extract_audio(input_video, str(wav_path))

    segments = transcribe(
        wav_path,
        workdir,
        model_file=model_file,
        vad_model_file=vad_model_file,
        language=language,
        stop_event=stop_event,
        on_progress=on_progress,
    )
    cache.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    _emit(on_progress, 1.0, 1.0, "Субтитры готовы")
    return segments


def write_srt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, 1):
            spk = (s.get("speaker") or "").strip()
            line = f"{spk}: {s['text']}" if spk else s["text"]
            f.write(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{line}\n\n")


EVENT_FORMAT = (
    "Output one event per line in exactly this format:\n"
    "FIRST-LAST | description\n"
    "where FIRST and LAST are ONLY the integer subtitle line numbers (NOT timestamps) of the first and last "
    "subtitle line of the event. Events follow the order of the lines and do not overlap. "
    "Never repeat the same line range twice. Output nothing except these lines."
)

PASS_SYSTEM = (
    "You are reading a movie's subtitles one part at a time. Each subtitle has a [HH:MM:SS] timestamp and speaker "
    "label (when detected), and long silences are marked. Use speaker labels, context, and time gaps to understand "
    "who speaks, and use character names only if supported by the dialogue or summary.\n"
    "You are given a summary of everything that happens {side} the current part. "
    "Describe ONLY what happens in the current part — NEVER repeat or copy events from the {side} summary.\n"
    "Distinguish living human speakers from pets/animals or deceased figures (never assign dialogue to a pet or dead character).\n"
    "Summarize the real events (conflict, decisions, location, phone call or flashback if clear) instead of "
    "repetitive 'X asks, Y answers' stenography. Do not invent facts. "
    "Spell character names consistently with the summary.\n"
    "Write in {language}."
)

FINAL_SYSTEM = (
    "You describe one part of a movie from its numbered subtitles (`N [HH:MM:SS] Speaker: text`). The subtitles have "
    "timestamps, speaker labels, and silence markers, so use them to track who speaks and spot scene/location/time jumps.\n"
    "You are given summaries of what happens BEFORE and AFTER this part so you know who is who and what the lines refer to. "
    "Describe ONLY what actually happens in the current part's subtitles — NEVER copy or repeat scenes from BEFORE/AFTER.\n"
    "Split the part into whole scenes: one event is one continuous conversation or situation in one place with the same participants. "
    "Do NOT split one conversation into small fragments; if the whole part is one scene, output 1 event "
    "(usually {count} events per part; start a new one only when place, time, or participants clearly change).\n"
    "Each event is {sentences} sentences. Summarize the substance (key decisions, revelations, objects, location, "
    "phone call or flashback if clear) instead of mechanical 'X asks..., Y answers...' chains. "
    "Clearly distinguish who is present in the scene vs who is only mentioned.\n"
    "Use consistent character names from BEFORE/AFTER and speaker labels; never assign "
    "dialogue to pets/animals or deceased figures.\n"
    "Write the descriptions in {language}.\n\n"
) + EVENT_FORMAT

REVIEW_SYSTEM = (
    "You are an editor checking a draft description of one part of a movie against its numbered subtitles (`N [HH:MM:SS]: text`).\n"
    "Go through the draft carefully:\n"
    "- every fact must be directly supported by the current part's subtitles; delete invented facts, ASR garbage, "
    "and any repeated/copied events that do not happen in these lines;\n"
    "- fix speaker mistakes (especially pets/animals or deceased figures treated as speakers) and unify name spelling;\n"
    "- merge neighbouring events that belong to the same continuous conversation or scene, and split where timestamps show a scene jump;\n"
    "- rewrite mechanical 'X asks..., Y answers...' chains into a concise, natural scene summary;\n"
    "- keep clear location, phone call / flashback cues, and distinguish participants from mentioned names;\n"
    "- fix line numbers so each event covers its subtitle lines without duplicates.\n"
    "Output the full corrected list of events of the current part. "
    "Write the descriptions in {language}.\n\n"
) + EVENT_FORMAT

COMPRESS_SYSTEM = (
    "You shorten story summaries. Keep the main events in chronological order, consistent character names "
    "(keeping clear who is human vs a pet/animal), and cause-and-effect links. Drop minor details. Write in {language}."
)

NO_PAST = "(this is the beginning of the film)"
NO_FUTURE = "(this is the end of the film)"

EVENT_RE = re.compile(r"^\s*[-*•]?\s*(\d+)(?!\s*:)\s*(?:[-–—]\s*(\d+)(?!\s*:))?\s*\|\s*(.+)$")

def split_chunks(segments, max_chars):
    chunks, first, size = [], 0, 0
    for i, s in enumerate(segments):
        if i > first and size + len(s["text"]) > max_chars:
            chunks.append((first, i - 1))
            first, size = i, 0
        size += len(s["text"])
    if segments:
        chunks.append((first, len(segments) - 1))
    return chunks


def _format_sub_lines(segments, a, b, with_numbers: bool) -> str:
    out = []
    for i in range(a, b + 1):
        if i > a:
            gap = segments[i]["start"] - segments[i - 1]["end"]
            if gap >= 15.0:
                out.append(f"[... пауза {int(round(gap))} сек ...]")
        stamp = ts(segments[i]["start"])[:8]
        spk = (segments[i].get("speaker") or "").strip()
        body = f"{spk}: {segments[i]['text']}" if spk else segments[i]["text"]
        if with_numbers:
            out.append(f"{i} [{stamp}]: {body}")
        else:
            out.append(f"[{stamp}] {body}")
    return "\n".join(out)


def plain_text(segments, a, b):
    return _format_sub_lines(segments, a, b, with_numbers=False)


def numbered_text(segments, a, b):
    return _format_sub_lines(segments, a, b, with_numbers=True)


def ask(llm, system, user, max_tokens):
    return chat(
        llm,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# События: разбор ответа модели и вывод обратно в текст
# ---------------------------------------------------------------------------

def _norm_event_text(text: str) -> str:
    return " ".join(re.sub(r"\W+", " ", text.lower()).split())


def parse_events(answer, first_line, last_line):
    """Ответ вида '12-18 | описание' -> [(первая реплика, последняя реплика, текст)].
    Номера зажимаем в границы куска и отсекаем дубли при зацикливании модели."""
    raw_events = []
    for line in answer.splitlines():
        line = line.strip()
        m = EVENT_RE.match(line)
        if m:
            a = int(m.group(1))
            b = int(m.group(2) or a)
            a, b = sorted((a, b))
            a = min(max(a, first_line), last_line)
            b = min(max(b, first_line), last_line)
            raw_events.append([a, b, m.group(3).strip()])
        elif raw_events and line:                             # модель перенесла описание на новую строку
            raw_events[-1][2] += " " + line

    deduped = []
    for a, b, text in raw_events:
        if not text:
            continue
        norm = _norm_event_text(text)
        if not norm:
            continue
        if deduped:
            pa, pb, ptext = deduped[-1]
            pnorm = _norm_event_text(ptext)
            if pnorm == norm or ((pa, pb) == (a, b) and (pnorm.startswith(norm) or norm.startswith(pnorm))):
                if len(text) > len(ptext):
                    deduped[-1] = (a, b, text)
                continue
        if any(_norm_event_text(et) == norm and (ea, eb) == (a, b) for ea, eb, et in deduped):
            continue
        deduped.append((a, b, text))
    return deduped


def format_events(events):
    return "\n".join(f"{a}-{b} | {text}" for a, b, text in events)


def plain_events(events):
    return "\n".join(f"- {text}" for _, _, text in events) if events else "(none)"


# ---------------------------------------------------------------------------
# Память модели: ближние сцены дословно, дальние - сжато
# ---------------------------------------------------------------------------

def context_text(far, recent, backward):
    """Собираем память в порядке фильма. Вперёд: далёкое прошлое + недавнее. Назад: ближнее будущее + далёкое."""
    parts = recent + [far] if backward else [far] + recent
    return "\n".join(p for p in parts if p).strip()


def remember(llm, far, recent, note, language, backward):
    """Новая заметка идёт в "недавние" без сжатия. Когда их больше RECENT_NOTES,
    самая далёкая от текущего места уходит в "далёкое", и только оно сжимается."""
    if backward:
        recent.insert(0, note)                                # заметки лежат в порядке фильма
        old = recent.pop() if len(recent) > RECENT_NOTES else None
    else:
        recent.append(note)
        old = recent.pop(0) if len(recent) > RECENT_NOTES else None

    if old:
        far = f"{old}\n{far}" if backward else f"{far}\n{old}"
        far = far.strip()
        if len(far) > MAX_FAR_CHARS:
            far = ask(llm, COMPRESS_SYSTEM.format(language=language),
                      f"Shorten to at most {FAR_WORDS} words:\n\n{far}", max_tokens=600)
    return far


# ---------------------------------------------------------------------------
# Проходы 1-2: вперёд (что было) и назад (что будет)
# ---------------------------------------------------------------------------

def read_pass(llm, segments, chunks, language, backward, on_progress=None, stop_event=None):
    """Один проход по фильму целиком.

    Возвращает для каждого куска память с той стороны, откуда мы к нему пришли:
    при проходе вперёд это "что было до него", при проходе назад - "что будет после".
    """
    order = range(len(chunks) - 1, -1, -1) if backward else range(len(chunks))
    side = "AFTER" if backward else "BEFORE"
    empty = NO_FUTURE if backward else NO_PAST
    total = len(chunks)

    far, recent, known = "", [], {}
    for step, i in enumerate(order, 1):
        _check_stop(stop_event)
        memory = context_text(far, recent, backward)
        known[i] = memory                                     # что мы знали, когда дошли до этого куска
        print(f"{'Назад' if backward else 'Вперёд'}: кусок {step}/{total}")
        if on_progress is not None:
            on_progress((step - 1) / total, step, total)
        a, b = chunks[i]
        note = ask(
            llm, PASS_SYSTEM.format(side=side, language=language),
            f"Summary of what happens {side.lower()} this part:\n{memory or empty}\n\n"
            f"Subtitles of the current part:\n{plain_text(segments, a, b)}\n\n"
            f"Describe in {NOTE_SENTENCES} sentences what happens in the current part.",
            max_tokens=600)
        far = remember(llm, far, recent, note, language, backward)
        if on_progress is not None:
            on_progress(step / total, step, total)
    return known


# ---------------------------------------------------------------------------
# Проход 3: события с точными границами. Проход 4: проверка и правка
# ---------------------------------------------------------------------------

def write_events(llm, segments, chunks, before, after, language, on_progress=None, stop_event=None):
    drafts = []
    total = len(chunks)
    for i, (a, b) in enumerate(chunks):
        _check_stop(stop_event)
        step = i + 1
        print(f"События: кусок {step}/{total}")
        if on_progress is not None:
            on_progress(i / total, step, total)
        user_prompt = (
            f"Summary of the film BEFORE this part:\n{before[i] or NO_PAST}\n\n"
            f"Summary of the film AFTER this part:\n{after[i] or NO_FUTURE}\n\n"
            f"Subtitles of the current part (each line starts with its number and timestamp):\n"
            f"{numbered_text(segments, a, b)}\n\n"
            "Split the current part into events."
        )
        answer = ask(
            llm,
            FINAL_SYSTEM.format(language=language, sentences=EVENT_SENTENCES, count=EVENT_COUNT),
            user_prompt,
            max_tokens=1500,
        )
        drafts.append(parse_events(answer, a, b) or [(a, b, " ".join(dict.fromkeys(l.strip() for l in answer.splitlines() if l.strip())))])
        if on_progress is not None:
            on_progress(step / total, step, total)
    return drafts


def review_events(llm, segments, chunks, drafts, language, on_progress=None, stop_event=None):
    """Редактор идёт по сюжету подряд. Предыдущий кусок уже исправлен, следующий пока черновик."""
    checked = []
    total = len(chunks)
    for i, (a, b) in enumerate(chunks):
        _check_stop(stop_event)
        step = i + 1
        print(f"Проверка: кусок {step}/{total}")
        if on_progress is not None:
            on_progress(i / total, step, total)
        before = plain_events(checked[-1]) if checked else "(none)"
        after = plain_events(drafts[i + 1]) if i + 1 < len(drafts) else "(none)"
        user_prompt = (
            f"Events before this part (already checked):\n{before}\n\n"
            f"Events after this part (draft):\n{after}\n\n"
            f"Subtitles of the current part (each line starts with its number and timestamp):\n"
            f"{numbered_text(segments, a, b)}\n\n"
            f"Draft events of the current part:\n{format_events(drafts[i])}\n\n"
            "Write the corrected events of the current part."
        )
        answer = ask(
            llm,
            REVIEW_SYSTEM.format(language=language),
            user_prompt,
            max_tokens=1800,
        )
        checked.append(parse_events(answer, a, b) or drafts[i])         # ответ не разобрался -> оставляем черновик
        if on_progress is not None:
            on_progress(step / total, step, total)
    return checked


def same_scene(llm, first, second):
    answer = ask(
        llm,
        "You decide whether two consecutive descriptions of a movie belong to the same continuous scene: "
        "the same place, the same conversation or situation, no jump in time. Answer YES or NO.",
        f"First description:\n{first}\n\nSecond description:\n{second}\n\nIs it the same scene?",
        max_tokens=4)
    up = answer.strip().upper()
    return up.startswith("YES") or up.startswith("ДА")


def merge_across_chunks(llm, events_by_chunk, language, on_progress=None, stop_event=None):
    merged = []
    total = max(1, len(events_by_chunk) - 1)
    for idx, chunk_events in enumerate(events_by_chunk):
        _check_stop(stop_event)
        if not chunk_events:
            continue
        if merged:
            step = min(idx, total)
            if on_progress is not None:
                on_progress((step - 1) / total, step, total)
            if same_scene(llm, merged[-1][2], chunk_events[0][2]):
                first, _, text = merged[-1]
                _, last, next_text = chunk_events[0]
                merged[-1] = (first, last, f"{text} {next_text}")
                merged.extend(chunk_events[1:])
            else:
                merged.extend(chunk_events)
            if on_progress is not None:
                on_progress(step / total, step, total)
        else:
            merged.extend(chunk_events)
    return merged


def build_plot(segments, llm_loader, language, review=True, on_progress=None, stop_event=None, llm=None):
    chunks = split_chunks(segments, CHUNK_CHARS)
    _check_stop(stop_event)
    owns_llm = llm is None
    if owns_llm:
        _emit(on_progress, 0.0, None, "Сюжет: загрузка языковой модели...")
        llm = llm_loader()
    passes = 4 if review else 3
    pass_span = 0.94 / passes

    def make_pass_cb(pass_num, pass_title):
        base = 0.02 + (pass_num - 1) * pass_span

        def _cb(frac, step, total):
            _emit(
                on_progress,
                base + frac * pass_span,
                frac,
                f"Сюжет ({pass_num}/{passes} • {pass_title}): кусок {step}/{total}",
            )

        return _cb

    def _merge_cb(frac, step, total):
        _emit(
            on_progress,
            0.96 + frac * 0.04,
            frac,
            f"Сюжет: склейка границ ({step}/{total})",
        )

    sorted_plot = []
    try:
        before = read_pass(
            llm, segments, chunks, language, backward=False,
            on_progress=make_pass_cb(1, "Вперёд") if on_progress is not None else None,
            stop_event=stop_event,
        )
        after = read_pass(
            llm, segments, chunks, language, backward=True,
            on_progress=make_pass_cb(2, "Назад") if on_progress is not None else None,
            stop_event=stop_event,
        )
        events = write_events(
            llm, segments, chunks, before, after, language,
            on_progress=make_pass_cb(3, "События") if on_progress is not None else None,
            stop_event=stop_event,
        )
        if review:
            events = review_events(
                llm, segments, chunks, events, language,
                on_progress=make_pass_cb(4, "Проверка") if on_progress is not None else None,
                stop_event=stop_event,
            )
        events = merge_across_chunks(
            llm, events, language,
            on_progress=_merge_cb if on_progress is not None else None,
            stop_event=stop_event,
        )
        plot = [(segments[a]["start"], segments[b]["end"], text) for a, b, text in events]
        sorted_plot = sorted(plot, key=lambda e: e[0])
    finally:
        if owns_llm:
            close_model(llm)
            del llm; free()

    _emit(on_progress, 1.0, 1.0, "Сюжет готов")
    return sorted_plot


def write_plot(events, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Сюжет\n\n")
        for start, end, text in events:
            f.write(f"[{ts(start)[:8]} - {ts(end)[:8]}] {text}\n\n")


def generate_subtitles(input_video, output_srt, output_plot=None, llm_loader=None,
                       model_name=None, language=None, batch_size=4, detect_sec=120,
                       plot_language="Russian", review=True, stop_event=None,
                       on_progress=None):
    workdir = Path("work") / Path(input_video).stem
    workdir.mkdir(parents=True, exist_ok=True)

    has_plot = llm_loader is not None and output_plot is not None
    subs_span = 0.20 if has_plot else 0.70
    diar_span = 0.15 if has_plot else 0.30

    def _subs_cb(stage_frac, step_frac, text):
        _emit(on_progress, stage_frac * subs_span, step_frac, text)

    model_file = model_name or WHISPER_MODEL_FILE
    segments = get_segments(
        input_video,
        workdir,
        model_file=model_file,
        language=language,
        stop_event=stop_event,
        on_progress=_subs_cb if on_progress is not None else None,
    )

    needs_diar = bool(segments) and not all(s.get("speaker") for s in segments)
    llm = None
    try:
        if (needs_diar or has_plot) and llm_loader is not None:
            _check_stop(stop_event)
            _emit(on_progress, subs_span, None, "Загрузка языковой модели для диаризации и сюжета...")
            llm = llm_loader()

        if needs_diar:
            def _diar_cb(frac, text):
                _emit(on_progress, subs_span + frac * diar_span, frac, text)

            segments = diarize_segments(
                input_video,
                workdir,
                segments,
                llm=llm,
                language=plot_language,
                fallback_wav=workdir / "audio_16k.wav",
                stop_event=stop_event,
                on_progress=_diar_cb if on_progress is not None else None,
            )
            (workdir / "segments.json").write_text(
                json.dumps(segments, ensure_ascii=False), encoding="utf-8"
            )

        write_srt(segments, output_srt)

        if has_plot:
            plot_start = subs_span + diar_span
            plot_span = 1.0 - plot_start

            def _plot_cb(stage_frac, step_frac, text):
                _emit(on_progress, plot_start + stage_frac * plot_span, step_frac, text)

            events = build_plot(
                segments,
                llm_loader,
                plot_language,
                review,
                on_progress=_plot_cb if on_progress is not None else None,
                stop_event=stop_event,
                llm=llm,
            )
            write_plot(events, output_plot)
    finally:
        if llm is not None:
            close_model(llm)
            del llm; free()

    return output_srt