import gc, json, re, shutil, subprocess, sys
from pathlib import Path

import torch
import whisperx

from models import chat, close_model

SR = 16000             

CHUNK_CHARS = 4000       
NOTE_SENTENCES = "5-8"   
EVENT_SENTENCES = "3-6" 
EVENT_COUNT = "1-3"      
RECENT_NOTES = 3       
MAX_FAR_CHARS = 2500     
FAR_WORDS = 300          


def free():
    gc.collect()
    torch.cuda.empty_cache()


def extract_audio(input_video, output_audio, track=0, sample_rate=16000, mono=True):
    Path(output_audio).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", input_video, "-map", f"0:a:{track}", "-vn",
           "-acodec", "pcm_s16le", "-ar", str(sample_rate)]
    if mono:
        cmd += ["-ac", "1"]
    cmd.append(output_audio)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка ffmpeg:\n{result.stderr}")
    print(f"Аудио сохранено в {output_audio}")
    return output_audio


def ts(t):
    h, rem = divmod(int(t * 1000), 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def extract_vocals(video, workdir, chunk_sec=600):
    final = workdir / "vocals_16k.wav"
    if final.exists():                                        
        return str(final)

    chunks = workdir / "chunks"
    shutil.rmtree(chunks, ignore_errors=True)                 
    chunks.mkdir()
    subprocess.run(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "2", "-ar", "44100",
                    "-f", "segment", "-segment_time", str(chunk_sec), "-c:a", "pcm_s16le",
                    str(chunks / "c_%03d.wav")], check=True)
    files = sorted(chunks.glob("c_*.wav"))

    subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", str(workdir),
                    *map(str, files)], check=True)            

    vocal_files = [(workdir / "htdemucs" / f.stem / "vocals.wav").resolve().as_posix() for f in files]
    lst = workdir / "list.txt"
    lst.write_text("".join(f"file '{v}'\n" for v in vocal_files), encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-ac", "1", "-ar", "16000", str(final)], check=True)
    return str(final)

def detect_language(model, audio, detect_sec):
    start = max(0, (len(audio) / SR - detect_sec) / 2)       
    chunk = audio[int(start * SR):int((start + detect_sec) * SR)]
    lang, prob, _ = model.model.detect_language(chunk, vad_filter=True,  
                                                language_detection_segments=max(1, int(detect_sec // 30)))
    print(f"Язык: {lang} ({prob:.2f}), кусок {start:.0f}-{start + detect_sec:.0f} с")
    return lang

def transcribe(audio, device, model_name, language, batch_size, detect_sec):
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model(model_name, device, compute_type=compute_type, vad_method="silero")
    if language is None:
        language = detect_language(model, audio, detect_sec)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    del model; free()

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    del model_a; free()

    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in result["segments"]]


def get_segments(input_video, workdir, device, model_name, language, batch_size, detect_sec):
    cache = workdir / "segments.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    vocals = extract_vocals(input_video, workdir)
    audio = whisperx.load_audio(vocals)                      
    segments = transcribe(audio, device, model_name, language, batch_size, detect_sec)
    cache.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    return segments


def write_srt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, 1):
            f.write(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n\n")


EVENT_FORMAT = (
    "Output one event per line in exactly this format:\n"
    "FIRST-LAST | description\n"
    "where FIRST and LAST are the numbers of the first and last subtitle line of the event. "
    "Events follow the order of the lines and do not overlap. "
    "Output nothing except these lines."
)

PASS_SYSTEM = (
    "You are reading a movie's subtitles one part at a time. The subtitles have no speaker labels and no "
    "scene descriptions: work out who speaks from the context, and use character names only if they "
    "appear in the dialogue.\n"
    "You are given a summary of everything that happens {side} the current part. "
    "Describe ONLY what happens in the current part. "
    "Do not invent facts that the text does not support.\n"
    "Spell character names exactly as they are spelled in the summary.\n"
    "Write in {language}."
)

FINAL_SYSTEM = (
    "You describe one part of a movie from its subtitles. The subtitles have no speaker labels and no "
    "scene descriptions.\n"
    "You are given summaries of what happens BEFORE and AFTER this part, so you know both the past and "
    "the future of the story. Use them to understand who is who and what the lines refer to. "
    "Describe ONLY what happens in the current part, in the order it happens.\n"
    "Split the part into events. An event is a whole scene: one continuous conversation or situation "
    "in one place with the same participants. Do NOT split one conversation into several small events. "
    "A part usually contains {count} events; start a new one only when the scene, the place or the "
    "participants really change.\n"
    "Each event is {sentences} sentences. Be detailed inside the event: mention every important "
    "decision and revelation, and what exactly was said or done. Do not skip details to save space.\n"
    "Use character names only if they appear in the dialogue, and spell them exactly as in the summaries. "
    "If something is unclear from the text, say that it is unclear instead of guessing.\n"
    "Write the descriptions in {language}.\n\n"
) + EVENT_FORMAT

REVIEW_SYSTEM = (
    "You are an editor. You check a draft description of one part of a movie against its subtitles. "
    "The subtitles have no speaker labels.\n"
    "Go through the draft event by event:\n"
    "- every fact must be supported by the subtitles; fix or remove facts that are wrong or invented;\n"
    "- add important events from the subtitles that the draft missed;\n"
    "- split an event only if it mixes different scenes or places; never split one continuous "
    "conversation into small events, and merge neighbouring events that belong to the same scene;\n"
    "- fix the line numbers so that each event covers exactly the lines where it happens;\n"
    "- make character names and facts consistent with the neighbouring events;\n"
    "- if something is unclear from the text, say that it is unclear instead of guessing.\n"
    "Output the full corrected list of events of the current part. "
    "Write the descriptions in {language}.\n\n"
) + EVENT_FORMAT

COMPRESS_SYSTEM = (
    "You shorten story summaries. Keep the main events in chronological order, the names, and the "
    "cause-and-effect links. Drop minor details. Write in {language}."
)

NO_PAST = "(this is the beginning of the film)"
NO_FUTURE = "(this is the end of the film)"

EVENT_RE = re.compile(r"^[^\w|]*(\d+)\s*(?:[-–—]\s*(\d+))?[^\w|]*\|\s*(.+)$")

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


def plain_text(segments, a, b):
    return "\n".join(segments[i]["text"] for i in range(a, b + 1))


def numbered_text(segments, a, b):
    return "\n".join(f"{i}: {segments[i]['text']}" for i in range(a, b + 1))


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

def parse_events(answer, first_line, last_line):
    """Ответ вида '12-18 | описание' -> [(первая реплика, последняя реплика, текст)].
    Номера зажимаем в границы куска. Если ни одной строки не распознано, вернётся пустой список."""
    events = []
    for line in answer.splitlines():
        line = line.strip()
        m = EVENT_RE.match(line)
        if m:
            a = int(m.group(1))
            b = int(m.group(2) or a)
            a, b = sorted((a, b))
            a = min(max(a, first_line), last_line)
            b = min(max(b, first_line), last_line)
            events.append([a, b, m.group(3).strip()])
        elif events and line:                                 # модель перенесла описание на новую строку
            events[-1][2] += " " + line
    return [tuple(e) for e in events]


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

def read_pass(llm, segments, chunks, language, backward):
    """Один проход по фильму целиком.

    Возвращает для каждого куска память с той стороны, откуда мы к нему пришли:
    при проходе вперёд это "что было до него", при проходе назад - "что будет после".
    """
    order = range(len(chunks) - 1, -1, -1) if backward else range(len(chunks))
    side = "AFTER" if backward else "BEFORE"
    empty = NO_FUTURE if backward else NO_PAST

    far, recent, known = "", [], {}
    for step, i in enumerate(order, 1):
        memory = context_text(far, recent, backward)
        known[i] = memory                                     # что мы знали, когда дошли до этого куска
        print(f"{'Назад' if backward else 'Вперёд'}: кусок {step}/{len(chunks)}")
        a, b = chunks[i]
        note = ask(
            llm, PASS_SYSTEM.format(side=side, language=language),
            f"Summary of what happens {side.lower()} this part:\n{memory or empty}\n\n"
            f"Subtitles of the current part:\n{plain_text(segments, a, b)}\n\n"
            f"Describe in {NOTE_SENTENCES} sentences what happens in the current part.",
            max_tokens=600)
        far = remember(llm, far, recent, note, language, backward)
    return known


# ---------------------------------------------------------------------------
# Проход 3: события с точными границами. Проход 4: проверка и правка
# ---------------------------------------------------------------------------

def write_events(llm, segments, chunks, before, after, language):
    drafts = []
    for i, (a, b) in enumerate(chunks):
        print(f"События: кусок {i + 1}/{len(chunks)}")
        answer = ask(
            llm, FINAL_SYSTEM.format(language=language, sentences=EVENT_SENTENCES, count=EVENT_COUNT),
            f"Summary of the film BEFORE this part:\n{before[i] or NO_PAST}\n\n"
            f"Summary of the film AFTER this part:\n{after[i] or NO_FUTURE}\n\n"
            f"Subtitles of the current part (each line starts with its number):\n"
            f"{numbered_text(segments, a, b)}\n\n"
            "Split the current part into events.",
            max_tokens=1500)
        drafts.append(parse_events(answer, a, b) or [(a, b, answer)])   # формат не соблюдён -> один общий период
    return drafts


def review_events(llm, segments, chunks, drafts, language):
    """Редактор идёт по сюжету подряд. Предыдущий кусок уже исправлен, следующий пока черновик."""
    checked = []
    for i, (a, b) in enumerate(chunks):
        print(f"Проверка: кусок {i + 1}/{len(chunks)}")
        before = plain_events(checked[-1]) if checked else "(none)"
        after = plain_events(drafts[i + 1]) if i + 1 < len(drafts) else "(none)"
        answer = ask(
            llm, REVIEW_SYSTEM.format(language=language),
            f"Events before this part (already checked):\n{before}\n\n"
            f"Events after this part (draft):\n{after}\n\n"
            f"Subtitles of the current part (each line starts with its number):\n"
            f"{numbered_text(segments, a, b)}\n\n"
            f"Draft events of the current part:\n{format_events(drafts[i])}\n\n"
            "Write the corrected events of the current part.",
            max_tokens=1800)
        checked.append(parse_events(answer, a, b) or drafts[i])         # ответ не разобрался -> оставляем черновик
    return checked


def same_scene(llm, first, second):
    answer = ask(
        llm,
        "You decide whether two consecutive descriptions of a movie belong to the same continuous scene: "
        "the same place, the same conversation or situation, no jump in time. Answer YES or NO.",
        f"First description:\n{first}\n\nSecond description:\n{second}\n\nIs it the same scene?",
        max_tokens=4)
    return answer.upper().startswith("YES")


def merge_across_chunks(llm, events_by_chunk, language):
    merged = []
    for chunk_events in events_by_chunk:
        if not chunk_events:
            continue
        if merged and same_scene(llm, merged[-1][2], chunk_events[0][2]):
            first, _, text = merged[-1]
            _, last, next_text = chunk_events[0]
            merged[-1] = (first, last, f"{text} {next_text}")
            merged.extend(chunk_events[1:])
        else:
            merged.extend(chunk_events)
    return merged


def build_plot(segments, llm_loader, language, review=True):
    chunks = split_chunks(segments, CHUNK_CHARS)
    llm = llm_loader()                                        
    try:
        before = read_pass(llm, segments, chunks, language, backward=False)
        after = read_pass(llm, segments, chunks, language, backward=True)
        events = write_events(llm, segments, chunks, before, after, language)
        if review:
            events = review_events(llm, segments, chunks, events, language)
        events = merge_across_chunks(llm, events, language)   
    finally:
        close_model(llm)
        del llm; free()

    plot = [(segments[a]["start"], segments[b]["end"], text) for a, b, text in events]
    return sorted(plot, key=lambda e: e[0])


def write_plot(events, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Сюжет\n\n")
        for start, end, text in events:
            f.write(f"[{ts(start)[:8]} - {ts(end)[:8]}] {text}\n\n")

def generate_subtitles(input_video, output_srt, output_plot=None, llm_loader=None,
                       model_name="large-v3", language=None, batch_size=4, detect_sec=120,
                       plot_language="Russian", review=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    workdir = Path("work") / Path(input_video).stem           
    workdir.mkdir(parents=True, exist_ok=True)

    segments = get_segments(input_video, workdir, device, model_name, language, batch_size, detect_sec)
    write_srt(segments, output_srt)

    if llm_loader is not None and output_plot is not None:
        events = build_plot(segments, llm_loader, plot_language, review)
        write_plot(events, output_plot)
    return output_srt