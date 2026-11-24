# Временная локальная заглушка озвучки на базе Piper TTS (вместо нестабильного edge_tts).
# Полностью сохраняет внешний интерфейс async def speak(text, target_sec, out, lang="ru", ...).

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import wave
import zipfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

from config import FFMPEG_DIR, PIPER_DIR, PIPER_REPO_ID

PIPER_RELEASE_URL = (
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
)

# Пути к моделям внутри репозитория rhasspy/piper-voices (для русского можно сменить irina на dmitri/denis/ruslan)
VOICES = {
    "ru": "ru/ru_RU/irina/medium/ru_RU-irina-medium",
    "en": "en/en_US/lessac/medium/en_US-lessac-medium",
    "de": "de/de_DE/thorsten/medium/de_DE-thorsten-medium",
    "fr": "fr/fr_FR/siwis/medium/fr_FR-siwis-medium",
    "es": "es/es_ES/davefx/medium/es_ES-davefx-medium",
    "it": "it/it_IT/paola/medium/it_IT-paola-medium",
    "pt": "pt/pt_BR/faber/medium/pt_BR-faber-medium",
    "pl": "pl/pl_PL/gosia/medium/pl_PL-gosia-medium",
    "uk": "uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium",
    "tr": "tr/tr_TR/dfki/medium/tr_TR-dfki-medium",
    "nl": "nl/nl_NL/mls/medium/nl_NL-mls-medium",
}

_CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\ufeff\xad]")
_MD_RE = re.compile(r"[*_`#]+")
_SPEAKABLE_RE = re.compile(r"[^\W_]")
_VOICE_CACHE: dict[str, tuple[str, str]] = {}


def _clean_text(text: str) -> str:
    cleaned = _CTRL_RE.sub(" ", text or "")
    cleaned = _MD_RE.sub("", cleaned)
    cleaned = (
        cleaned.replace("—", "-")
        .replace("–", "-")
        .replace("«", '"')
        .replace("»", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
        .replace("…", "...")
    )
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        raise ValueError("Пустой текст")
    if not _SPEAKABLE_RE.search(cleaned):
        raise ValueError("Текст не содержит произносимых символов")
    return cleaned


def _ffmpeg_bin() -> str:
    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidate = Path(FFMPEG_DIR) / exe_name
    return str(candidate) if candidate.is_file() else "ffmpeg"


def _ensure_piper_exe() -> Path:
    piper_dir = Path(PIPER_DIR)
    exe_path = piper_dir / ("piper.exe" if os.name == "nt" else "piper")
    if exe_path.is_file():
        return exe_path

    if os.name != "nt":
        found = shutil.which("piper")
        if found:
            return Path(found)
        raise FileNotFoundError(f"Не найден бинарник piper по пути: {exe_path}")

    print(f"Скачиваю Piper TTS в {piper_dir}...", flush=True)
    piper_dir.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    try:
        with tempfile.TemporaryDirectory() as td:
            zip_path = Path(td) / "piper.zip"
            with session.get(PIPER_RELEASE_URL, stream=True, timeout=(15, 120)) as resp:
                resp.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            extract_dir = Path(td) / "extracted"
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            src_dir = extract_dir / "piper"
            if not src_dir.is_dir():
                src_dir = extract_dir
            if piper_dir.exists():
                shutil.rmtree(piper_dir)
            shutil.move(str(src_dir), str(piper_dir))
    finally:
        session.close()

    if not exe_path.is_file():
        raise FileNotFoundError(f"Не удалось установить piper.exe в {piper_dir}")
    return exe_path


def _download_voice_file(filename: str) -> str:
    try:
        return hf_hub_download(repo_id=PIPER_REPO_ID, filename=filename, local_files_only=True)
    except Exception:
        return hf_hub_download(repo_id=PIPER_REPO_ID, filename=filename)


def _ensure_voice_model(lang: str) -> tuple[str, str]:
    voice_rel = VOICES.get(lang) or VOICES["ru"]
    if voice_rel in _VOICE_CACHE:
        onnx_p, json_p = _VOICE_CACHE[voice_rel]
        if os.path.isfile(onnx_p) and os.path.isfile(json_p):
            return onnx_p, json_p

    onnx_path = _download_voice_file(f"{voice_rel}.onnx")
    json_path = _download_voice_file(f"{voice_rel}.onnx.json")
    _VOICE_CACHE[voice_rel] = (onnx_path, json_path)
    return onnx_path, json_path


def _wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate) if rate > 0 else 0.0


def _build_atempo_filter(speed: float) -> str:
    """Строит цепочку фильтров atempo (каждый в безопасном диапазоне 0.5..2.0)."""
    factors = []
    rem = float(speed)
    while rem > 2.0 + 1e-6:
        factors.append(2.0)
        rem /= 2.0
    while rem < 0.5 - 1e-6:
        factors.append(0.5)
        rem /= 0.5
    factors.append(rem)
    return ",".join(f"atempo={f:.5f}" for f in factors)


def _speak_sync(text: str, target_sec: float, out: str, lang: str = "ru", tolerance: float = 0.05) -> float:
    cleaned = _clean_text(text)
    if target_sec <= 0:
        raise ValueError(f"Некорректный target_sec: {target_sec}")

    piper_exe = _ensure_piper_exe()
    onnx_path, json_path = _ensure_voice_model(lang)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = str(out_path) + ".tmp"

    with tempfile.TemporaryDirectory() as td:
        raw_wav = os.path.join(td, "piper_raw.wav")
        cmd_piper = [
            str(piper_exe),
            "-m", onnx_path,
            "-c", json_path,
            "-f", raw_wav,
            "--sentence-silence", "0.15",
        ]
        res = subprocess.run(
            cmd_piper,
            input=(cleaned + "\n").encode("utf-8"),
            capture_output=True,
            cwd=str(piper_exe.parent),
            creationflags=_CREATE_NO_WINDOW,
        )
        if res.returncode != 0 or not os.path.isfile(raw_wav) or os.path.getsize(raw_wav) == 0:
            err = res.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Ошибка синтеза Piper (код {res.returncode}): {err}")

        raw_dur = _wav_duration(raw_wav)
        goal = max(0.15, target_sec * (1.0 - tolerance))

        # За один проход точно укладываем длительность в target_sec через ffmpeg atempo (если фраза длиннее окна)
        af_args = []
        final_dur = raw_dur
        if raw_dur > target_sec and raw_dur > 0:
            speed = min(2.5, raw_dur / goal)
            af_args = ["-af", _build_atempo_filter(speed)]
            final_dur = raw_dur / speed

        ext = out_path.suffix.lower()
        codec_args = ["-c:a", "libmp3lame", "-q:a", "3"] if ext == ".mp3" else []
        cmd_ff = [
            _ffmpeg_bin(),
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", raw_wav,
            *af_args,
            *codec_args,
            "-f", "mp3" if ext == ".mp3" else "wav",
            tmp_out,
        ]
        ff_res = subprocess.run(
            cmd_ff,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_CREATE_NO_WINDOW,
        )
        if ff_res.returncode != 0 or not os.path.isfile(tmp_out) or os.path.getsize(tmp_out) == 0:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
            raise RuntimeError(f"Ошибка конвертации озвучки ffmpeg: {(ff_res.stderr or '').strip()}")

        os.replace(tmp_out, out_path)

    if final_dur > target_sec + 0.05:
        print(f"[warn] {out}: {final_dur:.2f}s > {target_sec:.2f}s (даже при ускорении 2.5x)")
    return final_dur


async def speak(text, target_sec, out, lang="ru", tolerance=0.05, max_iter=3):
    return await asyncio.to_thread(_speak_sync, text, target_sec, out, lang, tolerance)
