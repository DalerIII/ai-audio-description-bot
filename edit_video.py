import os
import subprocess
import sys
import numpy as np
from pathlib import Path

SR = 48000
_CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if (os.name == "nt" and getattr(sys, "frozen", False)) else 0


def _ffmpeg_run_kwargs():
    if _CREATE_NO_WINDOW:
        return {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "creationflags": _CREATE_NO_WINDOW,
        }
    return {}


def parse_name(name):
    _, start, end = Path(name).stem.rsplit("_", 2)
    return float(start), float(end)


def decode_voice(path):
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"]
    raw = subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        stdin=subprocess.DEVNULL,
        creationflags=_CREATE_NO_WINDOW,
    ).stdout
    return np.frombuffer(raw, dtype=np.int16)


def add_sound(video_path, duck, pad, output_path, on_progress=None):
    def emit(stage_frac, step_frac, text):
        if on_progress is not None:
            on_progress(stage_frac, step_frac, text)

    files = sorted(f for f in os.listdir("voices") if f.endswith(".mp3"))
    total = len(files)

    # 1) оригинальная дорожка -> raw s16 stereo 48k
    emit(0.0, None, "Монтаж: извлечение аудиодорожки (ffmpeg)...")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video_path,
         "-map", "0:a:0", "-f", "s16le", "-ac", "2", "-ar", str(SR), "orig.raw"],
        check=True,
        **_ffmpeg_run_kwargs(),
    )

    # memmap: правим файл на диске, не грузя всё в память
    audio = np.memmap("orig.raw", dtype=np.int16, mode="r+").reshape(-1, 2)
    n = len(audio)

    # 2) приглушаем интервалы
    for i, f in enumerate(files, 1):
        start, end = parse_name(f)
        idx_a, idx_b = int(start * SR), min(int(end * SR), n)
        audio[idx_a:idx_b] = (audio[idx_a:idx_b].astype(np.float32) * duck).astype(np.int16)
        frac = i / total
        emit(0.25 + 0.15 * frac, frac, f"Монтаж: приглушение оригинала ({i}/{total})...")

    # 3) подмешиваем голоса
    for i, f in enumerate(files, 1):
        start, _ = parse_name(f)
        seg = decode_voice(f"voices/{f}")
        idx_a = int(start * SR)
        k = min(idx_a + len(seg), n)
        seg = seg[: k - idx_a]
        mixed = audio[idx_a:k].astype(np.int32) + seg[:, None]
        audio[idx_a:k] = np.clip(mixed, -32768, 32767).astype(np.int16)
        frac = i / total
        emit(0.40 + 0.25 * frac, frac, f"Монтаж: наложение озвучки ({i}/{total})...")

    audio.flush()
    del audio

    # 4) муксинг: видео и субтитры копируем, аудио кодируем в AAC
    emit(0.65, None, "Монтаж: финальная сборка видео (ffmpeg)...")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-stats",
         "-i", video_path,
         "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "orig.raw",
         "-map", "0:v:0", "-map", "1:a", "-map", "0:s?",
         "-c:v", "copy", "-c:s", "mov_text",
         "-c:a", "aac", "-b:a", "192k",
         "-metadata:s:a:0", "title=Voiceover mix",
         "-movflags", "+faststart",
         output_path],
        check=True,
        **_ffmpeg_run_kwargs(),
    )

    os.remove("orig.raw")
    emit(1.0, 1.0, "Монтаж завершён")