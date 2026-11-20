import os
import subprocess
import numpy as np
from pathlib import Path

SR = 48000

def parse_name(name):
    _, start, end = Path(name).stem.rsplit("_", 2)
    return float(start), float(end)

def decode_voice(path):
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16)

def add_sound(video_path, duck, pad, output_path):
    files = sorted(f for f in os.listdir("voices") if f.endswith(".mp3"))

    # 1) оригинальная дорожка -> raw s16 stereo 48k
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video_path,
         "-map", "0:a:0", "-f", "s16le", "-ac", "2", "-ar", str(SR), "orig.raw"],
        check=True)

    # memmap: правим файл на диске, не грузя всё в память
    audio = np.memmap("orig.raw", dtype=np.int16, mode="r+").reshape(-1, 2)
    n = len(audio)

    # 2) приглушаем интервалы
    for f in files:
        start, end = parse_name(f)
        i, j = int(start * SR), min(int(end * SR), n)
        audio[i:j] = (audio[i:j].astype(np.float32) * duck).astype(np.int16)

    # 3) подмешиваем голоса
    for f in files:
        start, _ = parse_name(f)
        seg = decode_voice(f"voices/{f}")
        i = int(start * SR)
        k = min(i + len(seg), n)
        seg = seg[: k - i]
        mixed = audio[i:k].astype(np.int32) + seg[:, None]
        audio[i:k] = np.clip(mixed, -32768, 32767).astype(np.int16)

    audio.flush()
    del audio

    # 4) муксинг: видео и субтитры копируем, аудио кодируем в AAC
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
        check=True)

    os.remove("orig.raw")