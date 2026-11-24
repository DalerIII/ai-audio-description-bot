from fractions import Fraction
from functools import lru_cache
import json
import math
import os
import re
from shutil import rmtree
import subprocess
import time

import srt

from config import FFMPEG_DIR, FRAME_MAX_SIDE

_CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
_FRAME_RE = re.compile(r"frame\d+\.jpg")


def _bin(name: str) -> str:
    exe_name = f"{name}.exe" if os.name == "nt" else name
    candidate = FFMPEG_DIR / exe_name
    if candidate.is_file():
        return str(candidate)
    return name


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        creationflags=_CREATE_NO_WINDOW,
    )


def _format_ffmpeg_error(action: str, res: subprocess.CompletedProcess) -> str:
    raw = (res.stderr or "").strip() or (res.stdout or "").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    detail = "\n".join(lines[-15:])
    if detail:
        return f"Ошибка {action} (код {res.returncode}):\n{detail}"
    return f"Ошибка {action} (код {res.returncode})"


@lru_cache(maxsize=16)
def _probe_cached(video_path_str: str, mtime_ns: int) -> dict:
    cmd = [
        _bin("ffprobe"),
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json",
        video_path_str,
    ]
    res = _run(cmd)
    if res.returncode != 0:
        raise RuntimeError(_format_ffmpeg_error("ffprobe", res))
    return json.loads(res.stdout)


def _probe(video_path) -> dict:
    path_str = str(video_path)
    try:
        mtime_ns = os.stat(path_str).st_mtime_ns
    except OSError:
        mtime_ns = 0
    return _probe_cached(os.path.abspath(path_str), mtime_ns)


def _primary_video_stream(probe: dict) -> dict | None:
    video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    for stream in video_streams:
        if not stream.get("disposition", {}).get("attached_pic"):
            return stream
    return video_streams[0] if video_streams else None


def _parse_positive_float(value) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(num) and num > 0:
        return num
    return None


def _prepare_frames_dir(frames_dir: str = "frames") -> None:
    os.makedirs(frames_dir, exist_ok=True)
    with os.scandir(frames_dir) as entries:
        for entry in entries:
            for attempt in range(5):
                try:
                    if entry.is_dir(follow_symlinks=False):
                        rmtree(entry.path)
                    else:
                        os.unlink(entry.path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.1)


def _list_frames(frames_dir: str = "frames") -> list[str]:
    return [f for f in os.listdir(frames_dir) if _FRAME_RE.fullmatch(f)]


def load_subtitles(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return list(srt.parse(file.read()))


def find_pauses(file_path, period):
    pauses = []
    subs = load_subtitles(file_path)

    for prev, curr in zip(subs, subs[1:]):
        gap = (curr.start - prev.end).total_seconds()
        if gap >= period:
            pauses.append((prev.end, curr.start, gap))

    return pauses


def get_subtitle(frame_index, file_path):
    subs = load_subtitles(file_path)

    if frame_index >= len(subs):
        return None
    return subs[frame_index]


def get_fps(video_path):
    probe = _probe(video_path)
    video_stream = _primary_video_stream(probe)
    if video_stream is None:
        raise RuntimeError(f"Видеопоток не найден в файле: {video_path}")

    fps_str = video_stream["r_frame_rate"]
    num, den = map(int, fps_str.split("/"))
    if den == 0:
        raise ValueError(f"Некорректная частота кадров: {fps_str}")
    return num / den


def get_duration(video_path):
    probe = _probe(video_path)
    durations = []
    video_stream = _primary_video_stream(probe)
    if video_stream is not None:
        d = _parse_positive_float(video_stream.get("duration"))
        if d is not None:
            durations.append(d)
    d = _parse_positive_float(probe.get("format", {}).get("duration"))
    if d is not None:
        durations.append(d)
    return min(durations) if durations else None


def extract_frame(video_path, timecode, output_path):
    tc = max(0.0, float(timecode))
    cmd = [
        _bin("ffmpeg"),
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{tc:.3f}",
        "-i", str(video_path),
        "-map", "0:V:0",
    ]
    if FRAME_MAX_SIDE:
        # уменьшаем только большие кадры, пропорции сохраняются (любой формат видео)
        cmd += ["-vf", f"scale={FRAME_MAX_SIDE}:{FRAME_MAX_SIDE}:force_original_aspect_ratio=decrease"]
    cmd += [
        "-frames:v", "1",
        "-update", "1",
        "-strict", "unofficial",
        "-q:v", "2",
        str(output_path),
    ]
    res = _run(cmd)
    if res.returncode != 0 or not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(_format_ffmpeg_error(f"ffmpeg при вырезке кадра ({tc:.2f}с)", res))


def scene_window(pause_index, subtitles_path, period, video_path=None):
    pauses = find_pauses(subtitles_path, period)

    start = max(0.0, pauses[pause_index][0].total_seconds())
    end = pauses[pause_index][1].total_seconds()

    if video_path:
        duration = get_duration(video_path)
        if duration:
            end = min(end, max(0.0, duration - 0.1))
            start = min(start, end)

    return start, end


def slice(pause_index, shift, subtitles_path, period, video_path):
    shift = float(shift)
    if shift <= 0:
        raise ValueError(f"Некорректный шаг вырезания кадров: {shift}")

    _prepare_frames_dir("frames")
    start, end = scene_window(pause_index, subtitles_path, period, video_path)

    # Не заступаем на самую границу end, где уже начинается следующая реплика
    window_dur = max(0.05, end - start - 0.05)
    frac = Fraction(shift).limit_denominator(1000)
    fps_expr = f"{frac.denominator}/{frac.numerator}"
    vf_parts = [f"fps={fps_expr}:start_time=0"]
    if FRAME_MAX_SIDE:
        vf_parts.append(f"scale={FRAME_MAX_SIDE}:{FRAME_MAX_SIDE}:force_original_aspect_ratio=decrease")

    cmd = [
        _bin("ffmpeg"),
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{window_dur:.3f}",
        "-map", "0:V:0",
        "-vf", ",".join(vf_parts),
        "-strict", "unofficial",
        "-q:v", "2",
        os.path.join("frames", "frame%d.jpg"),
    ]
    res = _run(cmd)
    if res.returncode == 0 and _list_frames("frames"):
        return start, end

    single_err = _format_ffmpeg_error(f"ffmpeg при вырезке сцены ({start:.2f}с–{end:.2f}с)", res)
    _prepare_frames_dir("frames")

    current = start
    stop_at = max(start, end - 0.05)
    i = 1
    last_err = None
    while current <= stop_at + 1e-6:
        try:
            extract_frame(video_path, current, f"frames/frame{i}.jpg")
            i += 1
        except RuntimeError as e:
            last_err = str(e)
        current += shift

    if i == 1:
        try:
            extract_frame(video_path, max(0.0, start), "frames/frame1.jpg")
        except RuntimeError as e:
            last_err = str(e)

    if not _list_frames("frames"):
        raise RuntimeError(last_err or single_err)

    return start, end
