import ffmpeg
import srt
from shutil import rmtree
from os import makedirs
from config import FRAME_MAX_SIDE


def load_subtitles(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
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
    probe = ffmpeg.probe(video_path)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')

    fps_str = video_stream['r_frame_rate']
    num, den = map(int, fps_str.split('/'))
    fps = num / den

    return fps


def get_duration(video_path):
    probe = ffmpeg.probe(video_path)
    try:
        return float(probe['format']['duration'])
    except (KeyError, ValueError):
        for s in probe['streams']:
            if s['codec_type'] == 'video' and 'duration' in s:
                return float(s['duration'])
    return None


def extract_frame(video_path, timecode, output_path):
    stream = ffmpeg.input(video_path, ss=timecode)
    if FRAME_MAX_SIDE:
        # уменьшаем только большие кадры, пропорции сохраняются (любой формат видео)
        stream = stream.filter("scale", FRAME_MAX_SIDE, FRAME_MAX_SIDE,
                               force_original_aspect_ratio="decrease")
    (
        stream
        .output(output_path, vframes=1, **{'q:v': 2})
        .overwrite_output()
        .run(quiet=True)
    )


def scene_window(pause_index, subtitles_path, period, video_path=None):
    pauses = find_pauses(subtitles_path, period)

    start = max(0.0, pauses[pause_index][0].total_seconds() - 1)
    end = pauses[pause_index][1].total_seconds() + 1

    if video_path:
        duration = get_duration(video_path)
        if duration:
            end = min(end, duration - 0.1)

    return start, end


def slice(pause_index, shift, subtitles_path, period, video_path):
    rmtree("frames/", ignore_errors=True)
    makedirs("frames/")

    start, end = scene_window(pause_index, subtitles_path, period, video_path)

    current = start-1
    i = 1
    while current <= (end+1):
        extract_frame(video_path, current, f"frames/frame{i}.jpg")
        current += shift
        i += 1

    return start, end