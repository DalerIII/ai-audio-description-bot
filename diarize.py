"""Собственная диаризация для пайплайна Tiflo: Аудио (ONNX WeSpeaker + F0) + Сцены + LLM.

Не требует PyTorch, SciPy или scikit-learn — работает на numpy + onnxruntime + llama-server.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from config import (
    DIAR_ANCHOR_MIN_SEC,
    DIAR_MODEL_FILE,
    DIAR_REPO_ID,
    DIAR_SCENE_GAP_SEC,
    DIAR_SIM_THRESHOLD,
    FFMPEG_DIR,
)
from models import chat

SR = 16000
_CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if (os.name == "nt" and getattr(sys, "frozen", False)) else 0


# ---------------------------------------------------------------------------
# Вспомогательные утилиты (ffmpeg / ffprobe / WAV)
# ---------------------------------------------------------------------------

def _bin(name: str) -> str:
    exe = f"{name}.exe" if os.name == "nt" else name
    candidate = Path(FFMPEG_DIR) / exe
    return str(candidate) if candidate.is_file() else name


def _fmt_ts(sec: float) -> str:
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _probe_audio_channels(video_path: str | Path, track: int = 0) -> int:
    cmd = [
        _bin("ffprobe"),
        "-v", "error",
        "-select_streams", f"a:{track}",
        "-show_entries", "stream=channels",
        "-of", "default=nw=1:nk=1",
        str(video_path),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        creationflags=_CREATE_NO_WINDOW,
    )
    if res.returncode == 0:
        try:
            return int(res.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            pass
    return 2


def extract_diarization_audio(
    input_video: str | Path,
    workdir: str | Path,
    fallback_wav: str | Path | None = None,
    track: int = 0,
) -> Path:
    """Готовит чистую 16 kHz моно-дорожку для диаризации.

    Для 5.1/7.1 извлекает центральный канал (FC), где сведены диалоги без музыки.
    Для стерео/моно применяет речевой полосовой фильтр (80–7600 Гц).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out_wav = workdir / "diar_16k.wav"
    if out_wav.is_file() and out_wav.stat().st_size > 1024:
        return out_wav

    channels = _probe_audio_channels(input_video, track=track) if input_video else 2
    if channels >= 6:
        af = "pan=mono|c0=FC,highpass=f=80,lowpass=f=7600"
    else:
        af = "highpass=f=80,lowpass=f=7600"

    src = str(input_video) if input_video else str(fallback_wav)
    cmd = [
        _bin("ffmpeg"),
        "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", src,
    ]
    if input_video:
        cmd += ["-map", f"0:a:{track}", "-vn"]
    cmd += [
        "-af", af,
        "-ac", "1", "-ar", str(SR), "-acodec", "pcm_s16le",
        str(out_wav),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        creationflags=_CREATE_NO_WINDOW,
    )
    if res.returncode != 0 or not out_wav.is_file():
        if fallback_wav and Path(fallback_wav).is_file():
            return Path(fallback_wav)
        raise RuntimeError(f"Ошибка подготовки аудио для диаризации: {res.stderr.strip()}")
    return out_wav


def _load_wav_float32(wav_path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        if sw != 2:
            raise ValueError(f"Ожидался 16-bit PCM WAV, получено {sw * 8}-bit")
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1)
    return samples, sr


# ---------------------------------------------------------------------------
# Kaldi-совместимый 80-bin Log-Mel Fbank (чистый numpy, для WeSpeaker ONNX)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (np.exp(np.asarray(mel, dtype=np.float64) / 1127.0) - 1.0)


def _build_mel_filterbank(
    sr: int = SR,
    n_fft: int = 512,
    n_mels: int = 80,
    low_freq: float = 20.0,
    high_freq: float = 7600.0,
) -> np.ndarray:
    num_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, num_bins, dtype=np.float64)
    mel_points = np.linspace(_hz_to_mel(low_freq), _hz_to_mel(high_freq), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fb = np.zeros((n_mels, num_bins), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        up = (freqs - left) / max(1e-12, center - left)
        down = (right - freqs) / max(1e-12, right - center)
        fb[m] = np.maximum(0.0, np.minimum(up, down)).astype(np.float32)
    return fb


_MEL_FB = _build_mel_filterbank()
_POVEY_WIN = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(400, dtype=np.float32) / 399.0)) ** 0.85


def compute_kaldi_fbank(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """Возвращает нормализованные (CMN) 80-dim log-mel признаки формы (T, 80)."""
    frame_len = int(0.025 * sr)   # 400
    frame_step = int(0.010 * sr)  # 160
    n_fft = 512

    if len(wav) < frame_len:
        wav = np.pad(wav, (0, frame_len - len(wav)))

    # Масштаб под Kaldi (16-bit амплитуды), удаление DC и pre-emphasis 0.97
    sig = wav.astype(np.float32) * 32768.0
    n_frames = 1 + (len(sig) - frame_len) // frame_step
    strides = (sig.strides[0] * frame_step, sig.strides[0])
    frames = np.lib.stride_tricks.as_strided(sig, shape=(n_frames, frame_len), strides=strides).copy()

    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1]
    frames[:, 0] *= 0.03
    frames *= _POVEY_WIN[None, :]

    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    mel = np.maximum(power @ _MEL_FB.T, 1e-10)
    log_mel = np.log(mel)
    # Cepstral Mean Normalization (обязательно для WeSpeaker)
    log_mel -= log_mel.mean(axis=0, keepdims=True)
    return log_mel.astype(np.float32)


# ---------------------------------------------------------------------------
# Голосовой энкодер (WeSpeaker CAM++ ONNX с мягким fallback)
# ---------------------------------------------------------------------------

class SpeakerEncoder:
    def __init__(self, repo_id: str = DIAR_REPO_ID, model_file: str = DIAR_MODEL_FILE):
        self.session = None
        self.input_name = None
        self._load(repo_id, model_file)

    def _load(self, repo_id: str, model_file: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[diarize] onnxruntime не установлен — используется базовый спектральный fallback.")
            return

        try:
            try:
                model_path = hf_hub_download(repo_id=repo_id, filename=model_file, local_files_only=True)
            except Exception:
                model_path = hf_hub_download(repo_id=repo_id, filename=model_file)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
        except Exception as e:
            print(f"[diarize] Не удалось загрузить ONNX-модель {repo_id}/{model_file}: {e}. Использую fallback.")
            self.session = None

    def encode(self, wav: np.ndarray, sr: int = SR) -> np.ndarray:
        feats = compute_kaldi_fbank(wav, sr=sr)
        if self.session is not None:
            out = self.session.run(None, {self.input_name: feats[None, :, :]})[0]
            emb = np.asarray(out, dtype=np.float32).reshape(-1)
        else:
            # Fallback: разброс и динамика log-mel спектра (std + delta std)
            delta = np.diff(feats, axis=0) if len(feats) > 1 else np.zeros_like(feats)
            emb = np.concatenate([feats.std(axis=0), delta.std(axis=0)]).astype(np.float32)
            emb = emb - emb.mean()
        norm = float(np.linalg.norm(emb))
        return emb / max(norm, 1e-8)


# ---------------------------------------------------------------------------
# Оценка основного тона F0 (М/Ж) и качества сигнала (энергия речи)
# ---------------------------------------------------------------------------

def estimate_pitch_and_gender(wav: np.ndarray, sr: int = SR) -> tuple[float, str, float]:
    """Оценивает медианную частоту основного тона F0 (Гц), пол ('M'/'F'/'U') и RMS."""
    if len(wav) < int(0.05 * sr):
        return 0.0, "U", 0.0

    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < 0.004:
        return 0.0, "U", rms

    win_len = int(0.032 * sr)   # 512 отсчётов
    hop = int(0.012 * sr)       # 192 отсчёта
    min_lag = int(sr / 320.0)   # 320 Гц
    max_lag = int(sr / 75.0)    # 75 Гц

    f0_vals = []
    n_frames = 1 + max(0, (len(wav) - win_len) // hop)
    # Ограничиваем число кадров для скорости на длинных репликах
    step = max(1, n_frames // 80)
    for idx in range(0, n_frames, step):
        frame = wav[idx * hop : idx * hop + win_len]
        if len(frame) < win_len:
            break
        e = float(np.mean(frame ** 2))
        if e < (rms * 0.35) ** 2:
            continue
        frame = frame - frame.mean()
        ac = np.correlate(frame, frame, mode="full")[win_len - 1 :]
        if ac[0] <= 1e-9:
            continue
        ac = ac / ac[0]
        seg = ac[min_lag : min(max_lag + 1, len(ac))]
        if len(seg) == 0:
            continue
        peak_i = int(np.argmax(seg))
        peak_val = float(seg[peak_i])
        if peak_val >= 0.42:
            lag = min_lag + peak_i
            f0_vals.append(sr / float(lag))

    if len(f0_vals) < 3:
        return 0.0, "U", rms

    f0 = float(np.median(f0_vals))
    if f0 < 155.0:
        gender = "M"
    elif f0 > 180.0:
        gender = "F"
    else:
        gender = "U"
    return f0, gender, rms


# ---------------------------------------------------------------------------
# Разбиение на сцены + кластеризация с ограничениями внутри сцены
# ---------------------------------------------------------------------------

def split_into_scenes(
    segments: list[dict],
    gap_sec: float = DIAR_SCENE_GAP_SEC,
    max_lines_per_scene: int = 70,
) -> list[tuple[int, int]]:
    """Делит список сегментов на сцены [(first_idx, last_idx), ...]."""
    if not segments:
        return []
    raw_scenes: list[tuple[int, int]] = []
    start_i = 0
    for i in range(1, len(segments)):
        gap = segments[i]["start"] - segments[i - 1]["end"]
        if gap >= gap_sec:
            raw_scenes.append((start_i, i - 1))
            start_i = i
    raw_scenes.append((start_i, len(segments) - 1))

    # Слишком длинные непрерывные диалоги делим по наибольшей внутренней паузе
    scenes: list[tuple[int, int]] = []
    for a, b in raw_scenes:
        stack = [(a, b)]
        while stack:
            lo, hi = stack.pop()
            if hi - lo + 1 <= max_lines_per_scene:
                scenes.append((lo, hi))
                continue
            mid_lo = lo + max(10, (hi - lo) // 3)
            mid_hi = hi - max(10, (hi - lo) // 3)
            best_k = (lo + hi) // 2
            best_gap = -1.0
            for k in range(mid_lo, mid_hi + 1):
                g = segments[k]["start"] - segments[k - 1]["end"]
                if g > best_gap:
                    best_gap = g
                    best_k = k
            stack.append((best_k, hi))
            stack.append((lo, best_k - 1))
    scenes.sort(key=lambda x: x[0])
    return scenes


def _cluster_gender(genders: list[str]) -> str:
    m_cnt = sum(1 for g in genders if g == "M")
    f_cnt = sum(1 for g in genders if g == "F")
    if m_cnt >= 1 and f_cnt == 0:
        return "M"
    if f_cnt >= 1 and m_cnt == 0:
        return "F"
    if m_cnt >= 2 * max(1, f_cnt):
        return "M"
    if f_cnt >= 2 * max(1, m_cnt):
        return "F"
    return "U"


def _genders_conflict(g1: str, g2: str) -> bool:
    return (g1 == "M" and g2 == "F") or (g1 == "F" and g2 == "M")


def _cluster_scene_anchors(
    anchor_indices: list[int],
    embeddings: dict[int, np.ndarray],
    genders: dict[int, str],
    sim_threshold: float = DIAR_SIM_THRESHOLD,
) -> list[list[int]]:
    """Агломеративная кластеризация (average-linkage) якорных реплик сцены с запретом слияния М+Ж."""
    if not anchor_indices:
        return []
    clusters: list[list[int]] = [[idx] for idx in anchor_indices]
    if len(clusters) == 1:
        return clusters

    # Предвычисляем матрицу косинусной близости
    sim_map: dict[tuple[int, int], float] = {}
    for i, a in enumerate(anchor_indices):
        ea = embeddings[a]
        for b in anchor_indices[i + 1 :]:
            s = float(np.dot(ea, embeddings[b]))
            sim_map[(a, b)] = s
            sim_map[(b, a)] = s

    while len(clusters) > 1:
        best_sim = -2.0
        best_pair = None
        for i in range(len(clusters)):
            gi = _cluster_gender([genders[k] for k in clusters[i]])
            for j in range(i + 1, len(clusters)):
                gj = _cluster_gender([genders[k] for k in clusters[j]])
                if _genders_conflict(gi, gj):
                    continue
                vals = [sim_map[(a, b)] for a in clusters[i] for b in clusters[j]]
                avg_sim = float(np.mean(vals))
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_pair = (i, j)
        if best_pair is None or best_sim < sim_threshold:
            break
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)

    return clusters


def _detect_mvo(
    anchor_indices: list[int],
    embeddings: dict[int, np.ndarray],
    genders: dict[int, str],
) -> bool:
    """Определяет двухголосую закадровую озвучку (MVO), где 1 диктор на всех М и 1 на всех Ж."""
    if len(anchor_indices) < 20:
        return False
    rng = np.random.default_rng(42)
    single_actor_genders = 0
    checked_genders = 0

    for target_g in ("M", "F"):
        idxs = [i for i in anchor_indices if genders.get(i) == target_g]
        if len(idxs) < 10:
            continue
        checked_genders += 1
        sample = rng.choice(idxs, size=min(40, len(idxs)), replace=False)
        embs = np.stack([embeddings[int(i)] for i in sample], axis=0)
        centroid = embs.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
        sims = embs @ centroid
        # Если 85%+ всех реплик этого пола по всему фильму сидят вокруг одного голоса — это MVO
        if float(np.mean(sims >= 0.68)) >= 0.85:
            single_actor_genders += 1

    return checked_genders > 0 and single_actor_genders == checked_genders


# ---------------------------------------------------------------------------
# Сквозная сшивка голосовых кластеров между сценами
# ---------------------------------------------------------------------------

def _link_scenes_globally(
    scene_clusters: list[tuple[int, list[int], np.ndarray, str]],
    link_threshold: float = 0.56,
) -> dict[tuple[int, int], str]:
    """Сшивает локальные кластеры сцен в сквозные ID (SPK_1, SPK_2, ...).

    scene_clusters: [(scene_idx, member_indices, centroid, gender), ...]
    Возвращает {(scene_idx, local_cluster_idx): global_label}.
    """
    global_profiles: list[dict] = []
    mapping: dict[tuple[int, int], str] = {}

    # Группируем по сцене, чтобы знать локальный индекс
    per_scene_counter: dict[int, int] = {}
    for scene_idx, members, centroid, gender in scene_clusters:
        local_idx = per_scene_counter.get(scene_idx, 0)
        per_scene_counter[scene_idx] = local_idx + 1

        best_gid = None
        best_sim = -2.0
        for gid, prof in enumerate(global_profiles):
            if _genders_conflict(gender, prof["gender"]):
                continue
            # Не склеиваем два РАЗНЫХ кластера из одной и той же сцены
            if scene_idx in prof["scenes"]:
                continue
            sim = float(np.dot(centroid, prof["centroid"]))
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        if best_gid is not None and best_sim >= link_threshold:
            prof = global_profiles[best_gid]
            w_old = prof["count"]
            w_new = len(members)
            new_c = (prof["centroid"] * w_old + centroid * w_new) / (w_old + w_new)
            new_c /= max(float(np.linalg.norm(new_c)), 1e-8)
            prof["centroid"] = new_c
            prof["count"] = w_old + w_new
            prof["scenes"].add(scene_idx)
            if prof["gender"] == "U" and gender != "U":
                prof["gender"] = gender
            mapping[(scene_idx, local_idx)] = f"SPK_{best_gid + 1}"
        else:
            gid = len(global_profiles)
            global_profiles.append({
                "centroid": centroid.copy(),
                "count": len(members),
                "gender": gender,
                "scenes": {scene_idx},
            })
            mapping[(scene_idx, local_idx)] = f"SPK_{gid + 1}"

    return mapping


# ---------------------------------------------------------------------------
# Акустическая диаризация всего фильма (Этапы 1–2)
# ---------------------------------------------------------------------------

def diarize_audio_only(
    segments: list[dict],
    wav_path: str | Path,
    stop_event=None,
    on_progress=None,
) -> tuple[list[dict], list[tuple[int, int]], bool]:
    """Проставляет предварительные голосовые метки (acoustic_speaker, gender, is_anchor).

    Возвращает (обогащённые_сегменты, scenes, is_mvo).
    """
    if not segments:
        return segments, [], False

    def emit(frac: float, text: str) -> None:
        if on_progress is not None:
            on_progress(frac, text)

    def check_stop() -> None:
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Остановлено пользователем")

    emit(0.05, "Диаризация: загрузка аудио и модели голоса...")
    wav, sr = _load_wav_float32(wav_path)
    encoder = SpeakerEncoder()
    scenes = split_into_scenes(segments)

    embeddings: dict[int, np.ndarray] = {}
    genders: dict[int, str] = {}
    anchors_set: set[int] = set()
    total = len(segments)

    for idx, seg in enumerate(segments):
        if idx % 25 == 0:
            check_stop()
            emit(0.10 + 0.65 * (idx / max(1, total)), f"Диаризация: анализ голосов ({idx}/{total})...")
        s_i = max(0, int(seg["start"] * sr))
        e_i = min(len(wav), int(seg["end"] * sr))
        chunk = wav[s_i:e_i]
        dur = (e_i - s_i) / float(sr)

        _, gender, rms = estimate_pitch_and_gender(chunk, sr=sr)
        genders[idx] = gender

        if dur >= 0.8 and rms >= 0.006:
            embeddings[idx] = encoder.encode(chunk, sr=sr)
            if dur >= DIAR_ANCHOR_MIN_SEC and rms >= 0.010:
                anchors_set.add(idx)

    check_stop()
    emit(0.80, "Диаризация: кластеризация сцен...")
    all_anchors = sorted(anchors_set)
    is_mvo = _detect_mvo(all_anchors, embeddings, genders) if encoder.session is not None else False
    if is_mvo:
        print("[diarize] Обнаружена закадровая озвучка (MVO): голоса будут разделяться по полу и контексту диалога.")

    # Кластеризация внутри каждой сцены
    scene_clusters_meta: list[tuple[int, list[int], np.ndarray, str]] = []
    local_assignments: dict[int, tuple[int, int]] = {}  # seg_idx -> (scene_idx, local_cluster_idx)

    for scene_idx, (a, b) in enumerate(scenes):
        scene_anchors = [i for i in range(a, b + 1) if i in anchors_set]
        if not scene_anchors:
            # Если в короткой сцене нет длинных якорей, берём всё, у чего есть эмбеддинг
            scene_anchors = [i for i in range(a, b + 1) if i in embeddings]
        clusters = _cluster_scene_anchors(scene_anchors, embeddings, genders)

        # Считаем центроиды локальных кластеров сцены
        centroids = []
        c_genders = []
        for local_idx, members in enumerate(clusters):
            c = np.mean([embeddings[m] for m in members], axis=0)
            c /= max(float(np.linalg.norm(c)), 1e-8)
            g = _cluster_gender([genders[m] for m in members])
            centroids.append(c)
            c_genders.append(g)
            scene_clusters_meta.append((scene_idx, members, c, g))
            for m in members:
                local_assignments[m] = (scene_idx, local_idx)

        # Подтягиваем средние реплики (0.8..1.4с) только при высокой уверенности
        medium_th = DIAR_SIM_THRESHOLD + 0.07
        for i in range(a, b + 1):
            if i in local_assignments or i not in embeddings or not centroids:
                continue
            gi = genders[i]
            best_c = None
            best_s = -2.0
            for local_idx, (c, cg) in enumerate(zip(centroids, c_genders)):
                if _genders_conflict(gi, cg):
                    continue
                s = float(np.dot(embeddings[i], c))
                if s > best_s:
                    best_s = s
                    best_c = local_idx
            if best_c is not None and best_s >= medium_th:
                local_assignments[i] = (scene_idx, best_c)

    # Сквозная сшивка между сценами (для MVO оставляем локальные ID + пол, чтобы не сливать разных героев)
    if is_mvo:
        global_map = {}
        cnt: dict[int, int] = {}
        for scene_idx, _, _, _ in scene_clusters_meta:
            l_idx = cnt.get(scene_idx, 0)
            cnt[scene_idx] = l_idx + 1
            global_map[(scene_idx, l_idx)] = f"S{scene_idx + 1}_V{l_idx + 1}"
    else:
        global_map = _link_scenes_globally(scene_clusters_meta)

    enriched = []
    for idx, seg in enumerate(segments):
        item = dict(seg)
        item["gender"] = genders.get(idx, "U")
        item["is_anchor"] = idx in anchors_set
        if idx in local_assignments:
            item["acoustic_speaker"] = global_map[local_assignments[idx]]
        else:
            item["acoustic_speaker"] = None
        enriched.append(item)

    emit(1.0, "Диаризация: голосовой анализ завершён")
    return enriched, scenes, is_mvo


# ---------------------------------------------------------------------------
# Этап 3: LLM-доразметка коротких реплик и присвоение имён внутри сцен
# ---------------------------------------------------------------------------

_LLM_DIAR_SYSTEM = (
    "You assign the speaker for each subtitle line in one movie scene.\n"
    "Each line has an index, timestamp, and an optional acoustic voice hint like [SPK_1 (M)], [SPK_2 (F)], or [?] "
    "(short utterance where audio alone is unreliable). M = male voice, F = female voice, U = uncertain.\n"
    "RULES:\n"
    "1. Use the acoustic voice IDs as strong anchors for long lines, and use dialogue turn-taking (question -> answer), "
    "grammatical gender (e.g. 'я забыл' vs 'я забыла'), and direct address ('Ники, стой!' -> the speaker is NOT Nikki) "
    "to fill in every [?] line and fix rare acoustic errors (e.g. shouting/whispering split).\n"
    "2. If a character's name is clear from the scene dialogue or the known voice-to-name mapping from earlier scenes, "
    "output that real name (consistent spelling). Otherwise keep a short clear label in {language} "
    "(e.g. 'Мужчина 1', 'Женщина 1', 'Продавец', 'Ведущий').\n"
    "3. Never assign dialogue to pets/animals or deceased/absent people merely mentioned in conversation.\n"
    "4. Output ONLY one line per subtitle in exact format:\n"
    "INDEX | SpeakerName\n"
    "Cover every index from the scene. Output nothing else."
)

_LLM_LINE_RE = re.compile(r"^\s*[-*•]?\s*(\d+)\s*[|:]\s*(.+?)\s*$")


def _fallback_fill_short(segments: list[dict], a: int, b: int) -> None:
    """Простое сглаживание коротких реплик без LLM (чередование / ближайший совместимый по полу)."""
    scene_speakers = []
    for i in range(a, b + 1):
        spk = segments[i].get("speaker") or segments[i].get("acoustic_speaker")
        if spk and spk not in scene_speakers:
            scene_speakers.append(spk)

    for i in range(a, b + 1):
        if segments[i].get("speaker"):
            continue
        if segments[i].get("acoustic_speaker"):
            segments[i]["speaker"] = segments[i]["acoustic_speaker"]
            continue
        prev_spk = segments[i - 1].get("speaker") if i > a else None
        next_spk = None
        for j in range(i + 1, min(b + 1, i + 4)):
            s = segments[j].get("speaker") or segments[j].get("acoustic_speaker")
            if s:
                next_spk = s
                break
        if prev_spk and next_spk and prev_spk != next_spk:
            # Между двумя разными говорящими — берём того, к кому реплика ближе по времени
            gap_prev = segments[i]["start"] - segments[i - 1]["end"]
            gap_next = segments[i + 1]["start"] - segments[i]["end"] if i < b else 99.0
            segments[i]["speaker"] = prev_spk if gap_prev <= gap_next else next_spk
        elif prev_spk and len(scene_speakers) == 2:
            # В диалоге вдвоём короткий ответ после паузы обычно говорит собеседник
            gap_prev = segments[i]["start"] - segments[i - 1]["end"]
            other = scene_speakers[1] if scene_speakers[0] == prev_spk else scene_speakers[0]
            segments[i]["speaker"] = other if gap_prev >= 0.35 else prev_spk
        elif prev_spk:
            segments[i]["speaker"] = prev_spk
        elif next_spk:
            segments[i]["speaker"] = next_spk
        else:
            g = segments[i].get("gender", "U")
            segments[i]["speaker"] = "Мужчина" if g == "M" else ("Женщина" if g == "F" else "Голос")


def refine_speakers_with_llm(
    llm,
    segments: list[dict],
    scenes: list[tuple[int, int]],
    is_mvo: bool = False,
    language: str = "Russian",
    stop_event=None,
    on_progress=None,
) -> list[dict]:
    """Доразмечает короткие реплики и привязывает голосовые кластеры к именам через LLM."""
    if not segments or not scenes:
        return segments

    known_voice_names: dict[str, str] = {}  # SPK_1 -> "Бэр" (только для сквозных ID, не для MVO)
    total_scenes = len(scenes)
    system_prompt = _LLM_DIAR_SYSTEM.format(language=language)

    for s_idx, (a, b) in enumerate(scenes):
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Остановлено пользователем")
        if on_progress is not None:
            on_progress(s_idx / total_scenes, f"Диаризация (LLM): сцена {s_idx + 1}/{total_scenes}...")

        # Однострочные сцены без контекста не гоняем лишний раз через LLM
        if b - a < 1:
            spk = segments[a].get("acoustic_speaker")
            if spk and spk in known_voice_names:
                segments[a]["speaker"] = known_voice_names[spk]
            else:
                _fallback_fill_short(segments, a, b)
            continue

        lines = []
        for i in range(a, b + 1):
            seg = segments[i]
            ac = seg.get("acoustic_speaker")
            g = seg.get("gender", "U")
            if ac:
                known = known_voice_names.get(ac)
                tag = f"[{ac}={known} ({g})]" if known else f"[{ac} ({g})]"
            else:
                tag = f"[? ({g})]" if g != "U" else "[?]"
            lines.append(f"{i} [{_fmt_ts(seg['start'])}] {tag}: {seg['text']}")

        mvo_note = (
            "NOTE: This audio is a multi-voice voice-over (MVO), so one dubbing actor voices multiple characters of the same gender. "
            "Rely primarily on gender (M/F) and dialogue context rather than assuming identical voice IDs are the same person.\n\n"
            if is_mvo
            else ""
        )
        known_block = ""
        if known_voice_names and not is_mvo:
            pairs = ", ".join(f"{k} = {v}" for k, v in list(known_voice_names.items())[-12:])
            known_block = f"Previously identified voices: {pairs}\n\n"

        user_prompt = (
            f"{mvo_note}{known_block}"
            f"Scene subtitles:\n" + "\n".join(lines) + "\n\n"
            f"Assign speaker for every index from {a} to {b}."
        )
        max_tok = min(2048, max(256, (b - a + 1) * 14))
        try:
            answer = chat(
                llm,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tok,
                temperature=0.1,
            )
        except Exception as e:
            print(f"[diarize] Ошибка LLM на сцене {s_idx + 1}: {e}")
            answer = ""

        # Разбираем ответы LLM
        votes: dict[str, dict[str, int]] = {}
        for raw_line in answer.splitlines():
            m = _LLM_LINE_RE.match(raw_line)
            if not m:
                continue
            idx = int(m.group(1))
            name = m.group(2).strip().strip("[]\"'«»")
            if a <= idx <= b and name and name != "?":
                segments[idx]["speaker"] = name
                ac = segments[idx].get("acoustic_speaker")
                if ac and segments[idx].get("is_anchor") and not is_mvo:
                    # Запоминаем только реальные имена (не обезличенные SPK_ / Мужчина / Женщина)
                    low = name.lower()
                    if not low.startswith(("spk_", "мужчина", "женщина", "голос", "speaker", "man", "woman")):
                        votes.setdefault(ac, {})
                        votes[ac][name] = votes[ac].get(name, 0) + 1

        # Обновляем таблицу известных голосов по большинству голосов якорных реплик
        if not is_mvo:
            for ac, counts in votes.items():
                best_name, best_cnt = max(counts.items(), key=lambda kv: kv[1])
                if best_cnt >= 2 or ac not in known_voice_names:
                    known_voice_names[ac] = best_name

        _fallback_fill_short(segments, a, b)

    if on_progress is not None:
        on_progress(1.0, "Диаризация завершена")
    return segments


def diarize_segments(
    input_video: str | Path,
    workdir: str | Path,
    segments: list[dict],
    llm=None,
    language: str = "Russian",
    fallback_wav: str | Path | None = None,
    stop_event=None,
    on_progress=None,
) -> list[dict]:
    """Полный пайплайн диаризации: подготовка аудио -> ONNX + F0 + сцены -> (опционально) LLM."""
    if not segments:
        return segments

    def emit(frac: float, text: str) -> None:
        if on_progress is not None:
            on_progress(frac, text)

    emit(0.02, "Диаризация: подготовка аудиодорожки...")
    diar_wav = extract_diarization_audio(input_video, workdir, fallback_wav=fallback_wav)

    audio_span = 0.45 if llm is not None else 1.0
    enriched, scenes, is_mvo = diarize_audio_only(
        segments,
        diar_wav,
        stop_event=stop_event,
        on_progress=lambda f, t: emit(0.02 + f * (audio_span - 0.02), t),
    )

    if llm is not None:
        enriched = refine_speakers_with_llm(
            llm,
            enriched,
            scenes,
            is_mvo=is_mvo,
            language=language,
            stop_event=stop_event,
            on_progress=lambda f, t: emit(audio_span + f * (1.0 - audio_span), t),
        )
    else:
        for a, b in scenes:
            _fallback_fill_short(enriched, a, b)

    return enriched
