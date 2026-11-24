import os
import re
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

from config import (
    APP_DIR,
    LLAMA_SERVER_DIR, SERVER_HOST, SERVER_PORT,
    REPO_ID, MODEL_FILE, MMPROJ_FILE,
    N_CTX, N_GPU_LAYERS, N_THREADS, FLASH_ATTN,
    MMPROJ_OFFLOAD, N_PARALLEL, REASONING_BUDGET, LOG_VERBOSITY, MAX_REPLY,
    USE_EXTRA_SERVER_ARGS, N_BATCH, N_UBATCH, USE_MMAP, USE_MLOCK, OFFLOAD_KQV,
)

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", flags=re.DOTALL)
_NEW_PROMPT_RE = re.compile(r"new prompt.*task\.n_tokens\s*=\s*(\d+)")
_PROMPT_PROGRESS_RE = re.compile(
    r"prompt processing,\s*n_tokens\s*=\s*(\d+),\s*progress\s*=\s*[0-9.]+,\s*t\s*=\s*[0-9.]+\s*s\s*/\s*([0-9.]+)\s*tokens per second"
)


def _watch_server_output(proc: subprocess.Popen, on_stats=None) -> None:
    log_file = None
    frozen = getattr(sys, "frozen", False)
    if frozen:
        try:
            log_file = open(APP_DIR / "llama_server.log", "a", encoding="utf-8", errors="replace", buffering=1)
        except OSError:
            log_file = None

    current_prompt_total = None
    try:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            if log_file is not None:
                log_file.write(line)
            elif not frozen and sys.stdout is not None:
                sys.stdout.write(line)

            m_new = _NEW_PROMPT_RE.search(line)
            if m_new:
                current_prompt_total = int(m_new.group(1))
                if on_stats is not None:
                    on_stats(f"Модель: чтение 0/{current_prompt_total} ток...")
                continue

            m_prog = _PROMPT_PROGRESS_RE.search(line)
            if m_prog and on_stats is not None:
                done = int(m_prog.group(1))
                tps = float(m_prog.group(2))
                if current_prompt_total:
                    on_stats(f"Модель: чтение {done}/{current_prompt_total} ток ({tps:.1f} ток/с)")
                else:
                    on_stats(f"Модель: чтение {done} ток ({tps:.1f} ток/с)")
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if log_file is not None:
            log_file.close()


# === [EXTRA SERVER ARGS — "чёрный ящик", управляется через USE_EXTRA_SERVER_ARGS в config.py] ===
def _extra_server_args() -> list[str]:
    """Дополнительные параметры запуска llama-server (отключены, пока USE_EXTRA_SERVER_ARGS = False)."""
    if not USE_EXTRA_SERVER_ARGS:
        return []
    args = ["-b", str(N_BATCH), "-ub", str(N_UBATCH)]
    if not USE_MMAP:
        args.append("--no-mmap")
    if USE_MLOCK:
        args.append("--mlock")
    if not OFFLOAD_KQV:
        args.append("--no-kv-offload")
    return args
# === [/EXTRA SERVER ARGS] ===


def load_model(
    n_ctx: int = N_CTX,
    n_gpu_layers: int = N_GPU_LAYERS,
    reasoning_budget: int = REASONING_BUDGET,
    stop_event=None,
    repo_id: str = REPO_ID,
    model_file: str = MODEL_FILE,
    mmproj_file: str | None = MMPROJ_FILE,
    on_stats=None,
) -> dict:
    if not LLAMA_SERVER_DIR:
        raise ValueError("Не указан путь LLAMA_SERVER_DIR в config.py (папка с llama-server.exe)")
    if not repo_id or not model_file:
        raise ValueError("Не указаны репозиторий и файл модели.")
    server_dir = Path(LLAMA_SERVER_DIR)
    server_exe = server_dir / "llama-server.exe"
    if not server_exe.exists():
        raise FileNotFoundError(f"Не найден llama-server.exe по пути: {server_exe}")

    model_path = hf_hub_download(repo_id=repo_id, filename=model_file)
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Остановлено пользователем")

    mmproj_path = hf_hub_download(repo_id=repo_id, filename=mmproj_file) if mmproj_file else None
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Остановлено пользователем")

    cmd = [
        str(server_exe),
        "-m", model_path,
    ]
    if mmproj_path:
        cmd += ["--mmproj", mmproj_path]
        if not MMPROJ_OFFLOAD:
            cmd.append("--no-mmproj-offload")
    cmd += [
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "-c", str(n_ctx),
        "-ngl", str(n_gpu_layers),
        "-t", str(N_THREADS),
        "-fa", "on" if FLASH_ATTN else "off",
        "-np", str(N_PARALLEL),
    ]
    if reasoning_budget is not None:
        if reasoning_budget == 0:
            cmd += ["--reasoning", "off", "--reasoning-budget", "0"]
        else:
            cmd += ["--reasoning-budget", str(reasoning_budget)]
    if LOG_VERBOSITY is not None:
        cmd += ["-lv", str(LOG_VERBOSITY)]
    cmd += _extra_server_args()

    server_url = f"http://{SERVER_HOST}:{SERVER_PORT}"
    popen_kwargs = {
        "cwd": str(server_dir),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if os.name == "nt" and getattr(sys, "frozen", False):
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    proc = subprocess.Popen(cmd, **popen_kwargs)
    threading.Thread(target=_watch_server_output, args=(proc, on_stats), daemon=True).start()
    session = requests.Session()
    session.trust_env = False
    while True:
        if stop_event is not None and stop_event.is_set():
            close_model({"proc": proc, "url": server_url, "session": session})
            raise InterruptedError("Остановлено пользователем")
        if proc.poll() is not None:
            session.close()
            raise RuntimeError(f"llama-server завершился с кодом {proc.returncode}")
        try:
            resp = session.get(f"{server_url}/health", timeout=2)
            if resp.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return {
        "proc": proc,
        "url": server_url,
        "session": session,
        "model_file": model_file,
        "n_ctx": n_ctx,
        "reasoning_budget": reasoning_budget,
        "on_stats": on_stats,
        "_image_tokens_cache": {},
    }


def close_model(llm) -> None:
    if not llm:
        return
    _print_speed_summary()  # [SPEED STATS]
    session = llm.get("session")
    if session is not None:
        session.close()
    proc = llm.get("proc")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# === [SPEED STATS — можно целиком удалить] ===
_SPEED_LOG = []
_SPEED_FILE = Path("speed_stats.csv")


def _record_speed(data: dict, elapsed: float, model_file: str = MODEL_FILE, on_stats=None) -> None:
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}
    prompt_n = timings.get("prompt_n") or usage.get("prompt_tokens") or 0
    gen_n = timings.get("predicted_n") or usage.get("completion_tokens") or 0
    prompt_tps = timings.get("prompt_per_second") or 0.0
    gen_tps = timings.get("predicted_per_second") or (gen_n / elapsed if elapsed > 0 else 0.0)
    _SPEED_LOG.append((prompt_n, prompt_tps, gen_n, gen_tps, elapsed))

    write_header = not _SPEED_FILE.exists()
    with _SPEED_FILE.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("model,prompt_tokens,prompt_tps,gen_tokens,gen_tps,elapsed_sec\n")
        f.write(f"{model_file},{prompt_n},{prompt_tps:.2f},{gen_n},{gen_tps:.2f},{elapsed:.2f}\n")

    if on_stats is not None:
        on_stats(
            f"Модель: чтение {prompt_n} ток ({prompt_tps:.1f} ток/с) • "
            f"генерация {gen_n} ток ({gen_tps:.1f} ток/с)"
        )

    print(f"[speed] prompt: {prompt_n} tok ({prompt_tps:.1f} t/s) | gen: {gen_n} tok ({gen_tps:.1f} t/s) | {elapsed:.2f}s")


def _print_speed_summary() -> None:
    if not _SPEED_LOG:
        return
    avg_prompt = sum(x[1] for x in _SPEED_LOG) / len(_SPEED_LOG)
    avg_gen = sum(x[3] for x in _SPEED_LOG) / len(_SPEED_LOG)
    total_gen = sum(x[2] for x in _SPEED_LOG)
    print(f"[speed summary] requests: {len(_SPEED_LOG)} | total gen: {total_gen} tok | avg prompt: {avg_prompt:.1f} t/s | avg gen: {avg_gen:.1f} t/s")
    _SPEED_LOG.clear()
# === [/SPEED STATS] ===


# --- Общение с моделью через HTTP API llama-server ---

def chat(
    llm: dict,
    messages: list,
    max_tokens: int = MAX_REPLY,
    temperature: float = 0.3,
    repeat_penalty: float = 1.1,
) -> str:
    # max_tokens — лимит именно на итоговый ответ (content); бюджет на рассуждения (reasoning_budget)
    # идёт отдельно поверх него, чтобы мысли модели не съедали лимит ответа в 1500 (или 4..600) токенов.
    rb = llm.get("reasoning_budget", REASONING_BUDGET)
    no_think_extra = {"chat_template_kwargs": {"enable_thinking": False}}

    if rb == 0:
        attempts = ((max_tokens, no_think_extra), (max_tokens, None))
    else:
        think_allowance = rb if (rb is not None and rb > 0) else 2048
        attempts = ((max_tokens + think_allowance, None), (max_tokens, no_think_extra))

    for req_max_tokens, extra in attempts:
        payload = {
            "messages": messages,
            "max_tokens": req_max_tokens,
            "temperature": temperature,
        }
        if repeat_penalty is not None:
            payload["repeat_penalty"] = repeat_penalty
        if extra:
            payload.update(extra)
        t0 = time.perf_counter()  # [SPEED STATS]
        r = llm["session"].post(f"{llm['url']}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        _record_speed(
            data,
            time.perf_counter() - t0,
            llm.get("model_file", MODEL_FILE),
            on_stats=llm.get("on_stats"),
        )  # [SPEED STATS]
        # Извлекаем текст ответа и вырезаем возможные блоки рассуждений <think>...</think>
        text = data["choices"][0]["message"].get("content") or ""
        cleaned = _THINK_RE.sub("", text).strip()
        if cleaned:
            return cleaned
    return ""


def count_tokens(llm: dict, text: str) -> int:
    # Токенизируем строку без добавления спецтокенов (add_bos=False аналог add_special=False)
    r = llm["session"].post(
        f"{llm['url']}/tokenize",
        json={"content": text, "add_special": False},
    )
    r.raise_for_status()
    return len(r.json()["tokens"])


def cut_to_tokens(llm: dict, text: str, budget: int) -> str:
    # Получаем список ID токенов и обрезаем его до заданного лимита budget
    r = llm["session"].post(
        f"{llm['url']}/tokenize",
        json={"content": text, "add_special": False},
    )
    r.raise_for_status()
    tokens = r.json()["tokens"][:budget]

    # Детокенизируем обрезанный список ID обратно в строку через /detokenize
    r = llm["session"].post(
        f"{llm['url']}/detokenize",
        json={"tokens": tokens},
    )
    r.raise_for_status()
    return r.json().get("content", "")


def count_prompt_tokens(llm: dict, messages: list) -> int:
    # Если среди сообщений нет user-роли, добавляем заглушку (многие Jinja-шаблоны требуют минимум одно user-сообщение)
    payload_messages = list(messages)
    if not any(m.get("role") == "user" for m in payload_messages):
        payload_messages.append({"role": "user", "content": "."})
    # Выполняем минимальную генерацию (1 токен), чтобы сервер посчитал полный размер промпта (включая картинки) в usage.prompt_tokens
    r = llm["session"].post(
        f"{llm['url']}/v1/chat/completions",
        json={"messages": payload_messages, "max_tokens": 1},
    )
    r.raise_for_status()
    return r.json()["usage"]["prompt_tokens"]


def _jpeg_size(path: str) -> tuple[int, int] | None:
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                b = f.read(1)
                if not b:
                    return None
                if b != b"\xff":
                    continue
                while b == b"\xff":
                    b = f.read(1)
                if not b:
                    return None
                marker = b[0]
                if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                    continue
                seg_len_raw = f.read(2)
                if len(seg_len_raw) < 2:
                    return None
                seg_len = struct.unpack(">H", seg_len_raw)[0]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    sof = f.read(5)
                    if len(sof) < 5:
                        return None
                    h, w = struct.unpack(">HH", sof[1:5])
                    return (w, h)
                if seg_len < 2:
                    return None
                f.seek(seg_len - 2, os.SEEK_CUR)
    except OSError:
        return None


def count_image_tokens(llm: dict, data_url: str, image_path: str | None = None) -> int:
    # Кэшируем чистую стоимость одного кадра (без обвязки chat-шаблона) по разрешению кадра,
    # чтобы не гонять визуальный энкодер вхолостую на каждой паузе.
    cache = llm.setdefault("_image_tokens_cache", {})
    cache_key = _jpeg_size(image_path) if image_path else None
    if cache_key is None:
        cache_key = ("len", len(data_url) // 4096)
    if cache_key in cache:
        return cache[cache_key]

    base_tokens = count_prompt_tokens(
        llm,
        [{"role": "user", "content": [{"type": "text", "text": "."}]}],
    )
    with_img_tokens = count_prompt_tokens(
        llm,
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "."[:1]},
                ],
            }
        ],
    )
    img_tokens = max(32, with_img_tokens - base_tokens)
    cache[cache_key] = img_tokens
    return img_tokens
