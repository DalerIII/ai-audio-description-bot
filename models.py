import re
import subprocess
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

from config import (
    LLAMA_SERVER_DIR, SERVER_HOST, SERVER_PORT,
    REPO_ID, MODEL_FILE, MMPROJ_FILE,
    N_CTX, N_GPU_LAYERS, N_THREADS, FLASH_ATTN,
    MMPROJ_OFFLOAD, N_PARALLEL, REASONING_BUDGET, LOG_VERBOSITY, MAX_REPLY,
    USE_EXTRA_SERVER_ARGS, N_BATCH, N_UBATCH, USE_MMAP, USE_MLOCK, OFFLOAD_KQV,
)

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", flags=re.DOTALL)


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
) -> dict:
    if not LLAMA_SERVER_DIR:
        raise ValueError("Не указан путь LLAMA_SERVER_DIR в config.py (папка с llama-server.exe)")
    server_dir = Path(LLAMA_SERVER_DIR)
    server_exe = server_dir / "llama-server.exe"
    if not server_exe.exists():
        raise FileNotFoundError(f"Не найден llama-server.exe по пути: {server_exe}")

    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
    mmproj_path = hf_hub_download(repo_id=REPO_ID, filename=MMPROJ_FILE) if MMPROJ_FILE else None

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
        cmd += ["--reasoning-budget", str(reasoning_budget)]
    if LOG_VERBOSITY is not None:
        cmd += ["-lv", str(LOG_VERBOSITY)]
    cmd += _extra_server_args()

    server_url = f"http://{SERVER_HOST}:{SERVER_PORT}"
    proc = subprocess.Popen(cmd, cwd=str(server_dir))
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
    return {"proc": proc, "url": server_url, "session": session}


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


def _record_speed(data: dict, elapsed: float) -> None:
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
        f.write(f"{MODEL_FILE},{prompt_n},{prompt_tps:.2f},{gen_n},{gen_tps:.2f},{elapsed:.2f}\n")

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

def chat(llm: dict, messages: list, max_tokens: int = MAX_REPLY, temperature: float = 0.3) -> str:
    # Отправляем OpenAI-совместимый запрос на генерацию ответа в /v1/chat/completions
    t0 = time.perf_counter()  # [SPEED STATS]
    r = llm["session"].post(
        f"{llm['url']}/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    r.raise_for_status()
    data = r.json()
    _record_speed(data, time.perf_counter() - t0)  # [SPEED STATS]
    # Извлекаем текст ответа и вырезаем возможные блоки рассуждений <think>...</think>
    text = data["choices"][0]["message"]["content"] or ""
    return _THINK_RE.sub("", text).strip()


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


def count_image_tokens(llm: dict, data_url: str) -> int:
    # Оцениваем стоимость одного кадра в токенах через пробный чат-запрос с изображением
    return count_prompt_tokens(
        llm,
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "."},
                ],
            }
        ],
    )
