import sys
from pathlib import Path

APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
FFMPEG_DIR = APP_DIR / "bin" / "ffmpeg"

# ---- ваши существующие значения (подставьте свои) ----
N_CTX = 8192
RESERVED_TOKENS = 8192
SAFETY_MARGIN = 300
MAX_REPLY = max(1500, RESERVED_TOKENS // 10)

TOKENS_PER_WORD = 3        # запас для русского текста

# ---- настройки загрузки модели (значения-заглушки, подставьте свои) ----
LLAMA_SERVER_DIR = APP_DIR / "bin" / "llama"  # папка с llama-server.exe
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080

MODEL_PRESETS = [
    {
        "id": "qwen35-9b",
        "name": "Qwen 3.5 9B (Q4_K_M, ~6.5 ГБ) — рекомендуемая",
        "repo_id": "unsloth/Qwen3.5-9B-GGUF",
        "model_file": "Qwen3.5-9B-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
    },
    {
        "id": "qwen35-4b",
        "name": "Qwen 3.5 4B (Q4_K_M, ~3.5 ГБ) — лёгкая и быстрая",
        "repo_id": "unsloth/Qwen3.5-4B-GGUF",
        "model_file": "Qwen3.5-4B-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
    },
    {
        "id": "qwen3-vl-8b",
        "name": "Qwen3-VL 8B Instruct (Q4_K_M, ~6.2 ГБ)",
        "repo_id": "unsloth/Qwen3-VL-8B-Instruct-GGUF",
        "model_file": "Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
    },
    {
        "id": "qwen25-vl-7b",
        "name": "Qwen2.5-VL 7B Instruct (Q4_K_M, ~5.5 ГБ) — классика",
        "repo_id": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "model_file": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
    },
    {
        "id": "gemma4-e4b",
        "name": "Gemma 4 E4B (Q4_K_M, ~5.5 ГБ) — Google",
        "repo_id": "unsloth/gemma-4-E4B-it-GGUF",
        "model_file": "gemma-4-E4B-it-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
    },
    {
        "id": "gemma3-4b",
        "name": "Gemma 3 4B (Q4_K_M, ~3.3 ГБ) — ультралёгкая",
        "repo_id": "unsloth/gemma-3-4b-it-GGUF",
        "model_file": "gemma-3-4b-it-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
    },
    {
        "id": "custom",
        "name": "Своя модель (Hugging Face — из настроек)",
        "repo_id": "",
        "model_file": "",
        "mmproj_file": "",
    },
]

MODEL_PRESETS_BY_ID = {m["id"]: m for m in MODEL_PRESETS}
DEFAULT_MODEL_ID = "qwen35-9b"

REPO_ID = MODEL_PRESETS_BY_ID[DEFAULT_MODEL_ID]["repo_id"]
MODEL_FILE = MODEL_PRESETS_BY_ID[DEFAULT_MODEL_ID]["model_file"]
MMPROJ_FILE = MODEL_PRESETS_BY_ID[DEFAULT_MODEL_ID]["mmproj_file"]

N_GPU_LAYERS = 99          # -1 - все слои на GPU, 0 - только CPU
N_THREADS = 6              # потоков CPU
FLASH_ATTN = True          # flash attention
MMPROJ_OFFLOAD = False     # выгружать mmproj на GPU (False = --no-mmproj-offload)
N_PARALLEL = 1             # число параллельных слотов (-np)
REASONING_BUDGET = -1      # бюджет токенов на thinking (0 = выкл, -1 = без лимита)
LOG_VERBOSITY = 4          # уровень логирования (-lv, None = не передавать)

# === [EXTRA SERVER ARGS — "чёрный ящик", включите USE_EXTRA_SERVER_ARGS = True для активации] ===
USE_EXTRA_SERVER_ARGS = False
N_BATCH = 512              # размер батча при обработке промпта
N_UBATCH = 512             # физический микро-батч (<= N_BATCH)
USE_MMAP = True            # mmap для весов
USE_MLOCK = False          # блокировать модель в RAM
OFFLOAD_KQV = True         # KV-кэш на GPU
# === [/EXTRA SERVER ARGS] ===

# ---- бюджеты контекста по сюжету и субтитрам (в токенах) ----
PLOT_PAST_RECENT_TOKENS = 500    # недавние эпизоды, дословно
PLOT_PAST_SUMMARY_WORDS = 150    # размер сжатой предыстории
PLOT_FOLD_CHUNK_TOKENS = RESERVED_TOKENS // 2   # порция сюжета за один проход сжатия
PLOT_NOW_TOKENS = 500            # эпизод(ы), в которые попадает сцена
PLOT_FUTURE_TOKENS = 300         # ближайшее будущее
PLOT_FUTURE_MAX_ENTRIES = 2      # сколько следующих эпизодов показывать

SUBS_BEFORE_SEC = 40             # окно субтитров до сцены
SUBS_AFTER_SEC = 40              # окно субтитров после сцены
SUBS_TOKENS = 400                # общий бюджет субтитров

FRAME_MAX_SIDE = 768             # макс. размер большей стороны кадра (None - не менять)


# ---------------------------------------------------------------- промпты
PLOT_CONTEXT_TEMPLATE = """Представь, что ты созваниваешься по телефону с другом. Он слышит звуки из фильма, но не видит экран. Опиши ему новые кадры по очереди.

Контекст: {context}
Предыдущие кадры [ctx]: Если здесь есть текст, первый новый кадр сравнивай с последним кадром отсюда.

ЧТО ДЕЛАТЬ:
- Пиши строго по одной строчке на кадр в формате: «Кадр N: [описание]».
- Иди строго по порядку номеров. Если в кадре нет важных изменений, пиши: «Кадр N: без значимых изменений».
- Описывай только КРУПНЫЕ события: кто куда переместился, какую позу принял, что взял или сделал.
- Называй эмоции героев одним простым словом (рад, испуган, злится).
- Пиши короткими простыми предложениями, как очевидец. Имена бери из контекста.

ИГНОРИРУЙ И НЕ УПОМИНАЙ:
- Цвета, свет, темноту или освещение.
- Внешность, возраст, одежду (если с ней нет действия).
- Звуки, музыку, диалоги и реплики персонажей.
- Работу камеры, ракурсы, планы и монтаж.
- Вступления, заключения, мысли персонажей и рассуждения о сюжете.

Твое описание кадров для друга:"""

PLOT_COMPRESS_PROMPT = """Ты сжимаешь предысторию сюжета. Тебе дана уже сжатая предыстория и новые события. Объедини их в один связный текст не длиннее {max_words} слов.

Сохраняй: имена, кто кому кем приходится, важные предметы, места, текущие цели и отношения персонажей, значимые события в порядке их появления. Мелочи и повторы выбрасывай. Ничего не выдумывай и не добавляй от себя. Ответ - только текст предыстории."""

SCENE_SUMMARIZER_TEMPLATE = """Представь, что ты созваниваешься по телефону с другом. Он слышит голоса и звуки из фильма, но не видит экран. Перескажи ему покадровое описание в короткий связный рассказ (до {max_words} слов). Описывай только СОБЫТИЯ, а не мелкие движения.

Контекст: {ctx}
Описание: {scenes_text}

ЧТО ДЕЛАТЬ:
- Пиши строго по порядку, как идут события в тексте.
- Описывай только крупные действия: кто где находится, куда переместился, какую позу принял, что взял или сделал.
- Объединяй мелкие движения (жесты рук, повороты головы, взгляды) в одно общее событие.
- Называй эмоции одним простым словом (рад, испуган, расстроен). Если эмоция изменилась — отметь это.
- Пиши простыми короткими предложениями, без лишних слов.

ИГНОРИРУЙ И НЕ УПОМИНАЙ:
- Цвета, свет, темноту или освещение.
- Внешность, возраст, одежду (если с ней нет действия).
- Звуки, музыку, диалоги и реплики персонажей.
- Работу камеры, ракурсы и монтаж.
- Вступления, заключения и свои рассуждения.

Твой пересказ для друга:"""