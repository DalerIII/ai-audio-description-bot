import os
import sys
import json
import asyncio
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from config import (
    APP_DIR,
    FFMPEG_DIR,
    N_CTX,
    N_GPU_LAYERS,
    REASONING_BUDGET,
    MODEL_PRESETS,
    MODEL_PRESETS_BY_ID,
    DEFAULT_MODEL_ID,
)

if FFMPEG_DIR.is_dir():
    os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")
from models import load_model, close_model
from slicing import find_pauses
from describe import describe_scene
from voice import speak
from edit_video import add_sound
from subtitles import generate_subtitles

SETTINGS_FILE = "settings.json"

DEFAULT_SETTINGS = {
    "min_pause": 5.0,          # минимальная пауза (сек)
    "frame_step": 1.0,         # шаг вырезания кадров (сек)
    "words_per_sec": 1.4,      # слов в секунду для озвучки
    "first_pause": 1,          # с какой паузы начинать
    "last_pause": 10,          # на какой закончить (0 = все)
    "border": 1.0,             # граница между монтированным и настоящим звуком
    "orig_volume": 0.3,        # громкость оригинального аудио
    "skip_subs": True,         # пропустить генерацию субтитров, если файлы уже есть
    "n_ctx": N_CTX,
    "n_gpu_layers": N_GPU_LAYERS,
    "reasoning_budget": REASONING_BUDGET,
    "model_id": DEFAULT_MODEL_ID,
    "custom_repo_id": "",
    "custom_model_file": "",
    "custom_mmproj_file": "",
}


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                s.update(json.load(f))
        except Exception:
            pass
    if s.get("model_id") not in MODEL_PRESETS_BY_ID:
        s["model_id"] = DEFAULT_MODEL_ID
    return s


def save_settings(s):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


def resolve_model(settings):
    """Возвращает (repo_id, model_file, mmproj_file, display_name) для выбранной модели."""
    model_id = settings.get("model_id", DEFAULT_MODEL_ID)
    if model_id == "custom":
        repo_id = settings.get("custom_repo_id", "").strip()
        model_file = settings.get("custom_model_file", "").strip()
        mmproj_file = settings.get("custom_mmproj_file", "").strip() or None
        return repo_id, model_file, mmproj_file, "Своя модель"
    preset = MODEL_PRESETS_BY_ID.get(model_id) or MODEL_PRESETS_BY_ID[DEFAULT_MODEL_ID]
    return preset["repo_id"], preset["model_file"], preset["mmproj_file"], preset["name"]


class Stopped(Exception):
    pass


class App:
    def __init__(self, root):
        self.root = root
        root.title("Тифлокомментарии")
        root.geometry("640x480")

        self.settings = load_settings()
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.active_llm = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.movie_var = tk.StringVar()

        # --- выбор файла ---
        top = ttk.Frame(root, padding=(10, 10, 10, 6))
        top.pack(fill="x")
        ttk.Label(top, text="Фильм:", width=11).pack(side="left")
        ttk.Entry(top, textvariable=self.movie_var).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(top, text="Выбрать...", command=self.choose_movie).pack(side="left")

        # --- выбор модели ---
        model_row = ttk.Frame(root, padding=(10, 0, 10, 8))
        model_row.pack(fill="x")
        ttk.Label(model_row, text="Модель ИИ:", width=11).pack(side="left")
        self.model_var = tk.StringVar(value=MODEL_PRESETS_BY_ID[self.settings["model_id"]]["name"])
        self.model_combo = ttk.Combobox(
            model_row,
            textvariable=self.model_var,
            values=[m["name"] for m in MODEL_PRESETS],
            state="readonly",
        )
        self.model_combo.pack(side="left", fill="x", expand=True, padx=(5, 0))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)

        # --- кнопки ---
        btns = ttk.Frame(root, padding=(10, 0))
        btns.pack(fill="x")
        self.start_btn = ttk.Button(btns, text="Старт", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Стоп", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        ttk.Button(btns, text="Настройки", command=self.open_settings).pack(side="left")

        # --- прогресс ---
        prog = ttk.Frame(root, padding=10)
        prog.pack(fill="x")
        self.progress = ttk.Progressbar(prog, maximum=100)
        self.progress.pack(fill="x")
        self.status_var = tk.StringVar(value="Готов к работе")
        ttk.Label(prog, textvariable=self.status_var).pack(anchor="w", pady=(5, 0))

        # --- лог ---
        logf = ttk.Frame(root, padding=(10, 0, 10, 10))
        logf.pack(fill="both", expand=True)
        self.log = tk.Text(logf, state="disabled", height=10)
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.root.after(100, self.poll_queue)

    # ---------- UI ----------
    def choose_movie(self):
        path = filedialog.askopenfilename(
            title="Выберите фильм",
            filetypes=[("Видео", "*.mkv *.mp4 *.avi *.mov"), ("Все файлы", "*.*")])
        if path:
            self.movie_var.set(path)

    def on_model_change(self, _event=None):
        idx = self.model_combo.current()
        if 0 <= idx < len(MODEL_PRESETS):
            self.settings["model_id"] = MODEL_PRESETS[idx]["id"]
            save_settings(self.settings)

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Настройки")
        win.resizable(False, False)
        win.grab_set()

        fields = [
            ("min_pause", "Минимальная пауза (сек)", float),
            ("frame_step", "Шаг вырезания кадров (сек)", float),
            ("words_per_sec", "Слов в секунду (озвучка)", float),
            ("first_pause", "Начать с паузы №", int),
            ("last_pause", "Закончить паузой № (0 = все)", int),
            ("border", "Граница звука (сек)", float),
            ("orig_volume", "Громкость оригинала (0–1)", float),
            ("n_ctx", "Контекст модели (n_ctx)", int),
            ("n_gpu_layers", "Слоёв на GPU (n_gpu_layers)", int),
            ("reasoning_budget", "Бюджет thinking (0 = выкл)", int),
        ]
        vars_ = {}
        frm = ttk.Frame(win, padding=10)
        frm.pack()
        for row, (key, label, _) in enumerate(fields):
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=2)
            v = tk.StringVar(value=str(self.settings[key]))
            ttk.Entry(frm, textvariable=v, width=12).grid(row=row, column=1, padx=8)
            vars_[key] = v

        skip_var = tk.BooleanVar(value=self.settings["skip_subs"])
        ttk.Checkbutton(frm, text="Пропускать генерацию субтитров, если они уже есть",
                        variable=skip_var).grid(row=len(fields), column=0, columnspan=2,
                                                sticky="w", pady=6)

        custom_frame = ttk.LabelFrame(frm, text="Своя модель (если выбрана «Своя модель»)", padding=8)
        custom_frame.grid(row=len(fields) + 1, column=0, columnspan=2, sticky="ew", pady=(4, 6))
        custom_fields = [
            ("custom_repo_id", "HF репозиторий"),
            ("custom_model_file", "Файл модели (.gguf)"),
            ("custom_mmproj_file", "Файл mmproj (.gguf)"),
        ]
        custom_vars = {}
        for crow, (key, label) in enumerate(custom_fields):
            ttk.Label(custom_frame, text=label).grid(row=crow, column=0, sticky="w", pady=2)
            v = tk.StringVar(value=str(self.settings.get(key, "")))
            ttk.Entry(custom_frame, textvariable=v, width=38).grid(row=crow, column=1, padx=(8, 0), pady=2)
            custom_vars[key] = v

        def save():
            try:
                for key, _, typ in fields:
                    self.settings[key] = typ(vars_[key].get().replace(",", "."))
            except ValueError:
                messagebox.showerror("Ошибка", "Проверьте значения: где-то введено не число.")
                return
            self.settings["skip_subs"] = skip_var.get()
            for key, _ in custom_fields:
                self.settings[key] = custom_vars[key].get().strip()
            save_settings(self.settings)
            win.destroy()

        b = ttk.Frame(frm)
        b.grid(row=len(fields) + 2, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(b, text="Сохранить", command=save).pack(side="left", padx=5)
        ttk.Button(b, text="Отмена", command=win.destroy).pack(side="left")

    # ---------- связь с потоком ----------
    def post(self, kind, value):
        self.queue.put((kind, value))

    def poll_queue(self):
        try:
            while True:
                kind, value = self.queue.get_nowait()
                if kind == "progress":
                    self.progress["value"] = value
                elif kind == "status":
                    self.status_var.set(value)
                elif kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", value + "\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                elif kind == "done":
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.model_combo.configure(state="readonly")
                elif kind == "error":
                    messagebox.showerror("Ошибка", value)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_queue)

    def start(self):
        movie = self.movie_var.get().strip()
        if not movie or not os.path.exists(movie):
            messagebox.showwarning("Нет файла", "Сначала выберите фильм.")
            return
        repo_id, model_file, _, _ = resolve_model(self.settings)
        if not repo_id or not model_file:
            messagebox.showwarning(
                "Нет модели",
                "Укажите репозиторий и файл модели в Настройках или выберите готовую модель из списка.",
            )
            return
        self.stop_event.clear()
        self.progress["value"] = 0
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.model_combo.configure(state="disabled")
        self.worker = threading.Thread(target=self.run_pipeline, args=(movie, dict(self.settings)),
                                       daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.status_var.set("Останавливаю после текущего шага...")

    def on_close(self):
        self.stop_event.set()
        close_model(self.active_llm)
        self.active_llm = None
        self.root.destroy()

    # ---------- сам пайплайн ----------
    def check_stop(self):
        if self.stop_event.is_set():
            raise Stopped()

    def run_pipeline(self, movie_path, s):
        try:
            name = os.path.splitext(os.path.basename(movie_path))[0].lower()
            sub_path = f"subtitles/{name}_subtitle.srt"
            plot_path = f"subtitles/{name}_plot.md"
            out_path = f"result/tiflo_{name}.mp4"
            os.makedirs("subtitles", exist_ok=True)
            os.makedirs("voices", exist_ok=True)
            os.makedirs("result", exist_ok=True)

            repo_id, model_file, mmproj_file, model_name = resolve_model(s)

            def loader():
                self.check_stop()
                self.post("log", f"Загружаю модель: {model_name} ({repo_id} / {model_file})...")
                llm = load_model(
                    n_ctx=s["n_ctx"],
                    n_gpu_layers=s["n_gpu_layers"],
                    reasoning_budget=s["reasoning_budget"],
                    stop_event=self.stop_event,
                    repo_id=repo_id,
                    model_file=model_file,
                    mmproj_file=mmproj_file,
                )
                self.active_llm = llm
                return llm

            # Этап 1: субтитры и сюжет (0–20%)
            self.post("status", "Этап 1/4: субтитры и сюжет...")
            if s["skip_subs"] and os.path.exists(sub_path) and os.path.exists(plot_path):
                self.post("log", "Субтитры и сюжет уже есть, пропускаю.")
            else:
                self.post("log", "Генерирую субтитры и сюжет...")
                generate_subtitles(movie_path, output_srt=sub_path,
                                   output_plot=plot_path, llm_loader=loader)
            self.post("progress", 20)
            self.check_stop()

            # Этап 2: поиск пауз (20–25%)
            self.post("status", "Этап 2/4: поиск пауз...")
            pauses = find_pauses(sub_path, s["min_pause"])
            self.post("log", f"Найдено пауз: {len(pauses)}")
            self.post("progress", 25)
            self.check_stop()

            # Этап 3: описание и озвучка (25–90%)
            self.post("status", "Загружаю модель...")
            llm = loader()
            try:
                first = max(0, s["first_pause"])
                last = len(pauses) - 1 if s["last_pause"] <= 0 else min(s["last_pause"], len(pauses) - 1)
                indices = list(range(first, last + 1))
                total = max(1, len(indices))
                context = 0

                for n, i in enumerate(indices, start=1):
                    self.check_stop()
                    start = max(0.0, pauses[i][0].total_seconds())
                    end = pauses[i][1].total_seconds()
                    words_num = int((end - start) * s["words_per_sec"])

                    self.post("status", f"Этап 3/4: пауза {n} из {total}...")
                    kwargs = dict(max_words=words_num)
                    if context != 0:
                        kwargs["context"] = context
                    summary, context = describe_scene(llm, plot_path, sub_path, movie_path,
                                                      s["min_pause"], i, s["frame_step"], **kwargs)
                    self.post("log", f"[{n}/{total}] {summary}")

                    asyncio.run(speak(summary, target_sec=end - start,
                                      out=f"voices/voice_{start}_{end}.mp3"))
                    self.post("progress", 25 + 65 * n / total)
            finally:
                close_model(llm)
                self.active_llm = None

            self.check_stop()

            # Этап 4: монтаж (90–100%)
            self.post("status", "Этап 4/4: монтаж видео...")
            add_sound(movie_path, s["border"], s["orig_volume"], out_path)
            self.post("progress", 100)
            self.post("status", f"Готово: {out_path}")
            self.post("log", f"Готово! Результат: {out_path}")

        except (Stopped, InterruptedError):
            self.post("status", "Остановлено")
            self.post("log", "Остановлено пользователем.")
        except Exception as e:
            self.post("status", "Ошибка")
            self.post("log", f"Ошибка: {e}")
            self.post("error", str(e))
        finally:
            close_model(self.active_llm)
            self.active_llm = None
            self.post("done", None)


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        os.chdir(APP_DIR)
    root = tk.Tk()
    App(root)
    root.mainloop()