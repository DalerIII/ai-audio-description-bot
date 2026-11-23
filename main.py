import os
import sys
import json
import time
import threading
import queue
import traceback
import multiprocessing
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from config import (
    APP_DIR,
    FFMPEG_DIR,
    N_CTX,
    N_GPU_LAYERS,
    REASONING_BUDGET,
    MODEL_NAME,
    REPO_ID,
    MODEL_FILE,
    MMPROJ_FILE,
)


def _ensure_stdio():
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        stream = open(APP_DIR / "tiflo.log", "a", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        stream = open(os.devnull, "w", encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    if sys.__stdout__ is None:
        sys.__stdout__ = stream
    if sys.__stderr__ is None:
        sys.__stderr__ = stream


_ensure_stdio()

if FFMPEG_DIR.is_dir():
    os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")

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
    s.pop("model_id", None)
    return s


def save_settings(s):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


def resolve_model(settings):
    """Возвращает (repo_id, model_file, mmproj_file, display_name)."""
    repo_id = settings.get("custom_repo_id", "").strip()
    model_file = settings.get("custom_model_file", "").strip()
    if repo_id and model_file:
        mmproj_file = settings.get("custom_mmproj_file", "").strip() or None
        return repo_id, model_file, mmproj_file, "Своя модель"
    return REPO_ID, MODEL_FILE, MMPROJ_FILE, MODEL_NAME


class Stopped(Exception):
    pass


class App:
    def __init__(self, root):
        self.root = root
        root.title("Тифлокомментарии")
        root.geometry("660x520")

        self.settings = load_settings()
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.active_llm = None
        self.run_started_at = None
        self.current_overall_pct = 0.0
        self._step_indeterminate = False
        self._last_timer_sec = -1

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.movie_var = tk.StringVar()

        # --- выбор файла ---
        top = ttk.Frame(root, padding=(10, 10, 10, 6))
        top.pack(fill="x")
        ttk.Label(top, text="Фильм:", width=11).pack(side="left")
        ttk.Entry(top, textvariable=self.movie_var).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(top, text="Выбрать...", command=self.choose_movie).pack(side="left")

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

        overall_hdr = ttk.Frame(prog)
        overall_hdr.pack(fill="x")
        ttk.Label(overall_hdr, text="Общая готовность:").pack(side="left")
        self.overall_info_var = tk.StringVar(value="0%")
        ttk.Label(overall_hdr, textvariable=self.overall_info_var).pack(side="right")
        self.progress = ttk.Progressbar(prog, maximum=100, mode="determinate")
        self.progress.pack(fill="x", pady=(2, 8))

        step_hdr = ttk.Frame(prog)
        step_hdr.pack(fill="x")
        self.status_var = tk.StringVar(value="Готов к работе")
        ttk.Label(step_hdr, textvariable=self.status_var).pack(side="left")
        self.step_pct_var = tk.StringVar(value="")
        ttk.Label(step_hdr, textvariable=self.step_pct_var).pack(side="right")
        self.step_progress = ttk.Progressbar(prog, maximum=100, mode="determinate")
        self.step_progress.pack(fill="x", pady=(2, 0))

        self.tokens_var = tk.StringVar(value="")
        ttk.Label(prog, textvariable=self.tokens_var).pack(anchor="w", pady=(4, 0))

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

        custom_frame = ttk.LabelFrame(frm, text=f"Своя модель (оставьте пустым для {MODEL_NAME})", padding=8)
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
            custom_repo = custom_vars["custom_repo_id"].get().strip()
            custom_file = custom_vars["custom_model_file"].get().strip()
            if bool(custom_repo) != bool(custom_file):
                messagebox.showerror(
                    "Ошибка",
                    "Для своей модели укажите и HF репозиторий, и файл модели, либо оставьте оба поля пустыми.",
                )
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
    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _update_overall_info(self):
        pct = max(0.0, min(100.0, self.current_overall_pct))
        parts = [f"{pct:.1f}%"]
        if self.run_started_at is not None:
            elapsed = max(0.0, time.monotonic() - self.run_started_at)
            parts.append(f"прошло {self._format_duration(elapsed)}")
        self.overall_info_var.set(" • ".join(parts))

    def _set_step_progress(self, step_pct: float | None):
        if step_pct is None:
            if not self._step_indeterminate:
                self.step_progress.configure(mode="indeterminate")
                self.step_progress.start(12)
                self._step_indeterminate = True
            self.step_pct_var.set("...")
        else:
            if self._step_indeterminate:
                self.step_progress.stop()
                self.step_progress.configure(mode="determinate")
                self._step_indeterminate = False
            clamped = max(0.0, min(100.0, float(step_pct)))
            self.step_progress["value"] = clamped
            self.step_pct_var.set(f"{clamped:.0f}%")

    def post(self, kind, value):
        self.queue.put((kind, value))

    def poll_queue(self):
        try:
            while True:
                kind, value = self.queue.get_nowait()
                if kind == "progress_detail":
                    overall_pct, step_pct, status_text = value
                    if overall_pct is not None:
                        self.current_overall_pct = max(0.0, min(100.0, float(overall_pct)))
                        self.progress["value"] = self.current_overall_pct
                    self._set_step_progress(step_pct)
                    if status_text is not None:
                        self.status_var.set(status_text)
                    self._update_overall_info()
                elif kind == "progress":
                    self.current_overall_pct = max(0.0, min(100.0, float(value)))
                    self.progress["value"] = self.current_overall_pct
                    self._update_overall_info()
                elif kind == "status":
                    self.status_var.set(value)
                elif kind == "tokens":
                    self.tokens_var.set(value)
                elif kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", value + "\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                elif kind == "done":
                    if self._step_indeterminate:
                        self.step_progress.stop()
                        self.step_progress.configure(mode="determinate")
                        self._step_indeterminate = False
                        self.step_progress["value"] = 0
                        self.step_pct_var.set("")
                    self.run_started_at = None
                    self._last_timer_sec = -1
                    self._update_overall_info()
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                elif kind == "error":
                    messagebox.showerror("Ошибка", value)
        except queue.Empty:
            pass
        if self.run_started_at is not None:
            sec = int(time.monotonic() - self.run_started_at)
            if sec != self._last_timer_sec:
                self._last_timer_sec = sec
                self._update_overall_info()
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
        self.run_started_at = time.monotonic()
        self._last_timer_sec = -1
        self.current_overall_pct = 0.0
        self.progress["value"] = 0
        self._set_step_progress(0.0)
        self._update_overall_info()
        self.tokens_var.set("")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.worker = threading.Thread(target=self.run_pipeline, args=(movie, dict(self.settings)),
                                       daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.status_var.set("Останавливаю после текущего шага...")

    def on_close(self):
        self.stop_event.set()
        if self.active_llm is not None:
            from models import close_model
            close_model(self.active_llm)
            self.active_llm = None
        self.root.destroy()

    # ---------- сам пайплайн ----------
    def check_stop(self):
        if self.stop_event.is_set():
            raise Stopped()

    def run_pipeline(self, movie_path, s):
        try:
            from models import load_model, close_model

            name = os.path.splitext(os.path.basename(movie_path))[0].lower()
            sub_path = f"subtitles/{name}_subtitle.srt"
            plot_path = f"subtitles/{name}_plot.md"
            out_path = f"result/tiflo_{name}.mp4"
            os.makedirs("subtitles", exist_ok=True)
            os.makedirs("voices", exist_ok=True)
            os.makedirs("result", exist_ok=True)

            repo_id, model_file, mmproj_file, model_name = resolve_model(s)
            skip_stage1 = bool(s["skip_subs"] and os.path.exists(sub_path) and os.path.exists(plot_path))
            if skip_stage1:
                s1_range = (0.0, 0.0)
                s2_range = (0.0, 2.0)
                s3_load_range = (2.0, 5.0)
                s3_pauses_range = (5.0, 90.0)
                s4_range = (90.0, 100.0)
            else:
                s1_range = (0.0, 50.0)
                s2_range = (50.0, 52.0)
                s3_load_range = (52.0, 54.0)
                s3_pauses_range = (54.0, 92.0)
                s4_range = (92.0, 100.0)

            def report(overall_pct, step_pct, text):
                self.post("progress_detail", (overall_pct, step_pct, text))

            def map_range(rng, frac):
                a, b = rng
                f = max(0.0, min(1.0, float(frac)))
                return a + (b - a) * f

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
                    on_stats=lambda text: self.post("tokens", text),
                )
                self.active_llm = llm
                return llm

            # Этап 1: субтитры и сюжет
            if skip_stage1:
                self.post("log", "Субтитры и сюжет уже есть, пропускаю.")
                report(s1_range[1], 100.0, "Этап 1/4 • Субтитры и сюжет уже есть (пропуск)")
            else:
                from subtitles import generate_subtitles
                self.check_stop()
                self.post("log", "Генерирую субтитры и сюжет...")
                report(s1_range[0], None, "Этап 1/4 • Подготовка субтитров и сюжета...")

                def on_stage1(stage_frac, step_frac, text):
                    report(
                        map_range(s1_range, stage_frac),
                        None if step_frac is None else step_frac * 100.0,
                        f"Этап 1/4 • {text}",
                    )

                generate_subtitles(
                    movie_path,
                    output_srt=sub_path,
                    output_plot=plot_path,
                    llm_loader=loader,
                    stop_event=self.stop_event,
                    on_progress=on_stage1,
                )
            self.check_stop()

            # Этап 2: поиск пауз
            report(s2_range[0], None, "Этап 2/4 • Поиск пауз...")
            from slicing import find_pauses
            pauses = find_pauses(sub_path, s["min_pause"])
            self.post("log", f"Найдено пауз: {len(pauses)}")
            report(s2_range[1], 100.0, f"Этап 2/4 • Найдено пауз: {len(pauses)}")
            self.check_stop()

            # Этап 3: описание и озвучка
            report(s3_load_range[0], None, "Этап 3/4 • Загрузка модели...")
            import asyncio
            from describe import describe_scene
            from voice import speak
            llm = loader()
            report(s3_load_range[1], 100.0, "Этап 3/4 • Модель загружена")
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

                    p_start = map_range(s3_pauses_range, (n - 1) / total)
                    p_end = map_range(s3_pauses_range, n / total)
                    p_span = p_end - p_start

                    def on_pause_progress(frac, text, _p_start=p_start, _p_span=p_span, _n=n, _total=total):
                        f = max(0.0, min(1.0, float(frac)))
                        report(
                            _p_start + _p_span * 0.85 * f,
                            f * 85.0,
                            f"Этап 3/4 • Пауза {_n}/{_total}: {text}",
                        )

                    kwargs = dict(max_words=words_num, on_progress=on_pause_progress)
                    if context != 0:
                        kwargs["context"] = context
                    summary, context = describe_scene(llm, plot_path, sub_path, movie_path,
                                                      s["min_pause"], i, s["frame_step"], **kwargs)
                    self.post("log", f"[{n}/{total}] {summary}")

                    report(p_start + p_span * 0.85, 85.0, f"Этап 3/4 • Пауза {n}/{total}: озвучка (TTS)...")
                    asyncio.run(speak(summary, target_sec=end - start,
                                      out=f"voices/voice_{start}_{end}.mp3"))
                    report(p_end, 100.0, f"Этап 3/4 • Пауза {n}/{total}: готово")
            finally:
                close_model(llm)
                self.active_llm = None

            self.check_stop()

            # Этап 4: монтаж
            from edit_video import add_sound

            def on_stage4(stage_frac, step_frac, text):
                report(
                    map_range(s4_range, stage_frac),
                    None if step_frac is None else step_frac * 100.0,
                    f"Этап 4/4 • {text}",
                )

            add_sound(movie_path, s["border"], s["orig_volume"], out_path, on_progress=on_stage4)
            report(100.0, 100.0, f"Готово: {out_path}")
            self.post("log", f"Готово! Результат: {out_path}")

        except (Stopped, InterruptedError):
            self.post("progress_detail", (None, 0.0, "Остановлено"))
            self.post("log", "Остановлено пользователем.")
        except Exception as e:
            traceback.print_exc()
            self.post("progress_detail", (None, 0.0, "Ошибка"))
            self.post("log", f"Ошибка: {e}")
            self.post("error", str(e))
        finally:
            if self.active_llm is not None:
                from models import close_model
                close_model(self.active_llm)
                self.active_llm = None
            self.post("done", None)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if getattr(sys, "frozen", False):
        os.chdir(APP_DIR)
    root = tk.Tk()
    App(root)
    root.mainloop()