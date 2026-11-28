import base64
import os
import re
from math import floor
from models import (
    chat,
    count_tokens,
    cut_to_tokens,
    count_image_tokens,
)
from slicing import slice, load_subtitles
from plot import parse_plot, locate_plot, fmt_entry, fmt_time
from config import (
    N_CTX, MAX_REPLY, SAFETY_MARGIN, TOKENS_PER_WORD, REASONING_BUDGET,
    LAST_SUMMARY_COUNT,
    DEFAULT_COMMENT_LANGUAGE, get_comment_prompts, normalize_comment_lang,
    PLOT_PAST_RECENT_TOKENS, PLOT_PAST_SUMMARY_WORDS,
    PLOT_NOW_TOKENS, PLOT_FUTURE_TOKENS, PLOT_FUTURE_MAX_ENTRIES,
    SUBS_BEFORE_SEC, SUBS_AFTER_SEC, SUBS_TOKENS,
)

TOKENS_PER_FRAME_NOTE = 65

_SKIP_RE = re.compile(
    r"^[\s«\"'—–\-]*(?:"
    r"пропуск|без\s+(?:значимых\s+)?изменений|никаких\s+(?:других\s+)?событий|"
    r"больше\s+ничего\s+не\s+происходит|ничего\s+не\s+происходит|"
    r"skip|no\s+(?:significant\s+)?changes|nothing\s+(?:else\s+)?happens|no\s+other\s+events|"
    r"omitir|sin\s+cambios(?:\s+significativos)?|no\s+(?:pasa|ocurre)\s+nada(?:\s+más)?|"
    r"ningún\s+otro\s+evento"
    r")[\s.»\"'!…]*$",
    re.IGNORECASE,
)


def is_skip_summary(text: str) -> bool:
    if not text or not text.strip():
        return True
    return bool(_SKIP_RE.fullmatch(text.strip()))


# ---------------------------------------------------------------- утилиты
def image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = path.split(".")[-1].lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64,{b64}"


def format_ctx(ctx: str) -> str:
    return f"Контекст сцены:\n{ctx}"


def count_words(text: str) -> int:
    return len(text.split())


def take_fit(llm, lines, budget, from_end=False):
    """Берёт строки (с начала или с конца), пока влезают в бюджет токенов."""
    picked, used = [], 0
    seq = reversed(lines) if from_end else lines
    for line in seq:
        t = count_tokens(llm, line) + 1
        if used + t > budget:
            break
        picked.append(line)
        used += t

    if not picked and lines:
        # единственная строка длиннее бюджета - обрезаем, но не теряем совсем
        nearest = lines[-1] if from_end else lines[0]
        picked = [cut_to_tokens(llm, nearest, budget)]

    return picked[::-1] if from_end else picked


# ------------------------------------------- контекст по сюжету и субтитрам
def _condense_entry_line(entry) -> str:
    """Короткая выжимка одного старого эпизода без вызова LLM (первая смысловая часть)."""
    raw = entry.text.strip()
    first_clause = re.split(r"[;.]\s+", raw, maxsplit=1)[0].strip()
    if len(first_clause) > 140:
        first_clause = first_clause[:137].rstrip(" ,;:-") + "..."
    return f"[{fmt_time(entry.start)}] {first_clause}"


def plot_past_context(llm, past, context=None):
    """Недавнее прошлое — дословно, более ранние эпизоды — компактной хронологией без вызовов LLM."""
    lines = [fmt_entry(e) for e in past]
    recent = take_fit(llm, lines, PLOT_PAST_RECENT_TOKENS, from_end=True) if lines else []
    older_count = len(lines) - len(recent)
    if older_count <= 0:
        return "", recent, 0

    older_entries = past[:older_count]
    short_lines = [_condense_entry_line(e) for e in older_entries]
    budget_tokens = PLOT_PAST_SUMMARY_WORDS * TOKENS_PER_WORD

    # Если все короткие строки влезают — берём все; иначе равномерно прореживаем с сохранением завязки и конца
    total_tok = sum(count_tokens(llm, s) + 1 for s in short_lines)
    if total_tok <= budget_tokens:
        picked = short_lines
    else:
        keep_n = max(2, min(len(short_lines), int(len(short_lines) * budget_tokens / max(1, total_tok))))
        if keep_n >= len(short_lines):
            picked = short_lines
        else:
            idxs = sorted({round(i * (len(short_lines) - 1) / (keep_n - 1)) for i in range(keep_n)})
            picked = [short_lines[i] for i in idxs]
        picked = take_fit(llm, picked, budget_tokens, from_end=True)

    return "\n".join(picked), recent, older_count


def subtitles_window(llm, subs, start, end):
    def clean(text):
        return " ".join(re.sub(r"<[^>]+>|\{[^}]+\}", "", text).split())

    before, during, after = [], [], []
    for sub in subs:
        s, e = sub.start.total_seconds(), sub.end.total_seconds()
        text = clean(sub.content)
        if not text:
            continue
        line = f"[{fmt_time(s)}] {text}"
        if e <= start + 1e-3 and s >= start - SUBS_BEFORE_SEC:
            before.append(line)
        elif s >= end - 1e-3 and s <= end + SUBS_AFTER_SEC:
            after.append(line)
        elif e > start + 1e-3 and s < end - 1e-3:
            during.append(line)

    before = take_fit(llm, before, SUBS_TOKENS // 2, from_end=True)
    during = take_fit(llm, during, SUBS_TOKENS // 4)
    after = take_fit(llm, after, SUBS_TOKENS // 2)
    return before, during, after


def build_scene_context(llm, entries, subtitles_path, start, end, context=None):
    past, now, future, plot_mode = locate_plot(entries, start, end)
    summary, recent, folded = plot_past_context(llm, past, context)
    recent_summaries = list((context or {}).get("recent_summaries") or [])
    before, during, after = subtitles_window(llm, load_subtitles(subtitles_path), start, end)

    parts = [f"Сцена в кадрах (пауза без речи): {fmt_time(start)} - {fmt_time(end)}"]

    if recent_summaries:
        parts.append(
            "Предыдущие озвученные комментарии (не повторяй уже сказанное; "
            "если локация сменилась — назови новую):\n" + "\n".join(recent_summaries)
        )
    if summary:
        parts.append(f"Предыстория фильма (ранние эпизоды):\n{summary}")
    if recent:
        parts.append("Недавние события сюжета:\n" + "\n".join(recent))

    now_lines = take_fit(llm, [fmt_entry(e) for e in now], PLOT_NOW_TOKENS) if now else []
    if now_lines:
        if plot_mode == "inside":
            parts.append(
                "Сюжет на момент сцены (пауза находится внутри этого эпизода):\n" + "\n".join(now_lines)
            )
        elif plot_mode == "between" and len(now_lines) >= 2:
            parts.append(
                "Сюжет на момент сцены (пауза на стыке двух эпизодов — в кадрах завершение первой сцены, "
                "переход или безмолвное начало второй сцены до первой реплики):\n"
                f"• Перед паузой завершился диалог: {now_lines[0]}\n"
                f"• Сразу после паузы начнётся диалог: {now_lines[1]}"
            )
        elif plot_mode == "before_first":
            parts.append(
                "Сюжет на момент сцены (начало фильма до первой реплики — в кадрах завязка этого эпизода):\n"
                + "\n".join(now_lines)
            )
        elif plot_mode == "after_last":
            parts.append(
                "Сюжет на момент сцены (финал фильма после последней реплики — в кадрах завершение этого эпизода):\n"
                + "\n".join(now_lines)
            )
        else:
            parts.append("Сюжет на момент сцены:\n" + "\n".join(now_lines))
    else:
        parts.append("Файл сюжета пуст — опирайся на кадры и субтитры.")

    future_lines = take_fit(
        llm, [fmt_entry(e) for e in future[:PLOT_FUTURE_MAX_ENTRIES]], PLOT_FUTURE_TOKENS
    ) if future else []
    if future_lines:
        parts.append(
            "Дальнейшие эпизоды (только ориентир по именам и связям, не описывать заранее):\n"
            + "\n".join(future_lines)
        )

    if before:
        parts.append("Реплики прямо перед паузой:\n" + "\n".join(before))
    if during:
        parts.append("Реплики внутри окна:\n" + "\n".join(during))
    if after:
        parts.append("Реплики сразу после паузы:\n" + "\n".join(after))

    return "\n\n".join(parts), {
        "past_summary": summary,
        "folded": folded,
        "recent_summaries": recent_summaries,
    }


# ------------------------------------------- комментарии по кадрам
def _append_frame_part(content_parts: list, n: int, start: float, shift: float, prefix: str = "Кадр") -> None:
    label = f"{prefix} {n} ({fmt_time(start + (n - 1) * shift)}):"
    content_parts.append({"type": "text", "text": label})
    content_parts.append({"type": "image_url", "image_url": {"url": image_to_data_url(f"frames/frame{n}.jpg")}})


def _build_batch_messages(
    ctx_text: str,
    first: int,
    last: int,
    start: float,
    shift: float,
    prev_notes: list[str] | None = None,
    overlap_frame: int | None = None,
    lang: str = DEFAULT_COMMENT_LANGUAGE,
) -> list[dict]:
    lang = normalize_comment_lang(lang)
    prompts = get_comment_prompts(lang)
    frame_prefix = prompts["frame_prefix"]
    ref_prefix = prompts["ref_prefix"]

    content_parts: list[dict] = [{"type": "text", "text": format_ctx(ctx_text)}]

    if prev_notes:
        joined_prev = "\n".join(prev_notes).strip()
        if lang == "en":
            prev_header = "Already described previous frames of this same pause (use for continuity and do not re-describe):\n"
        elif lang == "es":
            prev_header = "Fotogramas previos ya descritos de esta misma pausa (úsalos para continuidad y no los vuelvas a describir):\n"
        else:
            prev_header = "Уже описанные предыдущие кадры этой же паузы (учитывай их для связности и не описывай заново):\n"
        content_parts.append({"type": "text", "text": f"{prev_header}{joined_prev}"})

    if overlap_frame is not None:
        if lang == "en":
            ref_text = (
                f"Reference frame from the end of the previous batch (already described above — use ONLY to see "
                f"movement and posture changes between Frame {overlap_frame} and Frame {first}):"
            )
        elif lang == "es":
            ref_text = (
                f"Fotograma de referencia del final del lote anterior (ya descrito arriba — úsalo SOLO para ver "
                f"el movimiento y cambio de postura entre el Fotograma {overlap_frame} y el Fotograma {first}):"
            )
        else:
            ref_text = (
                f"Опорный кадр с конца предыдущего батча (уже описан выше — используй ТОЛЬКО чтобы увидеть "
                f"перемещение и смену позы между Кадром {overlap_frame} и Кадром {first}):"
            )
        content_parts.append({"type": "text", "text": ref_text})
        _append_frame_part(content_parts, overlap_frame, start, shift, prefix=ref_prefix)

    for n in range(first, last + 1):
        _append_frame_part(content_parts, n, start, shift, prefix=frame_prefix)

    if overlap_frame is None:
        if lang == "en":
            closing = (
                f"Describe frames in order from Frame {first} to Frame {last} (strictly one line per frame: "
                f"\"Frame N: ...\") in English. Compare adjacent frames to capture movement, position changes, "
                f"and gestures, and refer to characters by their names from the context."
            )
        elif lang == "es":
            closing = (
                f"Describe en orden los fotogramas del Fotograma {first} al Fotograma {last} (estrictamente una línea "
                f"por fotograma: «Fotograma N: ...») en español. Compara fotogramas contiguos para describir con "
                f"precisión el movimiento, los desplazamientos y los gestos, y nombra a los personajes según el contexto."
            )
        else:
            closing = (
                f"Опиши по порядку кадры с Кадра {first} по Кадр {last} (строго по одной строке «Кадр N: ...»). "
                f"Сравнивай соседние кадры между собой, чтобы точно описать движение, перемещения и жесты, "
                f"и называй героев текущей сцены по именам из контекста."
            )
    else:
        if lang == "en":
            closing = (
                f"Compare Frame {first} with reference Frame {overlap_frame}, then each subsequent frame with the previous one. "
                f"Output descriptions ONLY for the new frames from Frame {first} to Frame {last} in English "
                f"(strictly one line per frame: \"Frame N: ...\", without repeating Frame {overlap_frame})."
            )
        elif lang == "es":
            closing = (
                f"Compara el Fotograma {first} con el Fotograma de referencia {overlap_frame} y luego cada fotograma "
                f"siguiente con el anterior. Escribe la descripción SOLO para los nuevos fotogramas del Fotograma {first} "
                f"al Fotograma {last} en español (estrictamente «Fotograma N: ...», sin repetir el Fotograma {overlap_frame})."
            )
        else:
            closing = (
                f"Сравни Кадр {first} с опорным Кадром {overlap_frame}, а затем каждый следующий кадр с предыдущим. "
                f"Выведи описание ТОЛЬКО для новых кадров с Кадра {first} по Кадр {last} "
                f"(строго по одной строке «Кадр N: ...», без повторного описания Кадра {overlap_frame})."
            )
    content_parts.append({"type": "text", "text": closing})

    return [
        {"role": "system", "content": prompts["plot_context"]},
        {"role": "user", "content": content_parts},
    ]


def image_comment(llm, ctx_text, start, shift, on_progress=None, lang=DEFAULT_COMMENT_LANGUAGE):
    prompts = get_comment_prompts(lang)
    total_frames = sum(1 for f in os.listdir("frames") if re.fullmatch(r"frame\d+\.jpg", f))
    if total_frames == 0:
        raise RuntimeError("Не удалось вырезать ни одного кадра для сцены.")

    first_frame_path = "frames/frame1.jpg"
    img = image_to_data_url(first_frame_path)
    tokens_per_image = count_image_tokens(llm=llm, data_url=img, image_path=first_frame_path)
    tokens_per_image += count_tokens(llm, f"{prompts['frame_prefix']} 000 (0:00:00):\n")

    n_ctx = llm.get("n_ctx", N_CTX)
    rb = llm.get("reasoning_budget", REASONING_BUDGET)
    think_reserve = 0 if rb == 0 else (rb if (rb is not None and rb > 0) else 1024)

    base_text_tokens = (
        count_tokens(llm, prompts["plot_context"])
        + count_tokens(llm, format_ctx(ctx_text))
        + 180  # запас на финальную инструкцию и обвязку шаблона чата
    )
    available = n_ctx - base_text_tokens - think_reserve - SAFETY_MARGIN
    per_frame_cost = tokens_per_image + TOKENS_PER_FRAME_NOTE

    max_slots_single = floor(available / per_frame_cost)
    if max_slots_single < 1:
        return "Слишком маленькое контекстное окно!"

    # Все кадры паузы помещаются в один запрос
    if total_frames <= max_slots_single:
        messages = _build_batch_messages(ctx_text, 1, total_frames, start, shift, lang=lang)
        reply_cap = min(MAX_REPLY, max(256, total_frames * 80))
        if on_progress is not None:
            on_progress(0.0, f"анализ кадров ({total_frames} шт.)...")
        res = chat(llm, messages, max_tokens=reply_cap, temperature=0.3)
        if on_progress is not None:
            on_progress(1.0, "кадры описаны")
        return res

    # Несколько батчей с визуальным нахлёстом в 1 кадр (overlap) и краткой сводкой прошлых батчей
    prev_notes_budget = 350
    slots_multi = floor((available - prev_notes_budget) / per_frame_cost)
    # В 1-м батче все слоты идут под новые кадры; со 2-го батча 1 слот занимает опорный кадр нахлёста
    new_per_next_batch = max(1, slots_multi - 1)
    first_batch_size = max(1, slots_multi)

    ranges = []
    cur = 1
    while cur <= total_frames:
        step_size = first_batch_size if cur == 1 else new_per_next_batch
        end_f = min(cur + step_size - 1, total_frames)
        ranges.append((cur, end_f))
        cur = end_f + 1

    scene_log = []
    total_batches = len(ranges)
    for batch_idx, (first, last) in enumerate(ranges, start=1):
        if on_progress is not None:
            on_progress((batch_idx - 1) / total_batches, f"анализ кадров (батч {batch_idx}/{total_batches})...")

        prev_lines = []
        if scene_log:
            all_prev_lines = [ln for chunk in scene_log for ln in chunk.splitlines() if ln.strip()]
            prev_lines = take_fit(llm, all_prev_lines, prev_notes_budget, from_end=True)

        overlap_frame = (first - 1) if first > 1 else None
        messages = _build_batch_messages(
            ctx_text,
            first,
            last,
            start,
            shift,
            prev_notes=prev_lines or None,
            overlap_frame=overlap_frame,
            lang=lang,
        )
        reply_cap = min(MAX_REPLY, max(256, (last - first + 1) * 80))
        temp_summary = chat(llm, messages, max_tokens=reply_cap, temperature=0.3)
        scene_log.append(temp_summary)
        if on_progress is not None:
            on_progress(batch_idx / total_batches, f"анализ кадров (батч {batch_idx}/{total_batches})")

    return scene_log


def summarizer(llm, scene_log, max_words, ctx_text, allow_skip=True, lang=DEFAULT_COMMENT_LANGUAGE):
    lang = normalize_comment_lang(lang)
    prompts = get_comment_prompts(lang)
    if isinstance(scene_log, str):
        scene_log = [scene_log]
    scene_text = "\n".join(scene_log)

    prompt = prompts["summarizer"].format(max_words=max_words, scenes_text=scene_text, ctx=ctx_text)
    if not allow_skip:
        if lang == "en":
            prompt += (
                "\n\nIMPORTANT: this is the first audio description — replying \"SKIP\" is FORBIDDEN. "
                "Briefly state the setting, who is on screen, and what is happening."
            )
        elif lang == "es":
            prompt += (
                "\n\nIMPORTANTE: este es el primer comentario de audiodescripción — responder «OMITIR» está PROHIBIDO. "
                "Indica brevemente el lugar, quién aparece en pantalla y qué ocurre."
            )
        else:
            prompt += (
                "\n\nВАЖНО: это первый тифлокомментарий — отвечать «ПРОПУСК» ЗАПРЕЩЕНО. "
                "Кратко назови место действия, кто в кадре и что происходит."
            )
    messages = [{"role": "user", "content": prompt}]
    res = chat(llm, messages, max_tokens=MAX_REPLY, temperature=0.3)

    if not allow_skip and is_skip_summary(res):
        if lang == "en":
            retry_prompt = (
                f"Briefly describe for a blind listener in English (up to {max_words} words, present tense, no emotion labels) "
                f"where the scene takes place, who is on screen, and what happens. Do NOT reply \"SKIP\".\n\n"
                f"Context:\n{ctx_text}\n\nFrame-by-frame notes:\n{scene_text}"
            )
        elif lang == "es":
            retry_prompt = (
                f"Describe brevemente en español para un oyente ciego (hasta {max_words} palabras, presente de indicativo, sin etiquetas de emociones) "
                f"dónde transcurre la escena, quién está en pantalla y qué ocurre. NO respondas «OMITIR».\n\n"
                f"Contexto:\n{ctx_text}\n\nDescripción fotograma a fotograma:\n{scene_text}"
            )
        else:
            retry_prompt = (
                f"Опиши кратко для незрячего слушателя (до {max_words} слов, настоящее время, без оценок эмоций), "
                f"где происходит сцена, кто в кадре и что происходит. Отвечать «ПРОПУСК» нельзя.\n\n"
                f"Контекст:\n{ctx_text}\n\nПокадровое описание:\n{scene_text}"
            )
        res = chat(llm, [{"role": "user", "content": retry_prompt}], max_tokens=MAX_REPLY, temperature=0.3)

    return res


def proofread_summary(llm, text: str, lang: str = DEFAULT_COMMENT_LANGUAGE) -> str:
    if not text or not text.strip() or is_skip_summary(text):
        return text

    prompts = get_comment_prompts(lang)
    max_tokens = min(MAX_REPLY, max(128, count_tokens(llm, text) * 2 + 64))
    corrected = chat(
        llm,
        messages=[
            {"role": "system", "content": prompts["grammar"]},
            {"role": "user", "content": text},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    ).strip()

    # Убираем случайные внешние кавычки, если их не было в исходном тексте
    if len(corrected) >= 2 and corrected[0] in '«"' and corrected[-1] in '»"' and text[0] not in '«"':
        corrected = corrected[1:-1].strip()

    if not corrected or is_skip_summary(corrected) or len(corrected) > len(text) * 2 + 80:
        return text
    return corrected


# ------------------------------------------- полный проход по одной паузе
def describe_scene(llm, plot_path, subtitles_path, video_path, period,
                   pause_index, shift, max_words, context=None, on_progress=None,
                   lang=DEFAULT_COMMENT_LANGUAGE):
    lang = normalize_comment_lang(lang)

    def emit(frac, text):
        if on_progress is not None:
            on_progress(frac, text)

    entries = parse_plot(plot_path)
    emit(0.0, "вырезаю кадры...")
    start, end = slice(pause_index, shift, subtitles_path, period, video_path)

    emit(0.15, "собираю контекст сцены...")
    ctx_text, context = build_scene_context(
        llm, entries, subtitles_path, start, end, context)

    scene_log = image_comment(
        llm,
        ctx_text,
        start,
        shift,
        on_progress=(lambda f, t: emit(0.25 + 0.55 * f, t)) if on_progress is not None else None,
        lang=lang,
    )
    emit(0.80, "итоговый пересказ сцены...")
    allow_skip = bool(context.get("recent_summaries"))
    summary = summarizer(llm, scene_log, max_words, ctx_text, allow_skip=allow_skip, lang=lang)

    if is_skip_summary(summary):
        emit(1.0, "описание сцены готово")
        return "", context

    emit(0.92, "проверка грамматики...")
    summary = proofread_summary(llm, summary, lang=lang)
    emit(1.0, "описание сцены готово")

    recent = list(context.get("recent_summaries") or [])
    recent.append(f"[{fmt_time(start)}] {summary}")
    context["recent_summaries"] = recent[-LAST_SUMMARY_COUNT:]
    return summary, context