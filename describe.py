import base64
import os
import re
from math import floor
from models import (
    chat,
    count_tokens,
    cut_to_tokens,
    count_prompt_tokens,
    count_image_tokens,
)
from slicing import slice, load_subtitles
from plot import parse_plot, split_plot, fmt_entry, fmt_time
from config import (
    N_CTX, MAX_REPLY, SAFETY_MARGIN, TOKENS_PER_WORD,
    PLOT_CONTEXT_TEMPLATE, PLOT_COMPRESS_PROMPT, SCENE_SUMMARIZER_TEMPLATE,
    PLOT_PAST_RECENT_TOKENS, PLOT_PAST_SUMMARY_WORDS, PLOT_FOLD_CHUNK_TOKENS,
    PLOT_NOW_TOKENS, PLOT_FUTURE_TOKENS, PLOT_FUTURE_MAX_ENTRIES,
    SUBS_BEFORE_SEC, SUBS_AFTER_SEC, SUBS_TOKENS,
)


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
def compress_plot(llm, summary, new_text):
    system = PLOT_COMPRESS_PROMPT.format(max_words=PLOT_PAST_SUMMARY_WORDS)
    user = (
        f"Уже сжатая предыстория:\n{summary or 'нет'}\n\n"
        f"Новые события сюжета:\n{new_text}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    result = chat(llm, messages, max_tokens=MAX_REPLY, temperature=0.3)

    # модель превысила лимит слов - одна повторная попытка сжатия
    if count_words(result) > PLOT_PAST_SUMMARY_WORDS * 1.3:
        result = chat(
            llm,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Уже сжатая предыстория:\nнет\n\nНовые события сюжета:\n{result}"},
            ],
            max_tokens=MAX_REPLY,
            temperature=0.3,
        )

    return result


def plot_past_context(llm, past, context=None):
    """Недавнее прошлое - дословно, всё что старше - в сжатой предыстории (инкрементально)."""
    lines = [fmt_entry(e) for e in past]
    recent = take_fit(llm, lines, PLOT_PAST_RECENT_TOKENS, from_end=True) if lines else []
    older_count = len(lines) - len(recent)

    summary = context["past_summary"] if context else ""
    folded = context["folded"] if context else 0
    if folded > older_count:
        # контекст ушёл дальше, чем нужно для этой сцены - пересобираем
        summary, folded = "", 0

    while folded < older_count:
        batch, used = [], 0
        while folded + len(batch) < older_count:
            line = lines[folded + len(batch)]
            t = count_tokens(llm, line) + 1
            if batch and used + t > PLOT_FOLD_CHUNK_TOKENS:
                break
            batch.append(line)
            used += t
        summary = compress_plot(llm, summary, "\n".join(batch))
        folded += len(batch)

    return summary, recent, folded


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
        if e <= start and s >= start - SUBS_BEFORE_SEC:
            before.append(line)
        elif s >= end and s <= end + SUBS_AFTER_SEC:
            after.append(line)
        elif e > start and s < end:
            during.append(line)

    before = take_fit(llm, before, SUBS_TOKENS // 2, from_end=True)
    during = take_fit(llm, during, SUBS_TOKENS // 4)
    after = take_fit(llm, after, SUBS_TOKENS // 2)
    return before, during, after


def build_scene_context(llm, entries, subtitles_path, start, end, context=None):
    past, now, future = split_plot(entries, start, end)
    summary, recent, folded = plot_past_context(llm, past, context)

    parts = [f"Сцена в кадрах: {fmt_time(start)} - {fmt_time(end)}"]

    if summary:
        parts.append(f"Сжатая предыстория (что было раньше):\n{summary}")
    if recent:
        parts.append("Недавние события сюжета:\n" + "\n".join(recent))

    now_lines = take_fit(llm, [fmt_entry(e) for e in now], PLOT_NOW_TOKENS) if now else []
    if now_lines:
        parts.append(
            "Сюжет на момент сцены (эпизод шире, чем сцена в кадрах; "
            "точное время события внутри эпизода неизвестно):\n" + "\n".join(now_lines))
    else:
        parts.append("Для этого момента описания в сюжете нет.")

    future_lines = take_fit(
        llm, [fmt_entry(e) for e in future[:PLOT_FUTURE_MAX_ENTRIES]], PLOT_FUTURE_TOKENS
    ) if future else []
    if future_lines:
        parts.append(
            "Ближайшее будущее (только подсказка, не описывать как происходящее "
            "в кадрах):\n" + "\n".join(future_lines))

    before, during, after = subtitles_window(llm, load_subtitles(subtitles_path), start, end)
    if before:
        parts.append("Субтитры до сцены (говорящий не указан):\n" + "\n".join(before))
    if during:
        parts.append("Субтитры внутри окна кадров (говорящий не указан):\n" + "\n".join(during))
    if after:
        parts.append("Субтитры после сцены (говорящий не указан):\n" + "\n".join(after))

    return "\n\n".join(parts), {"past_summary": summary, "folded": folded}


# ------------------------------------------- комментарии по кадрам
def frame_message(n, start, shift):
    label = f"Кадр {n}, {fmt_time(start + (n - 1) * shift)}"
    return {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_to_data_url(f"frames/frame{n}.jpg")}},
        {"type": "text", "text": label}
    ]}


def image_comment(llm, ctx_text, start, shift):
    messages = [
        {"role": "system", "content": f"{PLOT_CONTEXT_TEMPLATE}\n\n{format_ctx(ctx_text)}"},
    ]
    total_frames = len(os.listdir("frames"))

    img = image_to_data_url("frames/frame1.jpg")
    tokens_per_image = count_image_tokens(llm=llm, data_url=img)
    tokens_per_image += count_tokens(llm, "Кадр 000, 0:00:00")
    system_tokens = count_prompt_tokens(llm, messages)

    available = N_CTX - system_tokens - MAX_REPLY - SAFETY_MARGIN

    image_per_batch = floor(available / tokens_per_image)
    if image_per_batch < 1:
        return "Слишком маленькое контекстное окно!"

    # все кадры влезают за один проход
    if total_frames <= image_per_batch:
        for n in range(1, total_frames + 1):
            messages.append(frame_message(n, start, shift))

        return chat(llm, messages, max_tokens=MAX_REPLY, temperature=0.3)

    # нужно несколько проходов: резервируем место под описание предыдущего батча ([ctx])
    image_per_batch = floor((available - MAX_REPLY) / tokens_per_image)
    if image_per_batch < 1:
        return "Слишком маленькое контекстное окно!"

    scene_log = []
    base = list(messages)

    for first in range(1, total_frames + 1, image_per_batch):
        last = min(first + image_per_batch - 1, total_frames)

        messages = list(base)
        if scene_log:
            messages.append(
                {"role": "user", "content": f"[ctx]{scene_log[-1]}[ctx]"}
            )
        for n in range(first, last + 1):
            messages.append(frame_message(n, start, shift))

        temp_summary = chat(llm, messages, max_tokens=MAX_REPLY, temperature=0.3)
        scene_log.append(temp_summary)

    return scene_log


def summarizer(llm, scene_log, max_words, ctx_text):
    if isinstance(scene_log, str):
        scene_log = [scene_log]
    scene_text = "\n".join(scene_log)

    prompt = SCENE_SUMMARIZER_TEMPLATE.format(max_words=max_words, scenes_text=scene_text, ctx=ctx_text)
    messages = [{"role": "user", "content": prompt}]

    return chat(llm, messages, max_tokens=MAX_REPLY, temperature=0.3)


# ------------------------------------------- полный проход по одной паузе
def describe_scene(llm, plot_path, subtitles_path, video_path, period,
                   pause_index, shift, max_words, context=None):
    entries = parse_plot(plot_path)
    start, end = slice(pause_index, shift, subtitles_path, period, video_path)

    ctx_text, context = build_scene_context(
        llm, entries, subtitles_path, start, end, context)

    scene_log = image_comment(llm, ctx_text, start, shift)
    summary = summarizer(llm, scene_log, max_words, ctx_text)

    return summary, context