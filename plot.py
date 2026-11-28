import re
from dataclasses import dataclass

TS = r"(\d{1,2}(?::\d{2}){1,2})"
LINE_RE = re.compile(rf"^\s*\[\s*{TS}\s*[-–—]\s*{TS}\s*\]\s*(.*)$")


@dataclass
class PlotEntry:
    start: float
    end: float
    text: str


def ts_to_seconds(ts: str) -> float:
    total = 0
    for part in ts.split(":"):
        total = total * 60 + int(part)
    return float(total)


def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}"


def fmt_entry(e: PlotEntry) -> str:
    return f"[{fmt_time(e.start)} - {fmt_time(e.end)}] {e.text}"


def _norm_text(text: str) -> str:
    return " ".join(re.sub(r"\W+", " ", text.lower()).split())


def _is_duplicate_entry(prev: PlotEntry, curr: PlotEntry) -> bool:
    a, b = _norm_text(prev.text), _norm_text(curr.text)
    if not a or not b:
        return not b
    if a == b:
        return True
    same_span = abs(prev.start - curr.start) <= 1.0 and abs(prev.end - curr.end) <= 1.0
    if same_span and (a.startswith(b) or b.startswith(a)):
        return True
    return False


def parse_plot(path: str) -> list[PlotEntry]:
    raw_entries: list[PlotEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE_RE.match(line)
            if m:
                raw_entries.append(PlotEntry(ts_to_seconds(m.group(1)),
                                             ts_to_seconds(m.group(2)),
                                             m.group(3).strip()))
            elif raw_entries:
                raw_entries[-1].text += " " + line

    raw_entries.sort(key=lambda e: (e.start, e.end))

    # Отбрасываем повторы и оборванные дубли при зацикливании модели
    entries: list[PlotEntry] = []
    for curr in raw_entries:
        if not curr.text.strip():
            continue
        if entries and _is_duplicate_entry(entries[-1], curr):
            if len(curr.text) > len(entries[-1].text):
                entries[-1] = curr
            continue
        entries.append(curr)
    return entries


def locate_plot(entries: list[PlotEntry], start: float, end: float):
    """Точно привязывает паузу [start, end] к эпизодам сюжета.

    Таймкоды в _plot.md округлены до секунд и идут по границам реплик, а паузы находятся
    между репликами. Поэтому учитываем как паузы ВНУТРИ эпизода, так и паузы НА СТЫКЕ
    двух эпизодов, до первой реплики фильма и после последней реплики.

    Возвращает: (past, now, future, mode)
    где mode in {"empty", "inside", "between", "before_first", "after_last"}.
    """
    if not entries:
        return [], [], [], "empty"

    # Порог 1.5 сек отсекает ложное касание из-за округления HH:MM:SS до целых секунд
    min_overlap = min(1.5, max(0.2, (end - start) * 0.5))
    inside_indices = [
        idx for idx, e in enumerate(entries)
        if min(end, e.end) - max(start, e.start) > min_overlap
    ]

    if inside_indices:
        first_i, last_i = inside_indices[0], inside_indices[-1]
        past = entries[:first_i]
        now = entries[first_i : last_i + 1]
        future = entries[last_i + 1 :]
        return past, now, future, "inside"

    # Пауза лежит между эпизодами (или в самом начале / финале фильма)
    mid = (start + end) / 2.0
    prev_idx = None
    next_idx = None
    for idx, e in enumerate(entries):
        if e.end <= start + 1.5 or e.end <= mid:
            prev_idx = idx
        elif next_idx is None:
            next_idx = idx

    if prev_idx is None and next_idx is not None:
        # До первой реплики фильма
        return [], [entries[next_idx]], entries[next_idx + 1 :], "before_first"

    if prev_idx is not None and next_idx is None:
        # После последней реплики фильма
        return entries[:prev_idx], [entries[prev_idx]], [], "after_last"

    if prev_idx is not None and next_idx is not None:
        # Между двумя соседними сценами / эпизодами
        past = entries[:prev_idx]
        now = [entries[prev_idx], entries[next_idx]]
        future = entries[next_idx + 1 :]
        return past, now, future, "between"

    return [], [entries[0]], entries[1:], "before_first"


def split_plot(entries, start, end):
    past, now, future, _ = locate_plot(entries, start, end)
    return past, now, future