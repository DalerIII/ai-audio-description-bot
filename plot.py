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


def parse_plot(path: str) -> list:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE_RE.match(line)
            if m:
                entries.append(PlotEntry(ts_to_seconds(m.group(1)),
                                         ts_to_seconds(m.group(2)),
                                         m.group(3).strip()))
            elif entries:
                entries[-1].text += " " + line
    entries.sort(key=lambda e: e.start)
    return entries


def split_plot(entries, start, end):
    past = [e for e in entries if e.end <= start]
    future = [e for e in entries if e.start >= end]
    now = [e for e in entries if e.start < end and e.end > start]
    return past, now, future