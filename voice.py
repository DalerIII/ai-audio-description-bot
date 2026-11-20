import asyncio
import os
import edge_tts
from edge_tts.exceptions import NoAudioReceived
from mutagen.mp3 import MP3

VOICES = {
    "ru": "ru-RU-SvetlanaNeural",
    "en": "en-US-AriaNeural",
    "de": "de-DE-KatjaNeural",
    "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "pl": "pl-PL-ZofiaNeural",
    "uk": "uk-UA-PolinaNeural",
    "tr": "tr-TR-EmelNeural",
    "nl": "nl-NL-ColetteNeural",
}


async def _synth(text, voice, rate, out, retries=5):
    tmp = out + ".tmp"
    for attempt in range(1, retries + 1):
        try:
            await edge_tts.Communicate(text, voice, rate=rate).save(tmp)
            os.replace(tmp, out)
            return
        except NoAudioReceived:
            print(f"[retry {attempt}/{retries}] rate={rate} len={len(text)} out={out}")
            if attempt == retries:
                raise
            await asyncio.sleep(1.5 * attempt)


async def speak(text, target_sec, out, lang="ru", tolerance=0.05, max_iter=3):
    text = text.strip()
    if not text:
        raise ValueError("Пустой текст")
    if target_sec <= 0:
        raise ValueError(f"Некорректный target_sec: {target_sec}")

    voice = VOICES[lang]
    goal = target_sec * (1 - tolerance)  # целимся чуть короче цели
    rate_pct = 0
    dur = 0

    for i in range(max_iter):
        await _synth(text, voice, f"{rate_pct:+d}%", out)
        dur = MP3(out).info.length

        if dur <= target_sec or i == max_iter - 1:
            break

        current = 1 + rate_pct / 100
        new_pct = round((current * dur / goal - 1) * 100)
        new_pct = max(-50, min(100, new_pct))
        if new_pct == rate_pct:  # упёрлись в предел, дальше смысла нет
            break
        rate_pct = new_pct

    if dur > target_sec:
        print(f"[warn] {out}: {dur:.2f}s > {target_sec:.2f}s даже при {rate_pct:+d}%")
    return dur