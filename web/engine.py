"""engine.py — движок генерации web-морды: много моделей одновременно.

Сервис 10500 грузит один файл модели на старте и для веба не подходит, поэтому
здесь свой движок: базовая модель Silero (стоковые голоса) плюс дообученные
модели из реестра, которые подгружаются по требованию и держатся в LRU-кэше.
Синтез CPU (RTF ≈ 0.05), путь текста тот же, что у сервиса: silero-stress,
числа словами, нарезка на фразы, потолок паузы на точке.

Ручки генерации:
  speech_rate  темп, 4-й аргумент dur_predictor (как speech_rate у apply_tts)
  pitch_pct    тон в процентах от F0: pitch_coef = mean_std_coef · pct/100 —
               так пересчитывает просодию SSML сам Silero (SILERO.md, «pitch»)
"""
import os
import threading
import time
from collections import OrderedDict

import numpy as np
import torch

from voice.audio import pcm_bytes, resample, wav_from_pcm
from voice.model import SR, cap_pauses, encode, load, predict, synth
from voice.text import outside, prepare, split_text

from web import store
from web.refprep import accent

STOCK = ["aidar", "baya", "kseniya", "eugene", "xenia"]
MSC_STOCK = {"aidar": 2.32, "baya": 6.13, "kseniya": 3.57, "eugene": 2.76,
             "xenia": 4.24}

DEFAULTS = {"speech_rate": 1.0, "pitch_pct": 0.0, "pause": 0.5, "gap": 0.25,
            "numbers": True, "accent": True, "sr": SR}

_base = None
_base_lock = threading.Lock()


def base():
    """базовая модель и обёртка, общие на процесс: движок и финализация задач"""
    global _base
    with _base_lock:
        if _base is None:
            _, pk, tm = load()
            torch.set_grad_enabled(False)
            _base = (pk, tm)
        return _base


def clamp_params(p):
    """параметры запроса -> ограниченный словарь с дефолтами модели"""
    out = dict(DEFAULTS)
    for k in out:
        if k in p and p[k] is not None:
            out[k] = p[k]
    out["speech_rate"] = min(2.0, max(0.5, float(out["speech_rate"])))
    out["pitch_pct"] = min(60.0, max(-60.0, float(out["pitch_pct"])))
    out["pause"] = min(1.5, max(0.0, float(out["pause"])))
    out["gap"] = min(2.0, max(0.0, float(out["gap"])))
    out["sr"] = int(out["sr"]) if int(out["sr"]) in (8000, 16000, 24000, 48000) else SR
    out["numbers"] = bool(out["numbers"])
    out["accent"] = bool(out["accent"])
    return out


class Engine:
    def __init__(self, workers=2, cache=3):
        self.SR = SR
        t0 = time.time()
        self.pk, self.tm = base()
        self.alphabet = set(self.pk.symbols[3:])
        t_model = time.time() - t0
        self.acc = accent()
        self.sem = threading.Semaphore(workers)
        self.lock = threading.Lock()
        self.cache = OrderedDict()                  # model_id -> [tm, mtime, msc]
        self.cache_max = cache
        t0 = time.time()

        def warm():
            for k in range(3):
                self.generate("Прогрев голоса, замок и зАмок, попытка "
                             f"{k + 1}.", "stock:xenia", {})
        w = threading.Thread(target=warm)
        w.start()
        w.join()
        print(f"движок готов: модель {t_model:.1f} с, разметчик "
              f"{time.time() - t0:.1f} с (прогрев учтён)", flush=True)

    # ------------------------------------------------------------- голоса
    def _msc(self, speaker):
        """mean_std_coef голоса: у обёртки или из замеров SILERO.md"""
        v = getattr(self.pk, "mean_std_coef", None)
        if isinstance(v, dict) and speaker in v:
            try:
                return float(v[speaker])
            except (TypeError, ValueError):
                pass
        return MSC_STOCK.get(speaker)

    def tm_for(self, voice):
        """'stock:<имя>' или id модели -> (модель, спикер, msc)"""
        if voice.startswith("stock:"):
            name = voice[6:]
            if name not in STOCK:
                raise ValueError(f"нет стокового голоса {name}")
            return self.tm, name, self._msc(name)
        meta = store.model_meta(voice)
        if not meta:
            raise ValueError("модель не найдена")
        path = os.path.join(store.model_dir(voice), meta["file"])
        if not os.path.exists(path):
            raise ValueError("файл модели не найден — обучение не закончено?")
        mtime = os.path.getmtime(path)
        with self.lock:
            hit = self.cache.get(voice)
            if hit and hit[1] == mtime:
                self.cache.move_to_end(voice)
                return hit[0], "xenia", hit[2]
        tm = torch.jit.load(path, map_location="cpu").eval()
        torch.set_grad_enabled(False)
        msc = meta.get("msc") or self._msc("xenia")
        with self.lock:
            self.cache[voice] = [tm, mtime, msc]
            self.cache.move_to_end(voice)
            while len(self.cache) > self.cache_max:
                self.cache.popitem(last=False)
        return tm, "xenia", msc

    # ------------------------------------------------------------- синтез
    def generate(self, text, voice, params):
        p = clamp_params(params)
        tm, speaker, msc = self.tm_for(voice)
        pitch_coef = None
        if abs(p["pitch_pct"]) > 1e-6:
            pitch_coef = float(msc) * p["pitch_pct"] / 100.0
        # кадры ≈ durs_rate^-2.2 (замер check_knobs), поэтому для честного
        # «скорость × s» в предсказатель уходит корень: s=2 даёт вполовину длиннее
        durs_rate = p["speech_rate"] ** 0.5
        t0 = time.time()
        parts, marks, lost = [], [], []
        for i, s in enumerate(split_text(text)):
            m = prepare(s, self.acc if p["accent"] else None, p["numbers"])
            marks.append(m)
            lost += outside(m, self.alphabet)
            with self.sem:
                with torch.no_grad():
                    seq, sp = encode(self.pk, m, speaker)
                    mask, dur, pitch = predict(tm, seq, sp,
                                               durs_rate=durs_rate,
                                               pitch_coef=pitch_coef)
                    dur = cap_pauses(tm, seq, dur, p["pause"])
                    _, wav = synth(tm, seq, sp, mask, dur, pitch)
            if i:
                parts.append(np.zeros(int(p["gap"] * self.SR), dtype=np.float32))
            parts.append(wav.numpy())
        if not parts:
            raise ValueError("после подготовки не осталось ни одной фразы")
        x = np.concatenate(parts)
        if p["sr"] != self.SR:
            x = resample(x, self.SR, p["sr"])
        return {"pcm": pcm_bytes(x), "sr": p["sr"], "sec": len(x) / p["sr"],
                "text": " ".join(marks), "lost": sorted(set(lost)),
                "params": p, "synth_sec": round(time.time() - t0, 3),
                "voice": voice, "speaker": speaker}

    def wav(self, r):
        return wav_from_pcm(r["pcm"], r["sr"])
