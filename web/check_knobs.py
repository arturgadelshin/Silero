#!/usr/bin/env python3
"""check_knobs.py — сверка ручек генерации web-морды с apply_tts.

Семантика durs_rate/pitch_coef восстановлена по вскрытию (SILERO.md), но
живую обёртку мы не видим отсюда — этот скрипт прогоняет оба пути на
настоящей модели и печатает вердикты:

  1. что умеет apply_tts (подпись: есть ли speech_rate / pitch-просодия);
  2. наш predict(durs_rate=...) против apply_tts(speech_rate=...): близки ли
     длительности (в service.tts движок и apply_tts сходятся бит в бит при
     rate 1.0, тут то же самое с коэффициентом);
  3. pitch_coef: аддитивный ли сдвиг нормированного pitch (тон в процентах
     пересчитывается как mean_std_coef * pct/100).

Запуск на сервере:
  docker exec silero_web python -m web.check_knobs
Если вердикты не «ок», править web/engine.py (generate/clamp_params), а не
этот скрипт.
"""
import inspect

import numpy as np
import torch

from voice.model import SR, encode, load, predict, synth

TEXT = "Таймер на пять минут поставлен, замок и зАмок."


def ours(tm, pk, text, durs_rate=1.0, pitch_coef=None):
    seq, sp = encode(pk, text, "xenia")
    mask, dur, pitch = predict(tm, seq, sp, durs_rate=durs_rate, pitch_coef=pitch_coef)
    _, wav = synth(tm, seq, sp, mask, dur, pitch)
    return wav.numpy(), float(dur.sum()), pitch


def f0_med(x):
    import librosa
    f0, v, _ = librosa.pyin(x, fmin=60, fmax=400, sr=SR, frame_length=2048,
                            hop_length=600)
    return float(np.nanmedian(f0[v]))


def main():
    model, pk, tm = load()
    torch.set_grad_enabled(False)
    print("подпись apply_tts:", inspect.signature(model.apply_tts))
    kw_ok = {}
    for k in ("speech_rate", "pitch_rate"):
        try:
            model.apply_tts(text=TEXT, speaker="xenia", sample_rate=SR, **{k: 1.1})
            kw_ok[k] = True
        except (TypeError, ValueError):
            kw_ok[k] = False
    print(f"apply_tts принимает: {kw_ok}")

    base_w, base_frames, base_p = ours(tm, pk, TEXT)
    ref = model.apply_tts(text=TEXT, speaker="xenia", sample_rate=SR)
    print(f"\nбаза: наш путь {len(base_w) / SR:.2f} с, apply_tts "
          f"{len(ref) / SR:.2f} с, кадров {base_frames:.0f}")

    print("\n-- темп (durs_rate)")
    for rate in (0.8, 1.25):
        w, frames, _ = ours(tm, pk, TEXT, durs_rate=rate)
        ratio = frames / base_frames
        print(f"  durs_rate {rate}: {len(w) / SR:.2f} с, кадров "
              f"{frames:.0f} = {ratio:.2f}x")
        if kw_ok.get("speech_rate"):
            r2 = model.apply_tts(text=TEXT, speaker="xenia", sample_rate=SR,
                                 speech_rate=rate)
            same_dir = (len(r2) < len(ref)) == (len(w) < len(base_w))
            verdict = "ок: направление то же" if same_dir \
                else "РАСХОЖДЕНИЕ направления!"
            print(f"    apply_tts(speech_rate={rate}): {len(r2) / SR:.2f} с — {verdict}")

    print("\n-- тон (pitch_coef, сдвиг в единицах нормированного pitch)")
    f0_0 = f0_med(base_w)
    for msc in (4.24, ):                       # xenia из замеров SILERO.md
        for pct in (10, -10):
            coef = msc * pct / 100
            w, _, _ = ours(tm, pk, TEXT, pitch_coef=coef)
            f1 = f0_med(w)
            print(f"  pitch_pct {pct:+.0f}% (coef {coef:+.2f}): F0 {f0_0:.0f} -> "
                  f"{f1:.0f} Гц, ожидание ~{f0_0 * (1 + pct / 100):.0f} Гц")
    print("\nсмотри вердикты выше; при расхождении правь web/engine.py")


if __name__ == "__main__":
    main()
