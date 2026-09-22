#!/usr/bin/env python3
"""prosody_try.py — чем закрыть остаток разрыва по живости.

Ступень 2 даёт тембр, но предсказатель pitch сглаживает: σ по символам 0.52–0.65
против 0.61–0.98 в цели, размах F0 в звуке 44–49 Гц против 39–75 у F5. Причина
не в весах, а в лоссе: MSE по pitch оптимален на условном среднем, а среднее
всегда ровнее правды.

Скрипт меряет два вопроса до всякого дообучения:
  1. сколько даст простая растяжка предсказанного pitch вокруг среднего фразы
     (`p' = m + k·(p − m)`, пунктуация остаётся нулём) и растяжка длительностей;
  2. где потолок, если предсказатель pitch станет идеальным — вариант `gt_pitch`
     берёт истинный pitch цели при предсказанных длительностях.

Первое бесплатно и вкладывается в веса последнего слоя, второе показывает, стоит
ли вообще дообучать предсказатель.

  docker run --rm --network host -v ~/linkplay/tmp/silero/hub:/root/.cache/torch/hub \
    -v ~/linkplay/silero:/app -v ~/linkplay/tmp/silero:/w -w /app silero_ru \
    python -m probe.prosody_try --model /w/train/model_stage2.pt --out /w/prosody
"""
import argparse
import os

import numpy as np
import torch

from voice.asr import transcribe_pcm
from voice.audio import write16
from voice.model import SR, encode, load, load_state, predict, synth
from voice.paths import data
from voice.targets import load_stats, load_targets
from voice.text import cer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=data("train/model_stage2.pt"))
    ap.add_argument("--out", default=data("prosody"))
    ap.add_argument("--targets", default=data("targets"))
    ap.add_argument("--f5", default=data("f5"))
    ap.add_argument("--row", type=int, default=4)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--asr-host", default="127.0.0.1")
    ap.add_argument("--asr-port", type=int, default=10407)
    a = ap.parse_args()

    import librosa
    import torchaudio
    _, pk, tm = load()
    torch.set_grad_enabled(False)
    mu, sd, _ = load_stats(a.targets)
    tm = load_state(tm, a.model, a.row, "cpu")
    os.makedirs(a.out, exist_ok=True)
    val = load_targets(a.targets, "val")[:a.n]

    def stat(w):
        f0, v, _ = librosa.pyin(w, fmin=60, fmax=400, sr=SR,
                                frame_length=2048, hop_length=600)
        f = f0[v]
        return np.median(f), np.std(f), np.percentile(f, 95) - np.percentile(f, 5)

    def say(w, text):
        x16 = torchaudio.functional.resample(torch.as_tensor(w), SR, 16000).numpy()
        asr = transcribe_pcm((np.clip(x16, -1, 1) * 32767).astype("<i2").tobytes(),
                             a.asr_host, a.asr_port)
        return cer(text, asr)

    agg = {}
    for d in val:
        T = d["mel"].shape[1]
        seq, sp = encode(pk, d["text"], "xenia")
        mask, dur_p, pit_p = predict(tm, seq, sp)
        gt_pit = torch.nan_to_num((d["f0"] - mu) / sd, nan=0.0)[None, None]
        nz = pit_p.abs() > 1e-3                       # пунктуация остаётся нулём
        m = float(pit_p[nz].mean()) if nz.any() else 0.0
        s_p = float(pit_p[nz].std())
        s_t = float(gt_pit[0, 0][gt_pit[0, 0] != 0].std())
        print(f"\n{d['i']}: σ pitch предсказано {s_p:.2f}, цель {s_t:.2f} → "
              f"точное k {s_t / max(s_p, 1e-6):.2f}; длина предсказанная "
              f"{float(dur_p.sum()) / T:.2f}×", flush=True)

        def variant(tag, k, g, pit=None, centre=None):
            p = pit_p if pit is None else pit
            if pit is None and k != 1.0:
                c = m if centre is None else centre
                p = torch.where(nz, c + k * (pit_p - c), pit_p)
            dd = dur_p if g == 1.0 else torch.clamp(torch.round(dur_p * g), min=1)
            _, wav = synth(tm, seq, sp, mask, dd, p)
            w = wav.numpy()
            write16(f"{a.out}/{d['i']}_{tag}.wav", w)
            md, sg, rg = stat(w)
            c = say(w, d["text"])
            agg.setdefault(tag, []).append((rg, c))
            print(f"    {tag:16s} F0 {md:.0f} Гц, σ {sg:.1f}, размах {rg:.0f} Гц, "
                  f"CER {100 * c:.1f} %, длина {len(w) / SR:.2f} с", flush=True)

        variant("base", 1.0, 1.0)
        variant("p1.3", 1.3, 1.0)
        variant("p1.5", 1.5, 1.0)
        variant("p1.7", 1.7, 1.0)
        # вокруг нуля, а не среднего фразы: только такую растяжку можно вложить
        # в веса последнего слоя предсказателя и не менять код синтеза
        variant("p1.5z", 1.5, 1.0, centre=0.0)
        variant("p1.5_d1.08", 1.5, 1.08)
        variant("gt_pitch", 1.0, 1.0, pit=gt_pit)
        variant("gt_p1.5", 1.5, 1.0,
                pit=torch.where(nz, 1.5 * gt_pit, gt_pit))
        w, _ = librosa.load(os.path.join(a.f5, "wav", f"{d['i']}.wav"), sr=SR)
        md, sg, rg = stat(w)
        print(f"    {'F5 (эталон)':16s} F0 {md:.0f} Гц, σ {sg:.1f}, размах {rg:.0f} Гц")
        agg.setdefault("F5", []).append((rg, 0.0))

    print("\n== в среднем по фразам")
    for k, v in agg.items():
        r = np.mean([x[0] for x in v])
        c = np.mean([x[1] for x in v])
        print(f"   {k:16s} размах F0 {r:.0f} Гц, CER {100 * c:.2f} %")


if __name__ == "__main__":
    main()
