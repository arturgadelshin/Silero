#!/usr/bin/env python3
"""diag_voice.py — откуда берётся роботность и сколько осталось до потолка.

На каждой отложенной фразе синтезируются три версии одного текста:
  a_ceiling  цель шага 5 (мел из клона F5), верх выше маски — от стокового
             декодера. Это потолок: лучше вокодер Silero не сыграет;
  b_teacher  учительский путь — декодер с ИСТИННЫМИ длительностями и pitch.
             Разница с a показывает, что не дотягивает сам декодер;
  c_infer    обычный синтез: длительности и pitch от предсказателей. Разница
             с b — вклад предсказателей, это и есть «роботность».

Печатаются мел на учительском пути, ошибка длительностей, σ и корреляция pitch
с целью, а также F0 звука рядом с эталоном F5.

  docker run --rm -v ~/linkplay/tmp/silero/hub:/root/.cache/torch/hub \
    -v ~/linkplay/silero:/app -v ~/linkplay/tmp/silero:/w -w /app silero_ru \
    python -m probe.diag_voice --model /w/train/model_stage2.pt --out /w/diag2
"""
import argparse
import os

import numpy as np
import torch

from voice.audio import write16
from voice.model import SR, encode, load, load_state, predict, synth
from voice.paths import data
from voice.targets import load_stats, load_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=data("train/voice_stage1.pt"),
                    help="файл голоса [2,128] или целая модель ступени 2+")
    ap.add_argument("--out", default=data("diag"))
    ap.add_argument("--targets", default=data("targets"))
    ap.add_argument("--f5", default=data("f5"))
    ap.add_argument("--row", type=int, default=4)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()

    import librosa
    _, pk, tm = load()
    torch.set_grad_enabled(False)
    mu, sd, B = load_stats(a.targets)
    tm = load_state(tm, a.model, a.row, "cpu")
    os.makedirs(a.out, exist_ok=True)
    val = load_targets(a.targets, "val")[:a.n]

    def f0stat(w):
        f0, v, _ = librosa.pyin(w, fmin=60, fmax=400, sr=SR,
                                frame_length=2048, hop_length=600)
        f = f0[v]
        return np.median(f), np.std(f), np.percentile(f, 95) - np.percentile(f, 5)

    for d in val:
        T = d["mel"].shape[1]
        seq, sp = encode(pk, d["text"], "xenia")
        mask, dur_p, pit_p = predict(tm, seq, sp)
        gt_dur = d["dur"][None].float()
        gt_pit = torch.nan_to_num((d["f0"] - mu) / sd, nan=0.0)[None, None]
        mel_s, _ = synth(tm, seq, sp, mask, gt_dur, pit_p)
        tgt = d["mel"].float()
        ceil = tgt.clone()
        ceil[B:] = mel_s[0][B:, :T]                  # верх выше маски — от стока
        mel_b = tm.tacotron(seq, sp, mask, gt_dur, gt_pit, None)[0]
        _, wav_c = synth(tm, seq, sp, mask, dur_p, pit_p)
        for n, m in (("a_ceiling", ceil), ("b_teacher", mel_b[:, :T])):
            w = tm.vocoder(m[None], SR, 0.0, True).reshape(-1).numpy()
            write16(f"{a.out}/{d['i']}_{n}.wav", w)
        write16(f"{a.out}/{d['i']}_c_infer.wav", wav_c.numpy())

        l_b = float((mel_b[:B, :T] - tgt[:B]).abs().mean())
        dd = np.abs(dur_p[0].numpy() - d["dur"].numpy()).mean()
        pp, gp = pit_p[0, 0].numpy(), gt_pit[0, 0].numpy()
        print(f"{d['i']}: мел учительский |d| {l_b:.3f} | dur ошибка {dd:.2f} кадра, "
              f"длина {float(dur_p.sum()) / T:.2f}× | pitch σ пред {pp.std():.2f} "
              f"цель {gp.std():.2f} r {np.corrcoef(pp, gp)[0, 1]:.2f}", flush=True)
        for n in ("a_ceiling", "b_teacher", "c_infer"):
            w, _ = librosa.load(f"{a.out}/{d['i']}_{n}.wav", sr=SR)
            m_, s_, r_ = f0stat(w)
            print(f"    {n}: F0 {m_:.0f} Гц, σ {s_:.1f}, размах 5-95% {r_:.0f} Гц")
        w, _ = librosa.load(os.path.join(a.f5, "wav", f"{d['i']}.wav"), sr=SR)
        m_, s_, r_ = f0stat(w)
        print(f"    F5 (эталон): F0 {m_:.0f} Гц, σ {s_:.1f}, размах {r_:.0f} Гц")


if __name__ == "__main__":
    main()
