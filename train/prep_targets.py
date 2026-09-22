#!/usr/bin/env python3
"""prep_targets.py — шаг 5: из клонов F5 сделать цели для обучения Silero.

На фразу сохраняется всё, что нужно учительскому пути FastPitch:
  seq      токены Silero (та же строка с «+», что озвучивал F5)
  mel      цель декодера: log10(clamp(slaney_mel(|STFT|², 2400/2400/600, 192,
           0–24000 Гц, norm None), 1e-5)), центр кадра +300 — формула из
           вскрытия 2 (SILERO.md, «mel: формула найдена»), пол −5 = sil_value
  dur      кадров на символ: DTW по MFCC против стокового синтеза той же фразы
           с durs_rate, подогнанным под длину записи; минимум 1 кадр, сумма = T
  f0       средний F0 (Гц) по звонким кадрам символа, NaN — глухо/пунктуация

Нормировка pitch (F0 − μ)/σ считается вторым проходом по всему набору и лежит
в stats.pt: μ и σ принадлежат голосу, а не модели (SILERO.md, «pitch»).

F5 отдаёт 24 кГц, поэтому выше 12 кГц у мела пусто. Полосы выше границы
исключаются из лосса маской; граница берётся из данных (см. отчёт) и лежит в
stats.pt как mask_band. Верх при обучении остаётся от стокового декодера.

Запуск (сервер, CPU, части параллельно):
  docker run --rm -e SILERO_THREADS=6 -v ~/linkplay/tmp/silero/hub:/root/.cache/torch/hub \
    -v ~/linkplay/silero:/app -v ~/linkplay/tmp/silero:/w -w /app \
    silero_ru python -m train.prep_targets --part 0/6
  затем один раз: python -m train.prep_targets --stats
"""
import argparse
import json
import os
import warnings

import numpy as np
import torch

from voice.audio import dtw_durs, f0_track, feats, symbol_f0
from voice.model import HOP, SR, encode, load, predict, synth
from voice.paths import data

N_FFT = 2400
MELS = 192
CENTER = 300                      # смещение центра кадра, сэмплы


def mel_target(x, T, fb):
    """мел Silero из звука 48 кГц: log10 мощностного slaney-мела"""
    P = N_FFT
    xp = torch.nn.functional.pad(x[None, None], (P, P), mode="reflect")[0, 0]
    starts = torch.arange(T) * HOP + CENTER + P - N_FFT // 2
    idx = (starts[:, None] + torch.arange(N_FFT)[None]).clamp(0, len(xp) - 1)
    S = torch.fft.rfft(xp[idx] * torch.hann_window(N_FFT), dim=-1).abs().T ** 2
    return torch.log10(torch.clamp(fb.T @ S, min=1e-5))


def fix_durs(d, T):
    """минимум 1 кадр на символ и ровно T кадров в сумме"""
    d = np.maximum(np.asarray(d, dtype=np.int64), 1)
    while d.sum() > T:                       # отнимаем у самых длинных
        i = int(np.argmax(d))
        if d[i] <= 1:
            break
        d[i] -= 1
    if d.sum() < T:                          # добавляем последнему символу
        d[-1] += T - d.sum()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f5", default=data("f5"))
    ap.add_argument("--out", default=data("targets"))
    ap.add_argument("--voice", default="eugene", help="стоковый голос-опора для DTW")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stats", action="store_true", help="второй проход: μ, σ, маска, отчёт")
    ap.add_argument("--mask-hz", type=float, default=12000.0,
                    help="полосы выше этой частоты — вне лосса (F5 даёт 24 кГц)")
    a = ap.parse_args()

    warnings.filterwarnings("ignore")
    os.makedirs(a.out, exist_ok=True)

    if a.stats:
        return stats(a)

    import soundfile as sf
    import torchaudio

    _, pk, tm = load()
    torch.set_grad_enabled(False)
    fb = torchaudio.functional.melscale_fbanks(N_FFT // 2 + 1, 0.0, float(SR // 2), MELS,
                                               SR, None, "slaney")
    rows = [json.loads(s) for s in open(os.path.join(a.f5, "f5.jsonl"), encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    pi, pn = (int(v) for v in a.part.split("/"))
    rows = rows[pi::pn]
    done = 0
    bad = {"короткая": 0, "нет звонких": 0, "ошибка": 0}
    for n, r in enumerate(rows):
        path = os.path.join(a.out, f"{r['i']}.pt")
        if os.path.exists(path):
            done += 1
            continue
        y, sr = sf.read(os.path.join(a.f5, "wav", f"{r['i']}.wav"), dtype="float32")
        # широкий фильтр: при обычном (width 6) над 12 кГц остаётся мусор пересчёта
        x = torchaudio.functional.resample(torch.from_numpy(y), sr, SR,
                                           lowpass_filter_width=64, rolloff=0.99)
        T = len(x) // HOP
        if T < 20:
            bad["короткая"] += 1
            continue
        try:
            seq, sp = encode(pk, r["text"], a.voice)
            mask, d_pred, pitch = predict(tm, seq, sp)
            # опора той же длины, что запись: иначе DTW тянет путь и теряет символы
            rate = T / max(float(d_pred.sum()), 1.0)
            d_ref = torch.clamp(torch.round(d_pred * rate), min=1)
            mel_r, wav_r = synth(tm, seq, sp, mask, d_ref, pitch)
            # громкость записи — к уровню стокового синтеза
            g = float(torch.sqrt((wav_r ** 2).mean()) / (torch.sqrt((x ** 2).mean()) + 1e-9))
            x = x * g
            mel = mel_target(x, T, fb)
            d_est = fix_durs(dtw_durs(feats(wav_r.numpy(), mel_r.shape[-1]),
                                      d_ref[0].numpy(), feats(x.numpy(), T)), T)
            f0 = symbol_f0(f0_track(x.numpy(), T), d_est)
        except Exception as exc:                        # noqa: BLE001
            bad["ошибка"] += 1
            print(f"{r['i']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        for i, c in enumerate(seq[0].tolist()):         # пунктуация без pitch
            if c in tm.punkt_symbols:
                f0[i] = np.nan
        if not np.isfinite(f0).any():
            bad["нет звонких"] += 1
            continue
        torch.save({"i": r["i"], "split": r["split"], "text": r["text"],
                    "seq": seq[0].to(torch.int64), "mel": mel.to(torch.float16),
                    "dur": torch.from_numpy(d_est), "f0": torch.from_numpy(f0).float(),
                    "gain_db": round(20 * np.log10(g + 1e-9), 2)}, path)
        done += 1
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(rows)} готово {done} брак {bad}", flush=True)
    print(f"часть {a.part}: готово {done} из {len(rows)}, брак {bad}", flush=True)


def stats(a):
    """μ, σ по F0, граница пустых полос мела, проверки и отчёт"""
    files = sorted(f for f in os.listdir(a.out) if f.endswith(".pt") and f != "stats.pt")
    f0s, mels, bad_sum, zero_dur, splits = [], [], 0, 0, {}
    n_sym = n_voiced = 0
    for k, f in enumerate(files):
        d = torch.load(os.path.join(a.out, f))
        T = d["mel"].shape[1]
        if int(d["dur"].sum()) != T:
            bad_sum += 1
        zero_dur += int((d["dur"] < 1).sum())
        v = d["f0"][torch.isfinite(d["f0"])]
        f0s.append(v)
        n_sym += len(d["f0"])
        n_voiced += len(v)
        splits[d["split"]] = splits.get(d["split"], 0) + 1
        if k % 7 == 0:                                   # выборка для статистики полос
            mels.append(d["mel"][:, ::11].float())
    f0 = torch.cat(f0s)
    mu, sd = float(f0.mean()), float(f0.std())
    M = torch.cat(mels, 1)
    p95 = M.quantile(0.95, dim=1)
    import torchaudio
    fb = torchaudio.functional.melscale_fbanks(N_FFT // 2 + 1, 0.0, float(SR // 2), MELS,
                                               SR, None, "slaney")
    freqs = torch.arange(N_FFT // 2 + 1) * SR / N_FFT
    centers = (fb * freqs[:, None]).sum(0) / fb.sum(0).clamp(min=1e-9)
    # граница маски — по частоте, а не по данным: выше 12 кГц у звука 24 кГц
    # настоящего сигнала нет, а мусор пересчёта в мел всё равно попадает
    band = int((centers >= a.mask_hz).nonzero().flatten()[0])
    torch.save({"f0_mean": mu, "f0_std": sd, "mask_band": band,
                "mask_hz": float(centers[band]), "files": len(files)},
               os.path.join(a.out, "stats.pt"))
    print(f"== фраз {len(files)} {splits}, сумма dur ≠ кадрам мела: {bad_sum}, "
          f"нулевых длительностей: {zero_dur}")
    print(f"   F0 по символам: μ {mu:.1f} Гц, σ {sd:.1f} Гц, μ/σ {mu / sd:.2f} "
          f"(mean_std_coef eugene 2.76); звонких символов {100 * n_voiced / n_sym:.1f} %")
    quiet = (p95 < -4.0).nonzero().flatten()
    q = int(quiet.min()) if len(quiet) else MELS
    print(f"   маска лосса с полосы {band} из {MELS} (центр {float(centers[band]):.0f} Гц), "
          f"под маской {MELS - band} полос")
    print(f"   по данным мел садится на пол с полосы {q} "
          f"({float(centers[min(q, MELS - 1)]):.0f} Гц)")
    print("   мел по полосам (95-й процентиль): " + " ".join(
        f"{i}:{float(p95[i]):.1f}" for i in range(0, MELS, 16)))


if __name__ == "__main__":
    main()
