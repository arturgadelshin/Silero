#!/usr/bin/env python3
"""train.py — шаг 6: обучить Silero v5_5_ru своему голосу на целях шага 5.

Учительский путь FastPitch: в декодер подаются ИСТИННЫЕ длительности и pitch из
`targets/`, поэтому мел сравнивается кадр в кадр. Предсказатели длительности и
pitch учатся отдельно на тех же целях.

Ступени (`--stage`):
  1. только голос — три строки таблиц (мел 128 + dur 64 + pitch 64) в позиции
     `--row` (по умолчанию 4, место xenia). Экспорт — файл `[2,128]`, который
     Silero грузит как `speaker='random', voice_path=...` (см. load_random_voice).
  2. плюс декодер tacotron, `lin` и тела предсказателей. Экспорт — весь модуль
     через `torch.jit.save`. Декодер общий на все голоса, поэтому стоковые голоса
     надо переслушать.
  3. плюс текстовый кодировщик tacotron (`encoder`, `embedding` и проекции pitch
     и energy) — последнее, что остаётся стоковым. Весов вдвое больше, фраз те же
     1 095, поэтому lr меньше и ранняя остановка обязательна.

Лосс: L1 по мелу с маской (полосы выше 12 кГц вне лосса — у F5 там пусто),
MSE по `ln(1+dur)` и MSE по нормированному pitch. Всё, кроме учимого, заморожено;
вокодер в обучении не участвует.

Запуск (GPU; занимает один из трёх слотов RTX 3070):
  docker run --rm --gpus all -e SILERO_THREADS=4 \
    -v ~/linkplay/tmp/silero/hub:/root/.cache/torch/hub \
    -v ~/linkplay/silero:/app -v ~/linkplay/tmp/silero:/w -w /app \
    silero_ru python -m train.train --stage 1
  проверка после обучения (нужен gigaam, поэтому --network host):
  ... silero_ru python -m train.train --check /w/train/voice_stage1.pt
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from voice.asr import transcribe_pcm
from voice.audio import write16
from voice.model import EMB, SR, encode, load, load_state, predict, synth, voice_tensor
from voice.paths import data
from voice.targets import load_stats, load_targets
from voice.text import cer


def batches(items, size, shuffle, rng=None):
    """фразы схожей длины в один батч — меньше добивки"""
    items = sorted(items, key=lambda r: r["mel"].shape[1])
    bs = [items[i:i + size] for i in range(0, len(items), size)]
    if shuffle:
        rng.shuffle(bs)
    return bs


def collate(batch, row, mu, sd, dev):
    B = len(batch)
    L = max(len(r["seq"]) for r in batch)
    T = max(r["mel"].shape[1] for r in batch)
    seq = torch.zeros(B, L, dtype=torch.int64)
    smask = torch.ones(B, L, dtype=torch.bool)          # True = добивка
    dur = torch.zeros(B, L)
    pit = torch.zeros(B, L)
    mel = torch.full((B, 192, T), -5.0)
    fmask = torch.zeros(B, T, dtype=torch.bool)         # True = настоящий кадр
    for j, r in enumerate(batch):
        n, t = len(r["seq"]), r["mel"].shape[1]
        seq[j, :n] = r["seq"]
        smask[j, :n] = False
        dur[j, :n] = r["dur"].float()
        z = (r["f0"] - mu) / sd
        pit[j, :n] = torch.nan_to_num(z, nan=0.0)
        mel[j, :, :t] = r["mel"].float()
        fmask[j, :t] = True
    sp = torch.full((B,), row, dtype=torch.int64)
    return [x.to(dev) for x in (seq, smask, dur, pit, mel, fmask, sp)]


def losses(tm, b, band):
    seq, smask, dur, pit, mel, fmask, sp = b
    pred = tm.tacotron(seq, sp, smask, dur, pit[:, None, :], None)
    T = min(pred.shape[2], mel.shape[2])
    m = fmask[:, None, :T] & (torch.arange(192, device=mel.device)[None, :, None] < band)
    l_mel = ((pred[:, :, :T] - mel[:, :, :T]).abs() * m).sum() / m.sum().clamp(min=1)
    keep = ~smask
    ld = tm.dur_predictor(seq, sp, smask, 1.0, None)
    l_dur = (((ld - torch.log1p(dur)) ** 2) * keep).sum() / keep.sum().clamp(min=1)
    pp = tm.pitch_predictor(seq, sp, smask, None, None)[:, 0]
    l_pit = (((pp - pit) ** 2) * keep).sum() / keep.sum().clamp(min=1)
    return l_mel, l_dur, l_pit


def trainable(tm, stage, row, train_pred=True):
    """что учим: строки голоса всегда; со ступени 2 — декодер, lin и (по умолчанию)
    тела предсказателей. Без них голос выходит роботом: стоковый pitch-предсказатель
    занижает разброс (σ 0.5 против 0.9 в цели), а dur торопит речь на 10-15 %.
    Со ступени 3 — ещё текстовый кодировщик: он общий на все голоса и задаёт,
    как символ звучит в контексте, поэтому после него стоковые голоса меняются
    сильнее всего."""
    P = dict(tm.named_parameters())
    for p in P.values():
        p.requires_grad_(False)
    rows = ["tacotron.speaker_embedding.weight",
            "dur_predictor.dur_pred.speaker_embedding.weight",
            "pitch_predictor.pitch_pred.speaker_embedding.weight"]
    for k in rows:
        P[k].requires_grad_(True)
    extra = []
    if stage >= 2:
        pref = ["tacotron.decoder.", "tacotron.lin."]
        if train_pred:
            pref += ["dur_predictor.", "pitch_predictor."]
        if stage >= 3:
            pref += ["tacotron.encoder.", "tacotron.embedding.",
                     "tacotron.pitch_proj.", "tacotron.energy_proj."]
        for k, p in P.items():
            if any(k.startswith(x) for x in pref) and k not in rows:
                p.requires_grad_(True)
                extra.append(k)
    return [P[k] for k in rows], [P[k] for k in extra], rows, extra


def evaluate(tm, items, a, band, mu, sd, dev):
    tot = np.zeros(3)
    n = 0
    with torch.no_grad():
        for b in batches(items, a.batch, False):
            l = losses(tm, collate(b, a.row, mu, sd, dev), band)
            tot += np.array([float(x) for x in l]) * len(b)
            n += len(b)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=data("targets"))
    ap.add_argument("--out", default=data("train"))
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--row", type=int, default=4, help="строка голоса (4 = место xenia)")
    ap.add_argument("--init-voice", default="eugene", help="с какого голоса начать (F0 ближе всех)")
    ap.add_argument("--resume", default="", help="файл голоса [2,128] или чекпойнт ступени")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--w-dur", type=float, default=1.0)
    ap.add_argument("--w-pitch", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--no-train-pred", action="store_true",
                    help="ступень 2 без тел предсказателей (только декодер и lin)")
    ap.add_argument("--check", default="", help="не учить, а проверить этот голос/чекпойнт")
    ap.add_argument("--check-n", type=int, default=10)
    ap.add_argument("--asr-host", default="127.0.0.1")
    ap.add_argument("--asr-port", type=int, default=10407)
    a = ap.parse_args()
    if not a.lr:
        a.lr = {1: 1e-2, 2: 1e-4}.get(a.stage, 5e-5)

    _, pk, tm = load()
    dev = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"
    tm.to(dev)
    mu, sd, band = load_stats(a.targets)
    os.makedirs(a.out, exist_ok=True)

    if a.check:
        return check(a, pk, tm, dev, mu, sd)

    # старт: копия ближайшего стокового голоса в учимую строку
    src = pk.speakers.index(a.init_voice)
    P = dict(tm.named_parameters())
    with torch.no_grad():
        for k in ("tacotron.speaker_embedding.weight",
                  "dur_predictor.dur_pred.speaker_embedding.weight",
                  "pitch_predictor.pitch_pred.speaker_embedding.weight"):
            P[k][a.row] = P[k][src].clone()
    if a.resume:
        tm = load_state(tm, a.resume, a.row, dev)
        print(f"продолжаем с {a.resume}", flush=True)

    torch.set_grad_enabled(True)        # torch.hub отдаёт модель с выключенным grad
    rows, extra, rnames, enames = trainable(tm, a.stage, a.row, not a.no_train_pred)
    opt = torch.optim.AdamW(rows + extra, lr=a.lr, weight_decay=0.0)
    train = load_targets(a.targets, "train")
    val = load_targets(a.targets, "val")
    print(f"ступень {a.stage}: учим {len(rows)} строк голоса и {len(extra)} тензоров "
          f"({sum(p.numel() for p in extra):,} весов), lr {a.lr}", flush=True)
    print(f"фраз: обучение {len(train)}, проверка {len(val)}; маска мела с полосы {band}; "
          f"F0 μ {mu:.1f} σ {sd:.1f}; устройство {dev}", flush=True)

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    v0 = evaluate(tm, val, a, band, mu, sd, dev)
    print(f"до обучения ({a.init_voice}): проверка мел {v0[0]:.4f} dur {v0[1]:.4f} "
          f"pitch {v0[2]:.4f}", flush=True)
    best, bad, t0 = 1e9, 0, time.time()
    hist = []
    for ep in range(1, a.epochs + 1):
        # dropout включаем только там, где учим сами веса (tacotron 0.1, предсказатели 0.5)
        tm.tacotron.train(a.stage >= 2)
        tm.dur_predictor.train(a.stage >= 2 and not a.no_train_pred)
        tm.pitch_predictor.train(a.stage >= 2 and not a.no_train_pred)
        run, n = np.zeros(3), 0
        for b in batches(train, a.batch, True, rng):
            bb = collate(b, a.row, mu, sd, dev)
            l_mel, l_dur, l_pit = losses(tm, bb, band)
            loss = l_mel + a.w_dur * l_dur + a.w_pitch * l_pit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():           # учим ОДНУ строку, остальные голоса не трогаем
                for p in rows:
                    g = p.grad
                    if g is not None:
                        keep = g[a.row].clone()
                        g.zero_()
                        g[a.row] = keep
            torch.nn.utils.clip_grad_norm_(rows + extra, 5.0)
            opt.step()
            run += np.array([float(l_mel), float(l_dur), float(l_pit)]) * len(b)
            n += len(b)
        tm.tacotron.eval()
        tm.dur_predictor.eval()
        tm.pitch_predictor.eval()
        v = evaluate(tm, val, a, band, mu, sd, dev)
        tr = run / n
        hist.append({"epoch": ep, "train": tr.tolist(), "val": v.tolist()})
        print(f"эпоха {ep:3d}: обучение мел {tr[0]:.4f} dur {tr[1]:.4f} pitch {tr[2]:.4f} | "
              f"проверка мел {v[0]:.4f} dur {v[1]:.4f} pitch {v[2]:.4f} | "
              f"{time.time() - t0:.0f} с", flush=True)
        score = v[0] + a.w_dur * v[1] + a.w_pitch * v[2]
        if score < best - 1e-4:
            best, bad = score, 0
            torch.save(voice_tensor(tm, a.row), os.path.join(a.out, f"voice_stage{a.stage}.pt"))
            if a.stage >= 2:
                torch.jit.save(tm, os.path.join(a.out, f"model_stage{a.stage}.pt"))
            print(f"   лучший результат, сохранено", flush=True)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"ранняя остановка: {a.patience} эпох без улучшения", flush=True)
                break
    json.dump(hist, open(os.path.join(a.out, f"hist_stage{a.stage}.json"), "w"), indent=1)
    print(f"готово, лучший суммарный лосс {best:.4f}; голос в "
          f"{a.out}/voice_stage{a.stage}.pt", flush=True)


def check(a, pk, tm, dev, mu, sd):
    """синтез отложенными фразами: CER gigaam, F0, длина; стоковые голоса целы"""
    import librosa
    import torchaudio
    tm = load_state(tm, a.check, a.row, dev)
    torch.set_grad_enabled(False)
    out = os.path.join(a.out, "check_" + os.path.splitext(os.path.basename(a.check))[0])
    os.makedirs(out, exist_ok=True)
    val = load_targets(a.targets, "val")[:a.check_n]
    res = []
    for d in val:
        seq, sp = encode(pk, d["text"], "xenia")           # строка a.row = место xenia
        seq, sp = seq.to(dev), sp.to(dev)                  # обёртка отдаёт CPU-тензоры
        mask, dur, pitch = predict(tm, seq, sp)
        _, wav = synth(tm, seq, sp, mask, dur, pitch)
        w = wav.cpu().numpy()
        write16(os.path.join(out, f"{d['i']}.wav"), w)
        x16 = torchaudio.functional.resample(torch.as_tensor(w), SR, 16000).numpy()
        asr = transcribe_pcm((np.clip(x16, -1, 1) * 32767).astype("<i2").tobytes(),
                             a.asr_host, a.asr_port)
        f0, v, _ = librosa.pyin(w, fmin=60, fmax=400, sr=SR, frame_length=2048, hop_length=600)
        res.append((cer(d["text"], asr), float(np.nanmedian(f0[v])),
                    len(w) / SR, d["i"], asr))
    c = np.array([x[0] for x in res])
    f = np.array([x[1] for x in res])
    print(f"== свой голос: CER {100 * c.mean():.2f} % (макс {100 * c.max():.2f}), "
          f"F0 медиана {np.median(f):.0f} Гц [{f.min():.0f}–{f.max():.0f}]; "
          f"записи дали μ {mu:.0f} Гц")
    for x in sorted(res, reverse=True)[:3]:
        print(f"   CER {100 * x[0]:.1f}% {x[3]}: {x[4][:80]}")
    for v_ in ("aidar", "eugene"):                          # стоковые не сломаны?
        d = val[0]
        seq, sp = encode(pk, d["text"], v_)
        seq, sp = seq.to(dev), sp.to(dev)
        mask, dur, pitch = predict(tm, seq, sp)
        _, wav = synth(tm, seq, sp, mask, dur, pitch)
        w = wav.cpu().numpy()
        write16(os.path.join(out, f"stock_{v_}.wav"), w)
        x16 = torchaudio.functional.resample(torch.as_tensor(w), SR, 16000).numpy()
        asr = transcribe_pcm((np.clip(x16, -1, 1) * 32767).astype("<i2").tobytes(),
                             a.asr_host, a.asr_port)
        print(f"   {v_}: CER {100 * cer(d['text'], asr):.1f} %")


if __name__ == "__main__":
    main()
