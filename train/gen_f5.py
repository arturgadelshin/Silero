#!/usr/bin/env python3
"""gen_f5.py — шаг 3/4: озвучить тексты шага 1 клоном своего голоса через F5-TTS.

Модель Misha24-10/F5-TTS_RUSSIAN, ударение «+» перед гласной — та же строка, что
пойдёт в токены Silero. Опора одна: ref/ref.wav + её текст с ударениями.

Грабли из LinkPlay_A98 (tools/pc/gen_clones.py, mww.md §10), зашиты сюда:
  1. fix_duration, а не speed: F5 сам недооценивает длину и глотает слова.
     fix_duration — СУММА опоры и генерации, поэтому ref_dur + gen_dur; gen_dur
     по слогам в темпе самой опоры, с разбросом темпа ±7 %.
  2. Проверка по длине РЕЧИ, а не файла (F5 оставляет тишину по краям).
  3. Брак не пишется в набор, а считается по видам. Сверх gen_clones — ASR:
     CER gigaam по нормализованному тексту ≤ --max-cer, иначе новая попытка с
     другим сидом (до --tries).

Выход {out}:
  wav/{i}.wav      родные 24 кГц, моно int16, края обрезаны с полем --margin
  f5.jsonl         принятые: i, split, text, seed, fix_duration, speech_s, cer, asr
  rejects.jsonl    все забракованные попытки с причиной
  rej/{i}_{k}.wav  звук забракованных попыток (для разбора на пробе)

Запуск (сервер, GPU, gigaam на 127.0.0.1:10407 — нужен --network host):
  docker run --rm --gpus all --network host -e HF_HUB_OFFLINE=1 \
    -v ~/linkplay/tmp/hf:/root/.cache/huggingface -v ~/linkplay/silero:/app \
    -v ~/linkplay/tmp/silero:/w -w /app silero_ru python -m train.gen_f5 --limit 50
"""
import argparse
import json
import os
import random
import re
import time
import wave

import numpy as np

from voice.asr import transcribe_pcm
from voice.audio import speech_span, write16
from voice.paths import data
from voice.text import cer, syllables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=data("ruslan/texts.tsv"))
    ap.add_argument("--ref", default=data("ref/ref.wav"))
    ap.add_argument("--ref-text", default=data("ref/ref_text.txt"),
                    help="текст опоры в той же разметке, что и тексты (с «+»)")
    ap.add_argument("--out", default=data("f5"))
    ap.add_argument("--limit", type=int, default=0, help="только первые N текстов (проба)")
    ap.add_argument("--part", default="0/1", help="i/n: каждый n-й текст, начиная с i")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--max-cer", type=float, default=0.05)
    ap.add_argument("--jitter", type=float, default=0.07, help="разброс темпа, доля")
    ap.add_argument("--sent-pause", type=float, default=0.30,
                    help="добавка к длине на каждую внутреннюю границу фразы (. ! ? …)")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--asr-host", default="127.0.0.1")
    ap.add_argument("--asr-port", type=int, default=10407)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--repo", default="Misha24-10/F5-TTS_RUSSIAN")
    ap.add_argument("--ckpt", default="F5TTS_v1_Base_v2/model_last_inference.safetensors")
    ap.add_argument("--vocab", default="F5TTS_v1_Base/vocab.txt")
    ap.add_argument("--model", default="F5TTS_v1_Base")
    a = ap.parse_args()

    import soundfile as sf
    import torch
    import torchaudio
    from huggingface_hub import hf_hub_download
    from f5_tts.api import F5TTS

    ref_text = open(a.ref_text, encoding="utf-8").read().strip()
    ref_wav, ref_sr = sf.read(a.ref, dtype="float32")
    ref_dur = len(ref_wav) / ref_sr
    # темп опоры, с её паузами: секунд на слог
    sec_per_syl = ref_dur / max(1, syllables(ref_text))

    rows = [ln.rstrip("\n").split("\t") for ln in open(a.texts, encoding="utf-8") if ln.strip()]
    if a.limit:
        rows = rows[:a.limit]
    pi, pn = (int(v) for v in a.part.split("/"))
    rows = rows[pi::pn]

    os.makedirs(os.path.join(a.out, "wav"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "rej"), exist_ok=True)
    man_path = os.path.join(a.out, "f5.jsonl")
    done = set()
    if os.path.exists(man_path):
        done = {json.loads(s)["i"] for s in open(man_path, encoding="utf-8")}
    rows = [r for r in rows if r[0] not in done]
    print(f"опора {ref_dur:.2f} с, {sec_per_syl:.3f} с/слог; текстов {len(rows)} "
          f"(уже готово {len(done)})", flush=True)

    f5 = F5TTS(model=a.model, ckpt_file=hf_hub_download(a.repo, a.ckpt),
               vocab_file=hf_hub_download(a.repo, a.vocab), device=a.device)
    man = open(man_path, "a", encoding="utf-8")
    rej = open(os.path.join(a.out, "rejects.jsonl"), "a", encoding="utf-8")
    rng = random.Random(f"{a.seed}:{a.part}")
    stat = {"ok": 0, "fail": 0, "err": 0, "silent": 0, "short": 0, "long": 0, "cer": 0}
    sp, cers, t_gen, t_audio = [], [], 0.0, 0.0
    t_start = time.time()
    for n, (i, split, _sec, text) in enumerate(rows):
        syl = max(1, syllables(text))
        # внутренние границы фраз: концевой знак не в счёт
        n_sent = len(re.findall(r"[.!?…]+\s", text))
        base = syl * sec_per_syl + n_sent * a.sent_pause
        lo, hi = 0.45 * base, 1.6 * base + 0.5
        ok = False
        for k in range(a.tries):
            seed = rng.randrange(1 << 30)
            gen_dur = base * rng.uniform(1 - a.jitter, 1 + a.jitter)
            t0 = time.time()
            try:
                wav, sr, _ = f5.infer(ref_file=a.ref, ref_text=ref_text, gen_text=text,
                                      fix_duration=ref_dur + gen_dur, seed=seed,
                                      remove_silence=False, show_info=lambda *x, **kw: None)
            except Exception as exc:                    # noqa: BLE001
                stat["err"] += 1
                print(f"{i}: {type(exc).__name__}: {exc}", flush=True)
                continue
            t_gen += time.time() - t0
            x = np.asarray(wav, dtype=np.float32).reshape(-1)
            t_audio += len(x) / sr
            rec = {"i": i, "try": k, "seed": seed, "fix_duration": round(ref_dur + gen_dur, 3),
                   "text": text}
            span = speech_span(x, sr)
            why = None
            if span is None:
                why = "silent"
            else:
                d = (span[1] - span[0]) / sr
                rec["speech_s"] = round(d, 3)
                if d < lo:
                    why = "short"
                elif d > hi:
                    why = "long"
            if why is None:
                m = int(a.margin * sr)
                y = x[max(0, span[0] - m):min(len(x), span[1] + m)]
                y16 = torchaudio.functional.resample(torch.from_numpy(y), sr, 16000).numpy()
                pcm = (np.clip(y16, -1, 1) * 32767).astype("<i2").tobytes()
                asr = transcribe_pcm(pcm, a.asr_host, a.asr_port)
                c = cer(text, asr)
                rec.update(asr=asr, cer=round(c, 4))
                if c > a.max_cer:
                    why = "cer"
            if why:
                stat[why] += 1
                rec["why"] = why
                rej.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rej.flush()
                write16(os.path.join(a.out, "rej", f"{i}_{k}.wav"), x, sr)
                continue
            write16(os.path.join(a.out, "wav", f"{i}.wav"), y, sr)
            rec.update(split=split, sr=sr, dur_s=round(len(y) / sr, 3))
            man.write(json.dumps(rec, ensure_ascii=False) + "\n")
            man.flush()
            sp.append(rec["speech_s"])
            cers.append(rec["cer"])
            ok = True
            break
        stat["ok" if ok else "fail"] += 1
        if (n + 1) % 10 == 0 or n + 1 == len(rows):
            el = time.time() - t_start
            print(f"{n + 1}/{len(rows)} {stat} | RTF {t_gen / max(t_audio, 1e-9):.3f}, "
                  f"{el / (n + 1):.1f} с/текст", flush=True)
    man.close()
    rej.close()
    s = np.array(sp) if sp else np.zeros(1)
    c = np.array(cers) if cers else np.zeros(1)
    print(f"готово: принято {stat['ok']}, не вышло за {a.tries} попытки {stat['fail']}; "
          f"брак попыток — тихих {stat['silent']}, коротких {stat['short']}, длинных "
          f"{stat['long']}, CER>{a.max_cer} {stat['cer']}, ошибок {stat['err']}", flush=True)
    print(f"речь {s.sum() / 3600:.3f} ч, {s.mean():.2f}±{s.std():.2f} с; CER принятых "
          f"среднее {100 * c.mean():.2f} %, макс {100 * c.max():.2f} %", flush=True)


if __name__ == "__main__":
    main()
