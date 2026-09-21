#!/usr/bin/env python3
"""say.py — озвучить готовый текст обученной моделью.

Строка файла = одна реплика, пустая строка = удлинённая пауза. Текст проходит
ту же подготовку, что корпус шага 1 (`pre_ruaccent`/`post_ruaccent` +
RUAccent), поэтому «…», «–» и «ё» доживают до токенов, а кавычки и скобки не
слипаются со словами. Символы вне алфавита Silero — ошибка, а не молчаливая
потеря: цифры и латиницу надо раскрывать словами в исходном тексте.

  docker run --rm --network host -v ~/linkplay/tmp/silero/hub:/root/.cache/torch/hub \
    -v ~/linkplay/tmp/silero/ruaccent:/ruaccent -v ~/linkplay/tmp/silero:/w \
    silero_ru python /w/say.py --text /w/say/announce.txt --out /w/say/announce.wav
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import torch

# известные ошибки RUAccent, пойманные на шаге 1 (SILERO.md, сверка с RUSLAN):
# «самое» он превращает в редкое «самоё», в «джинсы» и «баловался» бьёт мимо
FIXES = {"само+ё": "с+амое", "джинс+ы": "дж+инсы", "б+аловался": "балов+ался"}


def fix_known(t):
    for bad, good in FIXES.items():
        t = re.sub(bad.replace("+", r"\+"), good, t, flags=re.IGNORECASE)
        t = re.sub(bad.replace("+", r"\+").capitalize(), good.capitalize(), t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/w/train/model_voice.pt")
    ap.add_argument("--speaker", default="xenia", help="строка обученного голоса")
    ap.add_argument("--row", type=int, default=4)
    ap.add_argument("--gap", type=float, default=0.35, help="пауза между строками, с")
    ap.add_argument("--par-gap", type=float, default=0.8, help="пауза на пустой строке, с")
    ap.add_argument("--ruaccent-dir", default="/ruaccent")
    ap.add_argument("--accent-model", default="turbo3.1")
    ap.add_argument("--no-accent", action="store_true",
                    help="текст уже с «+»: не грузить RUAccent вовсе")
    ap.add_argument("--dump-text", default="", help="куда сложить размеченный текст")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--asr-host", default="127.0.0.1")
    ap.add_argument("--asr-port", type=int, default=10407)
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from silero_probe2 import load, encode, predict, synth, write16, cer, SR
    from train_voice import load_state

    t0 = time.time()
    _, pk, tm = load()
    torch.set_grad_enabled(False)
    tm = load_state(tm, a.model, a.row, "cpu")
    alphabet = set(pk.symbols[3:])
    t_model = time.time() - t0

    acc = None
    if not a.no_accent:
        t0 = time.time()
        from prep_text import pre_ruaccent, post_ruaccent, patch_token_type_ids
        from ruaccent import RUAccent
        acc = RUAccent()
        try:
            acc.load(omograph_model_size=a.accent_model, use_dictionary=True,
                     workdir=a.ruaccent_dir)
        except TypeError:
            acc.load(omograph_model_size=a.accent_model, use_dictionary=True)
        patch_token_type_ids(acc.accent_model.session)
        t_acc = time.time() - t0
    else:
        t_acc = 0.0

    lines = [s.rstrip("\n") for s in open(a.text, encoding="utf-8")]
    bad = {ch for s in lines for ch in s if re.match(r"[0-9a-zA-Z]", ch)}
    if bad:
        raise SystemExit(f"цифры и латиница в тексте: {sorted(bad)} — раскройте словами")

    parts, total_cer, n, marked = [], 0.0, 0, []
    t_synth = t_speech = 0.0
    for s in lines:
        if not s.strip():
            parts.append(np.zeros(int(a.par_gap * SR), dtype=np.float32))
            marked.append("")
            continue
        t = s.strip() if a.no_accent else \
            fix_known(post_ruaccent(acc.process_all(pre_ruaccent(s))))
        marked.append(t)
        t1 = time.time()
        seq, sp = encode(pk, t, a.speaker)
        # Silero сам приводит текст к строчным, поэтому сверяем в нижнем регистре
        lost = {ch for ch in t.lower().replace("+", "") if ch not in alphabet and ch != " "}
        mask, dur, pitch = predict(tm, seq, sp)
        _, wav = synth(tm, seq, sp, mask, dur, pitch)
        w = wav.numpy()
        dt = time.time() - t1
        t_synth += dt
        t_speech += len(w) / SR
        parts.append(w)
        parts.append(np.zeros(int(a.gap * SR), dtype=np.float32))
        note = f"  ВНЕ АЛФАВИТА: {sorted(lost)}" if lost else ""
        print(f"{len(w) / SR:5.2f} с  за {dt:4.2f} с  {t}{note}", flush=True)
        if not a.no_asr:
            import torchaudio
            from asr_scan import transcribe_pcm
            x16 = torchaudio.functional.resample(torch.as_tensor(w), SR, 16000).numpy()
            asr = transcribe_pcm((np.clip(x16, -1, 1) * 32767).astype("<i2").tobytes(),
                                 a.asr_host, a.asr_port)
            c = cer(t, asr)
            total_cer += c
            n += 1
            if c > 0.02:
                print(f"         CER {100 * c:.1f} %: {asr}", flush=True)

    y = np.concatenate(parts)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write16(a.out, y)
    if a.dump_text:
        open(a.dump_text, "w", encoding="utf-8").write("\n".join(marked) + "\n")
    print(f"\n{a.out}: {len(y) / SR:.1f} с, пик {20 * np.log10(np.abs(y).max() + 1e-9):+.1f} дБ"
          + (f", CER по строкам {100 * total_cer / max(n, 1):.2f} %" if n else ""))
    print(f"время: модель {t_model:.1f} с, RUAccent {t_acc:.1f} с, синтез {t_synth:.1f} с "
          f"на {t_speech:.1f} с речи (RTF {t_synth / max(t_speech, 1e-9):.3f}), "
          f"потоков {torch.get_num_threads()}")


if __name__ == "__main__":
    main()
