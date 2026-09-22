#!/usr/bin/env python3
"""asr_scan.py — прогнать автосбор ложных будилок через ASR и найти клипы,
в прероллe которых звучит целевая фраза.

Зачем. Метка «ложная будилка» ставится по правилу «после хита тишина»
(vadsend: пустой финал серверного VAD). Правило смотрит ВПЕРЁД, а в файл
пишется преролл — то, что было ДО хита. Если «милый дом» прозвучал в разговоре
и продолжения не было, клип ляжет в негативы поверх настоящего позитива и
станет учить сеть не просыпаться на собственную фразу. Дыра узкая, но
проверяется дёшево: сотня клипов — семь минут аудио.

Говорит по wyoming (JSONL + бинарный payload по TCP) с контейнером
wyoming-onnxasr (gigaam-v3-rnnt, network=host, поэтому 127.0.0.1:<порт>).
Запускать НА сервере: ssh green@192.168.0.121 'python3 /home/green/linkplay/pc/asr_scan.py ...'
"""
import argparse
import glob
import json
import os
import re
import sys

from voice.asr import transcribe


# «милый дом» целиком, а также порознь: ASR на 4-секундном прероллe часто
# слышит только одно слово, и это уже повод посмотреть клип глазами.
RE_BOTH = re.compile(r"мил\w*\s+дом", re.I)
RE_MIL = re.compile(r"\bмил(ый|ая|ые|ого|ом)?\b", re.I)
RE_DOM = re.compile(r"\bдом(а|у|е|ом|ик\w*)?\b", re.I)


def verdict(text):
    if RE_BOTH.search(text):
        return "ФРАЗА"          # карантин: почти наверняка настоящий позитив
    if RE_MIL.search(text) and RE_DOM.search(text):
        return "оба слова"      # порознь в одной фразе — посмотреть
    if RE_MIL.search(text) or RE_DOM.search(text):
        return "одно слово"     # штатный трудный негатив, оставляем
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/green/linkplay/data/realneg")
    ap.add_argument("--glob", default="fw*.wav")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10407)
    ap.add_argument("--out", default="")
    ap.add_argument("--quarantine", default="",
                    help="куда перенести клипы с вердиктом ФРАЗА (по умолчанию не переносить)")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.dir, a.glob)))
    if not paths:
        sys.exit(f"нет файлов по {a.dir}/{a.glob}")
    rows, bad = [], 0
    for i, p in enumerate(paths, 1):
        try:
            txt = transcribe(p, a.host, a.port)
        except Exception as e:                    # noqa: BLE001
            print(f"{os.path.basename(p)}: ОШИБКА {e}", flush=True)
            bad += 1
            continue
        v = verdict(txt)
        rows.append({"file": os.path.basename(p), "text": txt, "verdict": v})
        if v or txt:
            print(f"[{i:3d}/{len(paths)}] {os.path.basename(p)}: "
                  f"{txt or '(тишина)'}{'   <<< ' + v if v else ''}", flush=True)
    n_phrase = sum(1 for r in rows if r["verdict"] == "ФРАЗА")
    n_both = sum(1 for r in rows if r["verdict"] == "оба слова")
    n_empty = sum(1 for r in rows if not r["text"])
    print(f"\nвсего {len(rows)}, пусто {n_empty}, одно слово "
          f"{sum(1 for r in rows if r['verdict'] == 'одно слово')}, "
          f"оба слова {n_both}, ФРАЗА {n_phrase}, ошибок {bad}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"-> {a.out}")
    if a.quarantine and n_phrase:
        os.makedirs(a.quarantine, exist_ok=True)
        for r in rows:
            if r["verdict"] == "ФРАЗА":
                os.rename(os.path.join(a.dir, r["file"]),
                          os.path.join(a.quarantine, r["file"]))
        print(f"перенесено в карантин: {n_phrase}")


if __name__ == "__main__":
    main()
