#!/usr/bin/env python3
"""fetch_ruslan_text.py — только текст корпуса RUSLAN, без аудио.

Корпус (Довлатов, 22 200 фраз, CC BY-NC-SA 4.0) лежит на HF в parquet вместе
со звуком (~10 ГБ). Звук не нужен: берём строки через datasets-server API
(поле audio там — ссылка, сам файл не качается).

  akuzdeuov/ruslan-tts   — text, все 22 200 фраз, порядок корпуса
  Gzaborey/ruslan-dataset — transcript + normalized_text с «+» (21 874 фразы),
                            для сверки ударений RUAccent

Выход: {out}/ruslan.tsv (i, text), {out}/ruslan_gz.tsv (i, transcript, normalized_text).
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co/rows"


def rows(dataset, offset, length, tries=6):
    q = urllib.parse.urlencode({"dataset": dataset, "config": "default",
                                "split": "train", "offset": offset, "length": length})
    for k in range(tries):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                d = json.load(r)
            return d["rows"], d.get("num_rows_total")
        except Exception as exc:                       # noqa: BLE001
            print(f"  {dataset} @{offset}: {type(exc).__name__}: {exc}; повтор", flush=True)
            time.sleep(5 * (k + 1))
    raise SystemExit(f"{dataset} @{offset}: не получилось за {tries} попыток")


def fetch(dataset, cols, path):
    out, total, off = [], None, 0
    while total is None or off < total:
        rs, total = rows(dataset, off, 100)
        if not rs:
            break
        for r in rs:
            out.append([str(r["row_idx"])] + [
                r["row"][c].replace("\t", " ").replace("\n", " ") for c in cols])
        off += len(rs)
        if off % 2000 < 100:
            print(f"  {dataset}: {off}/{total}", flush=True)
        time.sleep(0.2)
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write("\t".join(r) + "\n")
    print(f"{dataset}: {len(out)} строк -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/w/ruslan")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fetch("akuzdeuov/ruslan-tts", ["text"], os.path.join(a.out, "ruslan.tsv"))
    fetch("Gzaborey/ruslan-dataset", ["transcript", "normalized_text"],
          os.path.join(a.out, "ruslan_gz.tsv"))


if __name__ == "__main__":
    main()
