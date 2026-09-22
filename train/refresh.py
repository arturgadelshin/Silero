#!/usr/bin/env python3
"""refresh.py — выбросить то, что устарело после переразметки текста.

gen_f5 и prep_targets пропускают уже готовое: первый по номерам в f5.jsonl,
второй по наличию targets/<i>.pt. Поэтому, когда текст фразы изменился (сменили
разметчик ударений, поправили обвязку), достаточно удалить её звук и цели — и
обычный прогон конвейера пересчитает ровно их, а не весь корпус.

Сравниваются два texts.tsv: текущий и предыдущий (его надо было сохранить до
переразметки). Фразы, которых в старом файле нет, тоже считаются изменёнными.

  docker run --rm -v ~/linkplay/silero:/app -v ~/linkplay/tmp/silero:/w \
    -w /app silero_ru python -m train.refresh \
      --old /w/ruslan/texts_ruaccent.tsv --run

По умолчанию только показывает, что будет удалено.
"""
import argparse
import json
import os

from voice.paths import data


def read(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = line.rstrip("\n").split("\t")
        if len(r) == 4:
            out[r[0]] = r[3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=data("ruslan/texts.tsv"))
    ap.add_argument("--old", required=True, help="texts.tsv до переразметки")
    ap.add_argument("--f5", default=data("f5"))
    ap.add_argument("--targets", default=data("targets"))
    ap.add_argument("--run", action="store_true", help="удалять, а не показывать")
    a = ap.parse_args()

    new, old = read(a.texts), read(a.old)
    stale = sorted(i for i, t in new.items() if old.get(i) != t)
    print(f"фраз в корпусе {len(new)}, изменилось {len(stale)}")
    for i in stale[:5]:
        print(f"  {i}\n    было:  {old.get(i, '(не было)')[:90]}\n"
              f"    стало: {new[i][:90]}")

    man = os.path.join(a.f5, "f5.jsonl")
    keep = [s for s in open(man, encoding="utf-8")
            if json.loads(s)["i"] not in set(stale)] if os.path.exists(man) else []
    wavs = [os.path.join(a.f5, "wav", f"{i}.wav") for i in stale]
    pts = [os.path.join(a.targets, f"{i}.pt") for i in stale]
    n_wav = sum(os.path.exists(p) for p in wavs)
    n_pt = sum(os.path.exists(p) for p in pts)
    n_man = (sum(1 for _ in open(man, encoding="utf-8")) - len(keep)) if os.path.exists(man) else 0
    print(f"под нож: звук {n_wav}, записей в f5.jsonl {n_man}, целей {n_pt}")

    if not a.run:
        print("\nэто список; чтобы удалить, добавьте --run")
        return
    for p in wavs + pts:
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists(man):
        with open(man, "w", encoding="utf-8") as f:
            f.writelines(keep)
    print("удалено; дальше обычный конвейер: f5 -> targets -> stats -> обучение")


if __name__ == "__main__":
    main()
