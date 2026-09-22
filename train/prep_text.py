#!/usr/bin/env python3
"""prep_text.py — шаг 1: тексты RUSLAN с ударениями для F5 и Silero.

Одна строка на фразу, одинаковая для обеих моделей: F5_RUSSIAN и Silero
понимают «+» ПЕРЕД ударной гласной. Ударения ставятся в два слоя:

  1. RUAccent (Den4ikAI/ruaccent) — ударения и «ё» во всех словах, где уверен;
  2. акцентор Silero поверх — слова с «+» он не трогает (models/accentor.py:
     «If word already have stress, we won't process it again»), а дописывает
     то, что RUAccent оставил без ударения: односложные слова
     (stress_single_vowel) и пропуски. Так строка совпадает с тем, что Silero
     видел бы у себя в apply_tts, и F5 читает ту же строку.

Итоговая строка проходит через prepare_text_input Silero без изменений
(иначе символ вне алфавита Silero выпал бы из токенов, а в звуке F5 остался).

Запуск (сервер, CPU):
  docker run --rm -v ~/linkplay/tmp/silero/hub:/root/.cache/torch/hub \
    -v ~/linkplay/tmp/silero/ruaccent:/ruaccent -v ~/linkplay/silero:/app \
    -v ~/linkplay/tmp/silero:/w -w /app silero_ru python -m train.prep_text

Выход ({out}):
  texts.tsv      i, split(train/val), sec_est, текст для F5/Silero
  stress_cmp.txt сверка с авторскими «+» из Gzaborey/ruslan-dataset
  отчёт в stdout: фразы, часы по оценке, покрытие 47 символов.
"""
import argparse
import collections
import os
import random
import re

from voice.paths import data
from voice.text import Accent, stress_map, syllables

# темп RUSLAN по оценке — слогов в секунду; уточняется по F5 на шаге 3
SYL_PER_SEC = 4.5


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=data("ruslan"))
    ap.add_argument("--out", default=data("ruslan"))
    ap.add_argument("--ruaccent-dir", default="/ruaccent")
    ap.add_argument("--model", default="turbo3.1")
    ap.add_argument("--hours", type=float, default=2.0, help="сколько часов речи отобрать")
    ap.add_argument("--min-sec", type=float, default=2.0)
    ap.add_argument("--max-sec", type=float, default=12.0)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="только первые N фраз (проба)")
    ap.add_argument("--seed", type=int, default=20260918)
    a = ap.parse_args()

    import torch

    torch.set_num_threads(int(os.environ.get("SILERO_THREADS", "4")))
    model, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                              language="ru", speaker="v5_5_ru")
    pk = model.packages[0]
    symbols = pk.symbols
    alphabet = set(symbols[3:])

    acc = Accent(a.model, a.ruaccent_dir)

    rows = read_tsv(os.path.join(a.src, "ruslan.tsv"))
    if a.limit:
        rows = rows[:a.limit]
    gz = {}
    gz_path = os.path.join(a.src, "ruslan_gz.tsv")
    if os.path.exists(gz_path):
        for r in read_tsv(gz_path):
            if len(r) == 3:
                gz[r[1].strip()] = r[2]

    kept, drop, removed = [], collections.Counter(), collections.Counter()
    cmp_same = cmp_diff = 0
    diffs = []
    for n, (idx, text) in enumerate(rows):
        if n % 2000 == 0:
            print(f"  {n}/{len(rows)}", flush=True)
        text = text.strip()
        if re.search(r"[0-9a-zA-Z/*<>]", text):
            drop["цифры/латиница"] += 1
            continue
        sec = syllables(text) / SYL_PER_SEC
        if not (a.min_sec <= sec <= a.max_sec):
            drop["длина"] += 1
            continue
        t1 = acc(text)
        t2 = pk.accentor(t1, put_stress=True, put_stress_homo=True, put_yo=True,
                         put_yo_homo=True, stress_single_vowel=True)
        # F5 читает ровно то, что Silero превратит в токены: всё, что Silero
        # выкидывает из строки («», кавычки, скобки), выкидывается и для F5
        for ch in t2.lower():
            if ch not in alphabet:
                removed[ch] += 1
        sent, _, has = pk.prepare_text_input(t2, None)
        sent = re.sub(r"\s+", " ", sent).strip()
        if not has:
            drop["пусто"] += 1
            continue
        t2 = sent
        if text in gz:
            ref = stress_map(gz[text])
            mine = stress_map(t2)
            for w, v in ref.items():
                if w in mine:
                    if mine[w] == v:
                        cmp_same += 1
                    else:
                        cmp_diff += 1
                        if len(diffs) < 200:
                            diffs.append(f"{w}: RUSLAN {v}, наш {mine[w]} | {t2}")
        kept.append((idx, sec, t2))

    rng = random.Random(a.seed)
    rng.shuffle(kept)
    # жадный отбор: сначала фразы с редкими символами, пока не покрыты все
    cover = collections.Counter()
    need = alphabet - set(" ")
    chosen, rest = [], []
    for k in kept:
        s = set(k[2].lower())
        if (s & need) - set(cover):
            chosen.append(k)
            cover.update(s)
        else:
            rest.append(k)
    budget = a.hours * 3600
    total = sum(k[1] for k in chosen)
    for k in rest:
        if total >= budget:
            break
        chosen.append(k)
        cover.update(set(k[2].lower()))
        total += k[1]
    every = max(2, round(1 / a.val))       # каждая every-я фраза — в проверку
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "texts.tsv"), "w", encoding="utf-8") as f:
        for j, (idx, sec, t) in enumerate(chosen):
            f.write(f"{idx}\t{'val' if j % every == every - 1 else 'train'}\t{sec:.2f}\t{t}\n")
    with open(os.path.join(a.out, "stress_cmp.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(diffs) + "\n")

    miss = sorted(need - set(cover))
    n_q = sum(1 for k in chosen if "?" in k[2])
    n_e = sum(1 for k in chosen if "!" in k[2])
    print(f"== RUSLAN фраз {len(rows)}, годных {len(kept)}, отобрано {len(chosen)} "
          f"(≈{total / 3600:.2f} ч по оценке {SYL_PER_SEC} слог/с)")
    print(f"   вопросов {n_q}, восклицаний {n_e}, в проверку {len(chosen) // every}")
    print(f"   отброшено фраз: {dict(drop)}")
    print(f"   выкинуто символов вне алфавита Silero: {dict(removed.most_common(15))}")
    print(f"   символов Silero {len(need)}, не покрыто: {miss if miss else 'нет'}")
    print("   частоты символов: " + " ".join(f"{c!r}:{cover[c]}" for c in sorted(need)))
    tot = cmp_same + cmp_diff
    if tot:
        print(f"== ударения против авторских «+» RUSLAN: совпало {cmp_same}/{tot} "
              f"({100 * cmp_same / tot:.1f}%), примеры расхождений в stress_cmp.txt")
    print("== примеры:")
    for k in chosen[:8]:
        print("  ", k[2])


if __name__ == "__main__":
    main()
