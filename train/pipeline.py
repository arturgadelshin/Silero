#!/usr/bin/env python3
"""pipeline.py — конвейер обучения одной командой.

Ступени идут в том же порядке, что шаги SILERO.md, и запускаются как обычные
модули — pipeline только подставляет пути и печатает команду перед запуском,
чтобы любую ступень можно было повторить руками:

  text     шаг 1  тексты RUSLAN с ударениями            (CPU)
  refresh  выбросить звук и цели переразмеченных фраз  (CPU, нужен --old)
  f5       шаг 3/4 озвучка корпуса клоном F5            (GPU, сеть для gigaam)
  targets  шаг 5  цели: мел, длительности, pitch        (CPU, части параллельно)
  stats    шаг 5  второй проход: μ, σ F0 и маска мела   (CPU)
  stage1   шаг 6  только три строки голоса              (GPU)
  stage2   шаг 6  плюс декодер, lin и тела предсказателей (GPU)
  bake     просодия: растяжка pitch в веса              (CPU)
  check    приёмка обученной модели по CER gigaam       (CPU, --network host)

  docker run --rm --gpus all --network host \
    -v ~/linkplay/tmp/silero/hub:/root/.cache/torch/hub \
    -v ~/linkplay/tmp/silero/ruaccent:/ruaccent \
    -v ~/linkplay/silero:/app -v ~/linkplay/tmp/silero:/w -w /app \
    silero_ru python -m train.pipeline --from targets

Обучение занимает слот RTX 3070 надолго, поэтому по умолчанию печатается план,
а запускается он только с --run.
"""
import argparse
import subprocess
import sys

from voice.paths import data

ORDER = ["text", "refresh", "f5", "targets", "stats", "stage1", "stage2",
         "bake", "check"]


def steps(a):
    parts = [["-m", "train.prep_targets", "--part", f"{i}/{a.parts}"] for i in range(a.parts)]
    return {
        "text": [["-m", "train.prep_text"]],
        "refresh": [["-m", "train.refresh", "--old", a.old, "--run"]] if a.old else [],
        "f5": [["-m", "train.gen_f5"]],
        "targets": parts,
        "stats": [["-m", "train.prep_targets", "--stats"]],
        "stage1": [["-m", "train.train", "--stage", "1"]],
        "stage2": [["-m", "train.train", "--stage", "2",
                    "--resume", data("train/voice_stage1.pt")]],
        "bake": [["-m", "train.bake_prosody"]],
        "check": [["-m", "train.train", "--check", data("train/model_voice.pt")]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="text", choices=ORDER)
    ap.add_argument("--to", dest="stop", default="check", choices=ORDER)
    ap.add_argument("--only", default="", help="одна ступень вместо диапазона")
    ap.add_argument("--old", default="", help="прежний texts.tsv для ступени refresh")
    ap.add_argument("--parts", type=int, default=6, help="на сколько частей делить targets")
    ap.add_argument("--jobs", type=int, default=6, help="сколько частей targets считать разом")
    ap.add_argument("--run", action="store_true", help="не печатать план, а выполнять")
    a = ap.parse_args()

    plan = [a.only] if a.only else ORDER[ORDER.index(a.start):ORDER.index(a.stop) + 1]
    table = steps(a)
    for name in plan:
        cmds = [[sys.executable] + argv for argv in table[name]]
        for cmd in cmds:
            print(f"[{name}] " + " ".join(cmd), flush=True)
        if not a.run:
            continue
        # части targets считаются разом: они независимы и упираются в CPU
        jobs = a.jobs if name == "targets" else 1
        for i in range(0, len(cmds), jobs):
            procs = [subprocess.Popen(c) for c in cmds[i:i + jobs]]
            codes = [p.wait() for p in procs]
            if any(codes):
                raise SystemExit(f"ступень {name} упала: коды {codes}")
    if not a.run:
        print("\nэто план; чтобы выполнить, добавьте --run")


if __name__ == "__main__":
    main()
