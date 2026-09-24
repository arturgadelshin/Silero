#!/usr/bin/env python3
"""register.py — вписать готовую модель в реестр веб-морды без обучения.

Например, боевую model_voice.pt, на которой работает сервис:

    docker exec silero_web python -m web.register /w/train/model_voice.pt \
        --name artur

Файл подключается симлинком: это тот же файл, что читает сервис, и если его
переобучат вне морды — mtime изменится и движок перегрузит модель сам.
Превью озвучивается сразу. msc (μ/σ голоса, нужен ручке тональности) берётся
из stats.pt каталога целей, если он есть.
"""
import argparse
import os

from voice.paths import data

from web import jobs, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="путь к .pt (jit-модель или файл голоса [2,128])")
    ap.add_argument("--name", required=True)
    ap.add_argument("--kind", default="full", choices=["full", "probe"])
    ap.add_argument("--targets", default=data("targets"),
                    help="каталог целей обучения для msc (пусто — не задавать)")
    a = ap.parse_args()

    src = os.path.abspath(a.model)
    if not os.path.exists(src):
        raise SystemExit(f"нет файла {src}")
    mid = store.create_model(a.name, a.kind)
    tr = os.path.join(store.model_dir(mid), "train")
    os.makedirs(tr, exist_ok=True)
    dst = os.path.join(tr, os.path.basename(src))
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)

    meta = {"file": os.path.join("train", os.path.basename(src))}
    stats = os.path.join(a.targets, "stats.pt")
    if a.targets and os.path.exists(stats):
        import torch
        st = torch.load(stats, map_location="cpu")
        meta.update(msc=round(float(st["f0_mean"]) / float(st["f0_std"]), 2),
                    f0_mean=round(float(st["f0_mean"]), 1),
                    f0_std=round(float(st["f0_std"]), 1))
    store.save_meta(mid, **meta)
    jobs._preview(mid)
    print(f"модель «{a.name}» в реестре: {mid}")


if __name__ == "__main__":
    main()
