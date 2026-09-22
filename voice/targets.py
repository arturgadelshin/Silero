"""targets.py — цели обучения, сложенные prep_targets.

По файлу .pt на фразу (мел, длительности, pitch, токены, split) плюс stats.pt
со средним и разбросом F0 и границей маски мела. Читают и обучение, и
диагностика, поэтому чтение живёт в общем слое.
"""
import os

import torch

from voice.paths import data


def load_targets(d=None, split=None):
    d = d or data("targets")
    files = sorted(f for f in os.listdir(d) if f.endswith(".pt") and f != "stats.pt")
    out = []
    for f in files:
        r = torch.load(os.path.join(d, f), map_location="cpu")
        if split is None or r["split"] == split:
            out.append(r)
    return out


def load_stats(d=None):
    """среднее и σ F0 по корпусу и номер полосы, выше которой мел вне лосса"""
    st = torch.load(os.path.join(d or data("targets"), "stats.pt"))
    return st["f0_mean"], st["f0_std"], st["mask_band"]
