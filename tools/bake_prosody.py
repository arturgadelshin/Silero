#!/usr/bin/env python3
"""bake_prosody.py — вложить в веса растяжку pitch и замедление речи.

Предсказатель pitch сглаживает: MSE оптимален на условном среднем, а среднее
ровнее правды. Опыт `prosody_try.py` показал, что дело не в качестве
предсказания (истинный pitch даёт тот же размах F0), а в масштабе: растяжка
предсказанного pitch в 1.5 раза возвращает размах 47 → 63 Гц при 64 у F5.
Растяжка вокруг нуля и вокруг среднего фразы звучат одинаково, поэтому её
можно вложить прямо в последний слой и не менять код синтеза.

  pitch: p = W·h + b  →  умножаем W и b на k, получаем ровно k·p
  dur:   d = round(exp(y) − 1) — сдвиг b даёт не чистое умножение: к каждому
         символу добавляется (g − 1) кадра, а потом всё округляется, и на
         коротких символах набегает лишнее (ln 1.08 дал 1.107×). Поэтому сдвиг
         подбирается делением пополам по настоящей сумме кадров на пробных фразах.

Оба слоя общие на все голоса, поэтому стоковые голоса тоже станут живее и
медленнее. Они и так изменены ступенью 2.

  docker run --rm -v ~/linkplay/tmp/silero/hub:/root/.cache/torch/hub \
    -v ~/linkplay/tmp/silero:/w silero_ru python /w/bake_prosody.py \
    --model /w/train/model_stage2.pt --out /w/train/model_voice.pt
"""
import argparse
import math
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/w/train/model_stage2.pt")
    ap.add_argument("--out", default="/w/train/model_voice.pt")
    ap.add_argument("--pitch-gain", type=float, default=1.5)
    ap.add_argument("--dur-gain", type=float, default=1.0,
                    help="темп: 1.0 выбран на слух, 1.08 был первой пробой")
    ap.add_argument("--row", type=int, default=4)
    ap.add_argument("--probe", type=int, default=20, help="фраз для подбора сдвига dur")
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from silero_probe2 import load, encode, predict
    from train_voice import load_state

    _, pk, tm = load()
    torch.set_grad_enabled(False)
    tm = load_state(tm, a.model, a.row, "cpu")

    from silero_probe2 import phrases
    probe = [encode(pk, t, "xenia") for t in phrases(a.probe)]
    base = [predict(tm, s, v) for s, v in probe]
    d0 = sum(float(x[1].sum()) for x in base)

    P = dict(tm.named_parameters())
    P["pitch_predictor.pitch_pred.lin.weight"].mul_(a.pitch_gain)
    P["pitch_predictor.pitch_pred.lin.bias"].mul_(a.pitch_gain)

    bias = P["dur_predictor.dur_pred.lin.bias"]
    b0 = bias.clone()

    def ratio(delta):
        bias.copy_(b0 + delta)
        return sum(float(predict(tm, s, v)[1].sum()) for s, v in probe) / d0

    lo, hi = 0.0, math.log(a.dur_gain) + 0.1        # сумма кадров растёт по сдвигу
    for _ in range(20):
        mid = (lo + hi) / 2
        if ratio(mid) < a.dur_gain:
            lo = mid
        else:
            hi = mid
    delta = (lo + hi) / 2
    got_d = ratio(delta)

    now = [predict(tm, s, v) for s, v in probe]
    p0 = torch.cat([x[2].flatten() for x in base])
    p1 = torch.cat([x[2].flatten() for x in now])
    nz = p0.abs() > 1e-3
    got_p = float((p1[nz] / p0[nz]).median())
    print(f"pitch: σ {float(p0[nz].std()):.3f} → {float(p1[nz].std()):.3f}, "
          f"отношение по символам {got_p:.3f} (просили {a.pitch_gain})")
    print(f"dur:   сдвиг {delta:+.5f} (чистый ln g был бы {math.log(a.dur_gain):+.5f}), "
          f"кадров {d0:.0f} → {d0 * got_d:.0f}, {got_d:.3f}× "
          f"на {a.probe} фразах (просили {a.dur_gain})")
    if abs(got_p - a.pitch_gain) > 0.02 or abs(got_d - a.dur_gain) > 0.01:
        raise SystemExit("не совпало с заказанным — веса не те")

    torch.jit.save(tm, a.out)
    print(f"сохранено {a.out}, {os.path.getsize(a.out) / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
