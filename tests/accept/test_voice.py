"""Приёмка весов: модель грузится, говорит разборчиво, стоковые голоса целы.

Нужны веса и кэш torch hub, а для CER — gigaam по соседству, поэтому всё
пропускается, если чего-то нет. Считается долго: минуты, а не секунды.
"""
import os

import pytest

from voice.paths import data

MODEL = os.environ.get("VOICE_MODEL", data("train/model_voice.pt"))
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tts():
    if not os.path.exists(MODEL):
        pytest.skip(f"нет весов {MODEL}")
    import torch
    from voice.model import load, load_state
    _, pk, tm = load()
    torch.set_grad_enabled(False)
    return pk, load_state(tm, MODEL, 4, "cpu")


def synth_text(tts, text, speaker="xenia"):
    from voice.model import encode, predict, synth
    pk, tm = tts
    seq, sp = encode(pk, text, speaker)
    mask, dur, pitch = predict(tm, seq, sp)
    _, wav = synth(tm, seq, sp, mask, dur, pitch)
    return wav.numpy()


def test_model_speaks_at_all(tts):
    from voice.model import SR
    w = synth_text(tts, "Прив+ет, э+то пров+ерка г+олоса.")
    assert len(w) / SR > 0.8
    assert abs(w).max() > 0.05


def test_cer_of_own_voice_is_small(tts):
    from voice.asr import transcribe_wav48
    from voice.phrases import phrases
    from voice.text import cer
    try:
        errs = [cer(t, transcribe_wav48(synth_text(tts, t))) for t in phrases(5)]
    except OSError:
        pytest.skip("gigaam не отвечает")
    assert sum(errs) / len(errs) < 0.05


def test_stock_voices_still_work(tts):
    from voice.asr import transcribe_wav48
    from voice.text import cer
    t = "Сейчас в Москве плюс двенадцать градусов."
    try:
        c = cer(t, transcribe_wav48(synth_text(tts, t, "aidar")))
    except OSError:
        pytest.skip("gigaam не отвечает")
    assert c < 0.05


def test_pause_cap_shortens_a_period_inside_a_phrase(tts):
    from voice.model import HOP, SR, cap_pauses, encode, predict
    pk, tm = tts
    seq, sp = encode(pk, "Дов+ольно дл+инное предлож+ение про всё на св+ете. Аг+а.",
                     "xenia")
    _, dur, _ = predict(tm, seq, sp)
    lim = 0.3           # после длинной фразы модель даёт точке 0.89 с
    capped = cap_pauses(tm, seq, dur, lim)
    punkt = [i for i, c in enumerate(seq[0].tolist()) if c in tm.punkt_symbols]
    assert max(dur[..., i].item() for i in punkt) * HOP / SR > lim   # есть что резать
    assert max(capped[..., i].item() for i in punkt) * HOP / SR <= lim
    # буквы не трогаем, исходный тензор тоже
    letters = [i for i in range(seq.shape[1]) if i not in punkt]
    assert all(capped[..., i].item() == dur[..., i].item() for i in letters)
    assert max(dur[..., i].item() for i in punkt) * HOP / SR > lim


def test_silero_stress_marks_the_way_silero_expects():
    """приёмка разметчика: «+» перед ударной гласной, омографы, «ё»"""
    from voice.text import Stress
    try:
        acc = Stress()
    except ImportError:
        pytest.skip("silero-stress не установлен")
    out = acc("Замок на двери и замок на горе")
    assert "зам+ок н+а двер+и" in out.lower()      # который вешают
    assert "з+амок н+а гор+е" in out.lower()       # который стоит
    assert acc("Лева Королев") == "Л+ёва Корол+ёв"
    assert acc("Он ушёл...") == "+Он уш+ёл…"       # символы Silero на выходе
    assert acc("«Макса» — дом") == "М+акса – д+ом"   # «—» не в алфавите
    # U+2011 приводим к обычному дефису, иначе Silero молча съест его вместе
    # с соседями; односложное «по» помечается как любое односложное слово
    assert acc("по‑английски") == "п+о-англ+ийски"
