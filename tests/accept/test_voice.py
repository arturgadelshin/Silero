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
