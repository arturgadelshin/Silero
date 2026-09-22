"""Звук: заголовки WAV и границы речи. Моно-заголовок поверх стерео однажды
уже стоил вечера («замедленная речь» в SILERO.md), поэтому каналы проверяем.
"""
import io
import wave

import numpy as np
import pytest

from voice.audio import pcm_bytes, speech_span, wav_bytes, wav_from_pcm, write16


def read(b):
    with wave.open(io.BytesIO(b), "rb") as w:
        return (w.getnchannels(), w.getsampwidth(), w.getframerate(),
                w.readframes(w.getnframes()))


def test_pcm_is_rounded_not_truncated():
    x = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    got = np.frombuffer(pcm_bytes(x), dtype="<i2")
    assert list(got) == [0, 32767, -32767, 16384]


def test_pcm_clips_out_of_range():
    got = np.frombuffer(pcm_bytes(np.array([2.0, -2.0])), dtype="<i2")
    assert list(got) == [32767, -32768]                 # весь диапазон int16


def test_wav_header_matches_data():
    ch, width, sr, data = read(wav_bytes(np.zeros(100, dtype=np.float32), 24000))
    assert (ch, width, sr, len(data)) == (1, 2, 24000, 200)


def test_wav_from_pcm_keeps_bytes():
    pcm = np.arange(-4, 4, dtype="<i2").tobytes()
    ch, width, sr, data = read(wav_from_pcm(pcm, 48000, ch=2))
    assert data == pcm and ch == 2 and sr == 48000


def test_write16_roundtrip(tmp_path):
    x = np.sin(np.arange(480) / 10).astype(np.float32)
    p = tmp_path / "a.wav"
    write16(str(p), x, 48000)
    ch, width, sr, data = read(p.read_bytes())
    back = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32767
    assert sr == 48000 and ch == 1
    assert np.abs(back - x).max() < 1e-4


def test_speech_span_finds_the_loud_part():
    sr = 16000
    x = np.zeros(sr, dtype=np.float32)
    x[4000:8000] = np.sin(np.arange(4000) / 3)
    lo, hi = speech_span(x, sr)
    assert 3800 <= lo <= 4100 and 7900 <= hi <= 8200


def test_speech_span_on_silence():
    assert speech_span(np.zeros(16000, dtype=np.float32), 16000) is None


def test_speech_span_on_too_short():
    assert speech_span(np.zeros(10, dtype=np.float32), 16000) is None
