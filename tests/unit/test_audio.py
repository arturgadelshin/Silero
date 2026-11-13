"""Звук: заголовки WAV и границы речи. Моно-заголовок поверх стерео однажды
уже стоил вечера («замедленная речь» в SILERO.md), поэтому каналы проверяем.
"""
import io
import wave

import numpy as np
import pytest

from voice.audio import (loudness_normalize, lufs, pcm_bytes, speech_span,
                         wav_bytes, wav_from_pcm, write16)


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


# --- громкость (ITU-R BS.1770) ---------------------------------------------

def tone(freq=1000.0, amp=0.5, sec=2.0, sr=48000):
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_lufs_reference_tone():
    # 1 кГц на половину шкалы: -0.691 + 20*log10(0.5/sqrt(2)) = -9.72, плюс
    # +0.66 дБ подъём полки K-взвешивания на 1 кГц -> -9.06 LUFS. Калибровка
    # ITU (-20 dBFS 997 Гц -> -23.0 LUFS) соблюдается тем же расчётом.
    assert abs(lufs(tone()) - (-9.06)) < 0.1


def test_lufs_tracks_gain():
    a, b = tone(amp=0.2), tone(amp=0.4)             # +6 дБ амплитуды
    assert abs((lufs(b) - lufs(a)) - 6.02) < 0.1


def test_lufs_ignores_quiet_tail():
    # гейт выкидывает блоки чистой тишины, но граничные блоки «тон+тишина»
    # по правилам BS.1770 остаются — они и дают эти ~0.5 дБ (ffmpeg ebur128
    # на том же сигнале считает столько же)
    x = np.concatenate([tone(sec=1.5), np.zeros(48000, dtype=np.float32)])
    assert abs(lufs(x) - lufs(tone(sec=1.5))) < 0.6


def test_lufs_silence_is_minus_inf():
    assert lufs(np.zeros(48000, dtype=np.float32)) == float("-inf")


def test_normalize_hits_target_without_clip():
    x = tone(amp=0.1)
    y = loudness_normalize(x, target=-13.0)
    assert abs(lufs(y) - (-13.0)) < 0.2
    assert np.abs(y).max() <= 10 ** (-1.1 / 20) + 1e-6


def test_normalize_limits_loud_peaks():
    x = tone(amp=0.9)                                # выше цели без лимитера
    y = loudness_normalize(x, target=-13.0)
    assert np.abs(y).max() <= 10 ** (-1.1 / 20) + 1e-6
    assert abs(lufs(y) - (-13.0)) < 0.3


def test_normalize_keeps_silence_silent():
    x = np.zeros(4800, dtype=np.float32)
    assert np.abs(loudness_normalize(x)).max() == 0.0
