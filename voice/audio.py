"""audio.py — звук: запись WAV, спектр, F0, выравнивание.

Тяжёлые librosa и torchaudio импортируются внутри функций: сервису они нужны
только на ресемпле, а обучению и вскрытию — постоянно.
"""
import io
import wave

import numpy as np
import torch

from voice.model import HOP, SR


def pcm_bytes(x):
    # округление, а не отбрасывание: раньше сервис резал дробную часть, а
    # write16 округлял — разница меньше младшего бита, но пусть будет одна
    return np.clip(np.round(np.asarray(x) * 32767.0), -32768, 32767).astype("<i2").tobytes()


def wav_from_pcm(pcm, sr, ch=1):
    """WAV вокруг готового PCM — сервису не надо гонять байты обратно во float"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_bytes(x, sr, ch=1):
    return wav_from_pcm(pcm_bytes(x), sr, ch)


def write16(path, x, sr=SR):
    with open(path, "wb") as f:
        f.write(wav_bytes(np.asarray(x), sr))


def speech_span(x, sr, frame=0.02, rel_db=-40.0):
    """границы речи по энергии: кадры не тише пика на rel_db; None — речи нет.
    У F5 нет шума комнаты, поэтому порог по энергии здесь честный (у записи
    с микрофона он принимал шорох мыши за речь — SILERO.md, шаг 2)."""
    fl = max(1, int(frame * sr))
    nf = len(x) // fl
    if nf < 3:
        return None
    e = np.sqrt((x[:nf * fl].reshape(nf, fl) ** 2).mean(axis=1))
    peak = float(e.max())
    if peak < 1e-5:
        return None
    idx = np.nonzero(e >= peak * 10.0 ** (rel_db / 20.0))[0]
    return int(idx[0] * fl), int(min(len(x), (idx[-1] + 1) * fl))


def frames_mag(x, c, n_fft=2400, hop=HOP, power=1.0, T=None):
    """|STFT| с окном Ханна: кадр t центрирован в t*hop + c (reflect-поля)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    P = n_fft
    xp = torch.nn.functional.pad(x[None, None], (P, P), mode="reflect")[0, 0]
    T = T or len(x) // hop
    starts = torch.arange(T) * hop + c + P - n_fft // 2
    idx = starts[:, None] + torch.arange(n_fft)[None]
    fr = xp[idx.clamp(0, len(xp) - 1)] * torch.hann_window(n_fft)
    S = torch.fft.rfft(fr, dim=-1).abs().T          # [n_fft/2+1, T]
    return S ** power


def f0_track(wav, T):
    """F0 (Гц) по кадрам модели: 16 кГц, шаг 200 = 12.5 мс, центр кадра t*200+100.
    NaN — глухо."""
    import librosa
    import torchaudio
    x = torchaudio.functional.resample(torch.as_tensor(wav), SR, 16000).numpy()
    x = np.pad(x, (0, 200 * T + 1024))
    f0, vflag, _ = librosa.pyin(x[100:], fmin=60, fmax=500, sr=16000, frame_length=1024,
                                hop_length=200, center=True)
    f0 = f0[:T]
    f0[~vflag[:T]] = np.nan
    return f0


def symbol_f0(f0, dur):
    """Средний F0 по звонким кадрам каждого символа (NaN, если звонких нет)."""
    cum = np.concatenate([[0], np.cumsum(dur)]).astype(int)
    out = np.full(len(dur), np.nan)
    for i in range(len(dur)):
        seg = f0[cum[i]:cum[i + 1]]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def feats(wav, T):
    """MFCC 1..20 из звука (16 кГц, шаг 12.5 мс, центр t*200+100), CMVN по фразе.
    Считаются из ЗВУКА, а не из мела модели — как будет с настоящей записью."""
    import librosa
    import torchaudio
    x = torchaudio.functional.resample(torch.as_tensor(wav), SR, 16000).numpy()
    x = np.pad(x, (0, 200 * T + 1024))[100:]
    m = librosa.feature.mfcc(y=x, sr=16000, n_mfcc=21, n_fft=1024, hop_length=200,
                             n_mels=80, center=True)[1:, :T]
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-5)


def dtw_durs(ref_f, ref_dur, tgt_f):
    """Длительности символов в цели: границы опорных символов через путь DTW."""
    import librosa
    _, wp = librosa.sequence.dtw(X=ref_f, Y=tgt_f, metric="euclidean")
    wp = wp[::-1]
    first = {}
    for r, t in wp:
        first.setdefault(int(r), int(t))
    cum = np.concatenate([[0], np.cumsum(ref_dur)]).astype(int)
    Tt = tgt_f.shape[1]
    b = [first.get(min(c, ref_f.shape[1] - 1), Tt) if c < ref_f.shape[1] else Tt for c in cum]
    b[0], b[-1] = 0, Tt
    b = np.maximum.accumulate(np.array(b))
    return np.diff(b)


def resample(x, src, dst):
    import torchaudio
    with torch.no_grad():
        return torchaudio.functional.resample(torch.as_tensor(x), src, dst).numpy()
