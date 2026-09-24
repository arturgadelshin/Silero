"""refprep.py — опорная запись для клонирования: декодирование и чистка.

Веб-морда принимает файл (wav/mp3/flac/ogg и что сумеет ffmpeg) или запись
с микрофона (браузер отдаёт уже WAV 48 кГц моно). Дальше запись надо привести
к тому виду, в котором её ест F5 — так же, как опору готовили руками в
SILERO.md, шаг 2:

  - моно 48 кГц, без постоянной составляющей;
  - границы речи по звонким кадрам, а не по порогу «пик − 30 дБ» (порог
    принимал шорох мыши за речь);
  - паузы внутри сжаты до 0.3 с, по краям 0.2 с тишины — F5 переносит паузы
    опоры в генерацию и режет опору длиннее ~12 с;
  - громкость поднята до пика −3 dBFS.

Текст опоры размечается ударениями тем же silero-stress, что и у сервиса,
и показывается пользователю для ручной правки («+» перед ударной гласной).
"""
import os
import subprocess
import threading

import numpy as np

from voice.paths import data

MAX_PAUSE = 0.30
EDGE = 0.20
FRAME = 0.02
PEAK_DB = -3.0

_acc = None
_acc_lock = threading.Lock()


def accent():
    """один раз загруженный silero-stress (~3.8 с) — общий на движок и опоры"""
    global _acc
    with _acc_lock:
        if _acc is None:
            from voice.text import accentor
            _acc = accentor("silero")
        return _acc


def mark(text):
    """чистый текст -> текст с ударениями в формате Silero"""
    return accent()(text)


# ---------------------------------------------------------------- декодирование
def decode(path):
    """любой вход -> (моно float32, частота). Цепочка: soundfile -> torchaudio
    -> ffmpeg; у soundfile свежий libsndfile читает и mp3."""
    err = None
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return x.mean(axis=1), int(sr)
    except Exception as exc:                        # noqa: BLE001
        err = exc
    try:
        import torchaudio
        t, sr = torchaudio.load(path)
        return t.mean(dim=0).numpy().astype("float32"), int(sr)
    except Exception:                               # noqa: BLE001
        pass
    import shutil
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
                            "-ac", "1", "-ar", "48000", "-"],
                           capture_output=True, timeout=120)
        if r.returncode == 0 and r.stdout:
            return np.frombuffer(r.stdout, dtype="<f4").copy(), 48000
    raise ValueError(f"не удалось прочитать аудио: {type(err).__name__}: {err}")


def to48(x, sr):
    if sr == 48000:
        return x
    from voice.audio import resample
    return resample(x, sr, 48000)


# ---------------------------------------------------------------- чистка
def _frames(x, sr):
    fl = max(1, int(FRAME * sr))
    nf = len(x) // fl
    return x[:nf * fl].reshape(nf, fl), nf, fl


def _energy_mask(x, sr, rel_db=-35.0):
    fr, nf, fl = _frames(x, sr)
    if nf < 5:
        return None
    e = np.sqrt((fr ** 2).mean(axis=1))
    peak = float(e.max())
    if peak < 1e-4:
        return None
    return e >= peak * 10.0 ** (rel_db / 20.0)


def _voiced_mask(x, sr):
    """звонкие кадры pyin; None — если librosa не справился"""
    try:
        import librosa
        hop = max(1, int(FRAME * sr))
        fl = 2048 if sr >= 24000 else 1024
        _, v, _ = librosa.pyin(x.astype(np.float64), fmin=60, fmax=500, sr=sr,
                               frame_length=fl, hop_length=hop)
        return np.nan_to_num(v.astype(bool))
    except Exception:                               # noqa: BLE001
        return None


def clean(x, sr):
    """обрезка краёв, сжатие пауз, нормировка громкости. Возвращает
    (звук, инфо) — инфо с секундами речи и предупреждениями."""
    warns = []
    x = np.asarray(x, dtype=np.float32) - float(np.mean(x))
    if len(x) / sr > 90:
        raise ValueError("запись длиннее 90 с — для опоры нужна одна фраза")
    if float(np.abs(x).max()) >= 0.999:
        warns.append("есть клиппинг — запись слишком громкая, запишите тише")
    em = _energy_mask(x, sr)
    if em is None:
        raise ValueError("речь не найдена: тишина или очень тихая запись")
    vm = _voiced_mask(x, sr)
    if vm is None or not vm.any():
        warns.append("не нашли звонких кадров — проверьте запись на слух")
        vm = np.zeros(len(em), dtype=bool)
    if len(vm) < len(em):                           # у pyin своё число кадров
        vm = np.pad(vm[:len(em)], (0, len(em) - len(vm[:len(em)])))
    speech = em | vm[:len(em)]
    nz = np.nonzero(speech)[0]
    first, last = int(nz[0]), int(nz[-1])           # в кадрах

    keep = np.zeros(last - first + 1, dtype=bool)
    keep[speech[first:last + 1]] = True
    fl = max(1, int(FRAME * sr))
    max_pause = int(MAX_PAUSE / FRAME)
    # тишина внутри: держим середину отрезка не длиннее max_pause кадров
    silent = ~keep
    k = 0
    while k < len(silent):
        if not silent[k]:
            k += 1
            continue
        j = k
        while j < len(silent) and silent[j]:
            j += 1
        if j - k > max_pause:
            cut_l = k + (j - k - max_pause) // 2
            silent[cut_l:cut_l + max_pause] = False
        k = j
    keep |= ~silent

    _, nf, fl = _frames(x, sr)
    idx = np.nonzero(keep)[0] + first
    y = np.concatenate([
        np.zeros(int(EDGE * sr), dtype=np.float32),
        np.concatenate([x[i * fl:(i + 1) * fl] for i in idx]),
        np.zeros(int(EDGE * sr), dtype=np.float32),
    ]) if len(idx) else np.zeros(int(EDGE * 2 * sr), dtype=np.float32)

    peak = float(np.abs(y).max())
    if peak > 1e-6:
        y = y * (10.0 ** (PEAK_DB / 20.0) / peak)
    speech_sec = len(idx) * FRAME
    dur = len(y) / sr
    if dur > 15:
        warns.append(f"опора вышла {dur:.1f} с — F5 режет всё после ~12 с, "
                     "сократите фразу или перезапишите")
    if dur < 2.5:
        warns.append(f"опора всего {dur:.1f} с — для клона лучше 5–10 с")
    if peak and 20 * np.log10(float(np.abs(x).max())) < -35:
        warns.append("запись очень тихая — поднимите усиление микрофона")
    return y, {"speech_sec": round(speech_sec, 2), "duration": round(dur, 2),
               "warnings": warns}


def asr_check(x48, text_plain):
    """CER расшифровки против текста опоры (gigaam по wyoming); None — ASR нет"""
    try:
        from voice.asr import transcribe_pcm
        from voice.audio import pcm_bytes, resample
        from voice.text import cer
        x16 = resample(x48, 48000, 16000)
        hyp = transcribe_pcm(pcm_bytes(x16), timeout=30.0)
        if not hyp:
            return None
        return round(cer(text_plain, hyp), 3), hyp
    except Exception:                               # noqa: BLE001
        return None


def f0_median(x, sr):
    """медиана F0 чистой записи: pyin по звонким кадрам. Нужна выбору
    стартовой строки и DTW-опоры prep_targets (eugene брался под artur как
    ближайший по F0 — женской записи нужна женская опора)."""
    try:
        import librosa
        fl = 2048 if sr >= 24000 else 1024
        f0, v, _ = librosa.pyin(np.asarray(x, dtype=np.float64), fmin=60,
                                fmax=500, sr=sr, frame_length=fl,
                                hop_length=max(1, int(FRAME * sr)))
        f0 = f0[v.astype(bool)]
        if not len(f0):
            return None
        return float(np.nanmedian(f0))
    except Exception:                                # noqa: BLE001
        return None


def prepare_upload(raw_path, text_plain):
    """полный путь загрузки: декодировать, почистить, разметить текст.
    Возвращает dict для API и звук 48 кГц. Из аудио ничего не вырезается —
    путь как у оригинального проекта (чистка пауз, края, пик −3 dBFS)."""
    x, sr = decode(raw_path)
    y, info = clean(to48(x, sr), 48000)
    f0 = f0_median(y, 48000)
    marked = mark(text_plain)
    chk = asr_check(y, text_plain)
    out = dict(info, marked_text=marked, asr=None, f0_med=f0)
    if chk is not None:
        out["asr"] = {"cer": chk[0], "text": chk[1]}
    return out, y
