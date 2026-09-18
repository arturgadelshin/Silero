#!/usr/bin/env python3
"""silero_probe2.py — вскрытие 2 Silero TTS v5_5_ru: закрыть оговорки SILERO.md.

Только чтение модели (стоковые веса не сохраняются; всё записанное — во --out).
Запуск (сервер, CPU):
  docker run --rm -v ~/linkplay/tmp/silero/hub:/root/.cache/torch/hub \
    -v ~/linkplay/tmp/silero:/w kws_tts python /w/silero_probe2.py <cmd> --out /w/probe2

Подкоманды:
  src    исходники из *.debug_pkl пакета + поиск параметров по строкам, режим train()
  mel    формула мела: перебор sr/окна/шкалы/fmin/fmax по звуку вокодера (аффинная подгонка)
  pitch  нормировка pitch: F0 синтеза против поданного pitch, опыт со сдвигом δ
  align  DTW-выравниватель длительностей на синтетической правде
  space  интерполяция/случайные голоса + CER gigaam; сохранение своего голоса
"""
import argparse
import collections
import os
import pickle
import re
import wave
import zipfile

import numpy as np
import torch

HUB = "snakers4/silero-models"
PKG = "/root/.cache/torch/hub/snakers4_silero-models_master/src/silero/model/v5_5_ru.pt"
SR = 48000
HOP = 600
VOICES = ["aidar", "baya", "kseniya", "eugene", "xenia"]


def write16(path, x, sr=SR):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.clip(np.round(np.asarray(x) * 32767.0), -32768, 32767)
                       .astype("<i2").tobytes())


def load():
    torch.set_num_threads(int(os.environ.get("SILERO_THREADS", "8")))
    model, _ = torch.hub.load(HUB, "silero_tts", language="ru", speaker="v5_5_ru")
    pk = model.packages[0]
    return model, pk, pk.models[0]


def encode(pk, text, voice):
    """Вход модели ровно как в apply_tts (ударения акцентора, sos/eos)."""
    sp_ids, _ = pk.get_speakers(voice)
    sentences, _, breaks, rates, pitches, sp = pk.prepare_tts_model_input(
        text, ssml=False, speaker_ids=sp_ids, lang=None)
    seq, symb_durs, durs_rate, pitch_coefs = pk.merge_batch_model(sentences, breaks, rates, pitches)
    return seq, sp


def predict(tm, seq, sp):
    """Предсказанные длительности (кадры на символ) и нормированный pitch."""
    mask = torch.zeros(1, seq.shape[1], dtype=torch.bool)
    ld = tm.dur_predictor(seq, sp, mask, 1.0, None)
    dur = torch.round(torch.clamp(torch.exp(ld) - 1, min=0))
    pitch = tm.pitch_predictor(seq, sp, mask, None, None)
    # как в forward: на пунктуации pitch обнуляется
    cum = torch.cumsum(torch.nn.functional.pad(dur[0], (1, 0)), 0)
    for i, c in enumerate(seq[0].tolist()):
        if c in tm.punkt_symbols:
            pitch[0, 0, i] = 0.0
    return mask, dur, pitch


def synth(tm, seq, sp, mask, dur, pitch):
    mel = tm.tacotron(seq, sp, mask, dur, pitch, None)
    wav = tm.vocoder(mel, SR, 0.0, True)
    return mel, wav.reshape(-1)


def phrases(n):
    """Фразы для замеров: ответы ассистента + разнообразные предложения."""
    base = [
        "Таймер на пять минут поставлен.", "Сейчас в Москве плюс двенадцать градусов.",
        "Будильник сработает в семь тридцать утра.", "Включаю музыку в гостиной.",
        "Громкость уменьшена до сорока процентов.", "Извините, я не расслышала вопрос.",
        "Завтра ожидается дождь и сильный ветер.", "Напоминание удалено.",
        "Хорошо, выключаю свет на кухне.", "До конца таймера осталась одна минута.",
        "Съешь же ещё этих мягких французских булок да выпей чаю.",
        "Широкая электрификация южных губерний даст мощный толчок подъёму сельского хозяйства.",
        "В чащах юга жил бы цитрус? Да, но фальшивый экземпляр!",
        "Эй, жлоб! Где туз? Прячь юных съёмщиц в шкаф.",
        "Любя, съешь щипцы, вздохнёт мэр, кайф жгуч.",
        "Флегматичная эта верблюдица жует у подъезда засыхающий горький шиповник.",
        "Мы с тобой пойдём гулять, когда закончится дождь.",
        "Почему небо голубое, а трава зелёная?",
        "Позвони маме и скажи, что я задержусь на час.",
        "Эта песня играла на прошлой неделе по радио.",
    ]
    rng = np.random.RandomState(7)
    words = " ".join(base).lower().replace(".", "").replace(",", "").replace("!", "") \
        .replace("?", "").split()
    out = list(base)
    while len(out) < n:
        k = rng.randint(4, 12)
        s = " ".join(rng.choice(words, k))
        out.append(s[0].upper() + s[1:] + rng.choice([".", "?", "!", "."]))
    return out[:n]


# ---------------------------------------------------------------- src
def cmd_src(a):
    z = zipfile.ZipFile(PKG)
    root = os.path.join(a.out, "src_debug")
    os.makedirs(root, exist_ok=True)
    files = collections.defaultdict(dict)       # имя файла -> {строка: текст фрагмента}
    for n in z.namelist():
        if not n.endswith(".debug_pkl"):
            continue
        o = pickle.loads(z.read(n))
        if not (isinstance(o, tuple) and o and o[0] == "FORMAT_WITH_STRING_TABLE"):
            continue
        table, ranges = o[1], o[2]
        for rec in ranges:
            src = rec[1][0]
            idx, fidx, line = src
            if not idx:
                continue
            text = "".join(table[i] for i in idx)
            files[table[fidx]].setdefault(line, text)
    print(f"== исходных файлов в debug_pkl: {len(files)}")
    for fname, frags in sorted(files.items()):
        if "site-packages" in fname or not fname.endswith(".py"):
            continue
        out = os.path.join(root, fname.replace("/mnt/junior/tts_data/raw/pitch_predictor/", "")
                           .lstrip("/").replace("/", "__"))
        with open(out, "w", encoding="utf-8") as f:
            for line in sorted(frags):
                f.write(f"# ---- {fname}:{line}\n{frags[line]}\n")
        body = "".join(frags.values())
        drop = sorted(set(re.findall(r"self\.(\w*dropout\w*)", body)))
        train = len(re.findall(r"self\.training", body))
        print(f"  {fname.split('silero_vocoder_jit_pitch/')[-1]:45s} фрагментов {len(frags):3d}"
              f"  dropout: {','.join(drop) or '-'}  self.training: {train}")

    # все строки пакета, похожие на параметры обработки звука
    print("\n== строки пакета с параметрами звука/pitch")
    pat = re.compile(r"(fmin|fmax|f_min|f_max|mel_fmin|mel_fmax|n_mels|n_fft|hop_length|"
                     r"win_length|sampling_rate|sample_rate|pitch_mean|pitch_std|f0|"
                     r"mean_std|norm_pitch|log_dur|preemph)", re.I)
    seen = set()
    for n in z.namelist():
        if n.endswith((".storage",)) or "/data/" in n and n.endswith(".storage"):
            continue
        try:
            s = z.read(n).decode("latin1")
        except Exception:                        # noqa: BLE001
            continue
        for m in pat.finditer(s):
            ctx = s[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
            ctx = re.sub(r"[^\x20-\x7eЀ-ӿ]", ".", ctx)
            key = (n.split("/")[-1], ctx)
            if key not in seen and "site-packages" not in ctx:
                seen.add(key)
    by = collections.Counter(k[0] for k in seen)
    for k in sorted(seen)[:80]:
        print(f"  [{k[0]}] {k[1]}")
    print(f"  всего совпадений {len(seen)} по файлам {dict(by)}")

    # dropout в графе: какие модули и с каким p
    _, pk, tm = load()
    print("\n== Dropout-модули в графе")
    ps = collections.Counter()
    for name, m in tm.named_modules():
        if getattr(m, "original_name", "") == "Dropout":
            try:
                p = m.p
            except Exception:                    # noqa: BLE001
                p = "?"
            ps[(name.split(".")[0], p)] += 1
    for (top, p), c in sorted(ps.items(), key=str):
        print(f"  {top:16s} p={p}  x{c}")

    # поведение train(): меняется ли выход
    seq, sp = encode(pk, "Таймер на пять минут поставлен.", "aidar")
    torch.set_grad_enabled(False)
    mask, dur, pitch = predict(tm, seq, sp)
    tm.train(False)
    m0 = tm.tacotron(seq, sp, mask, dur, pitch, None)
    tm.train(True)
    m1 = tm.tacotron(seq, sp, mask, dur, pitch, None)
    m2 = tm.tacotron(seq, sp, mask, dur, pitch, None)
    d0 = tm.dur_predictor(seq, sp, mask, 1.0, None)
    d1 = tm.dur_predictor(seq, sp, mask, 1.0, None)
    tm.train(False)
    print(f"\n== train(True): мел eval-train {(m1 - m0).abs().mean():.4f}, "
          f"train-train {(m2 - m1).abs().mean():.4f}; dur train-train {(d1 - d0).abs().mean():.4f}")


# ---------------------------------------------------------------- mel
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


def collect(n, voices=VOICES):
    """Пары (мел tacotron, звук вокодера) по n фразам на голос."""
    _, pk, tm = load()
    torch.set_grad_enabled(False)
    data = []
    for v in voices:
        for t in phrases(n):
            seq, sp = encode(pk, t, v)
            mask, dur, pitch = predict(tm, seq, sp)
            mel, wav = synth(tm, seq, sp, mask, dur, pitch)
            data.append((v, t, seq, dur, pitch, mel[0], wav))
    return data, pk, tm


def cached(a, n):
    """collect() с кэшем на диске: синтез дорогой, а формулы перебираются много раз."""
    path = os.path.join(a.out, f"mel_data_{n}.pt")
    if os.path.exists(path):
        return torch.load(path)
    data, _, _ = collect(n)
    data = [(v, t, seq, dur, pitch, mel, wav) for v, t, seq, dur, pitch, mel, wav in data]
    torch.save(data, path)
    return data


def cmd_mel(a):
    import torchaudio
    data = cached(a, a.n)
    Y = torch.cat([d[5] for d in data], 1)
    print(f"== фраз {len(data)}, кадров {Y.shape[1]}; мел tacotron: min {Y.min():.2f} "
          f"mean {Y.mean():.2f} max {Y.max():.2f}")

    def spec(sr, n_fft, win, c48, power):
        Xs = []
        hop = int(round(0.0125 * sr))
        for d in data:
            wav = d[6]
            T = d[5].shape[1]
            if sr != SR:
                wav = torchaudio.functional.resample(wav, SR, sr)
            c = int(round(c48 * sr / SR))
            x = torch.nn.functional.pad(wav[None, None], (n_fft, n_fft), mode="reflect")[0, 0]
            starts = torch.arange(T) * hop + c + n_fft - n_fft // 2
            idx = (starts[:, None] + torch.arange(n_fft)[None]).clamp(0, len(x) - 1)
            w = torch.zeros(n_fft)
            lo = (n_fft - win) // 2
            w[lo:lo + win] = torch.hann_window(win)
            Xs.append(torch.fft.rfft(x[idx] * w, dim=-1).abs().T ** power)
        return torch.cat(Xs, 1)

    def score(X, sr, n_fft, fmin, fmax, norm, scale):
        fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, float(fmin), float(fmax),
                                                   192, sr, norm, scale)
        L = torch.log(torch.clamp(fb.T @ X, min=1e-5))
        # аффинная связь: наклон 1 = ln, 0.434 = log10; сдвиг = усиление
        A = torch.stack([L.flatten(), torch.ones(L.numel())], 1)
        sol = torch.linalg.lstsq(A, Y.flatten()[:, None]).solution[:, 0]
        r = (Y - (sol[0] * L + sol[1])).abs().mean().item()
        return r, sol[0].item(), sol[1].item()

    res = []
    configs = []
    for sr in (48000, 24000, 22050):
        for n_fft, win in ((2400, 2400), (2048, 2048), (4096, 2400), (1024, 1024), (1200, 1200),
                           (1102, 1102), (2048, 1200)):
            n_fft_s = int(round(n_fft * sr / SR)) if sr != SR and n_fft in (2400, 4096, 1200) else n_fft
            win_s = int(round(win * sr / SR)) if sr != SR and win in (2400, 1200) else win
            if sr != SR and (n_fft_s, win_s) in [(c[1], c[2]) for c in configs if c[0] == sr]:
                continue
            configs.append((sr, n_fft_s, min(win_s, n_fft_s)))
    for sr, n_fft, win in configs:
        for power in (1.0, 2.0):
            X = spec(sr, n_fft, win, 300, power)
            for scale in ("slaney", "htk"):
                for norm in (None, "slaney"):
                    for fmin in (0, 20, 40, 55, 70, 80, 90, 125):
                        for fmax in sorted({7600, 8000, 11025, 12000, sr // 2}):
                            if fmax > sr // 2:
                                continue
                            r, k, b = score(X, sr, n_fft, fmin, fmax, norm, scale)
                            res.append((r, k, b, sr, n_fft, win, power, scale, norm, fmin, fmax))
        best_sr = min((x for x in res if x[3] == sr and x[4] == n_fft), key=lambda x: x[0])
        print(f"  sr={sr} n_fft={n_fft} win={win}: лучший |d|={best_sr[0]:.3f} "
              f"(наклон {best_sr[1]:.3f}, power {best_sr[6]}, {best_sr[7]}/{best_sr[8]}, "
              f"fmin {best_sr[9]}, fmax {best_sr[10]})", flush=True)
    res.sort(key=lambda r: r[0])
    print("\n== лучшие 10 формул (после аффинной подгонки)")
    for r in res[:10]:
        print(f"  |d|={r[0]:.3f} наклон {r[1]:.3f} сдвиг {r[2]:+.3f}  sr={r[3]} n_fft={r[4]} "
              f"win={r[5]} power={r[6]} scale={r[7]} norm={r[8]} fmin={r[9]} fmax={r[10]}")
    torch.save(res[:50], os.path.join(a.out, "mel_grid.pt"))


# ---------------------------------------------------------------- pitch
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


def cmd_pitch(a):
    import warnings
    warnings.filterwarnings("ignore")
    _, pk, tm = load()
    torch.set_grad_enabled(False)
    msc = tm.mean_std_coef
    texts = phrases(a.n)
    print(f"== фраз на голос {len(texts)}; mean_std_coef = {[round(v, 2) for v in msc]}")
    for vi, v in enumerate(VOICES):
        P, F = [], []
        zero_voiced = zero_total = 0
        shifts = collections.defaultdict(list)
        for ti, t in enumerate(texts):
            seq, sp = encode(pk, t, v)
            mask, dur, pitch = predict(tm, seq, sp)
            mel, wav = synth(tm, seq, sp, mask, dur, pitch)
            T = mel.shape[-1]
            d = dur[0].numpy()
            f0 = f0_track(wav.numpy(), T)
            sf = symbol_f0(f0, d)
            p = pitch[0, 0].numpy()
            ok = ~np.isnan(sf) & (np.abs(p) > 1e-3) & (d > 0)
            P.append(p[ok])
            F.append(sf[ok])
            zero_total += int((np.abs(p) < 1e-3).sum())
            zero_voiced += int(((np.abs(p) < 1e-3) & ~np.isnan(sf)).sum())
            # опыт: весь pitch сдвинут на δ (кроме нулей пунктуации)
            if ti < 6:
                nz = torch.abs(pitch) > 1e-3
                for delta in (-1.0, -0.5, 0.5, 1.0, 2.0):
                    _, w2 = synth(tm, seq, sp, mask, dur, torch.where(nz, pitch + delta, pitch))
                    f2 = f0_track(w2.numpy(), T)
                    both = ~np.isnan(f0) & ~np.isnan(f2)
                    if both.sum() > 5:
                        shifts[delta].append((np.median(f2[both] - f0[both]),
                                              np.median(np.log2(f2[both] / f0[both])) * 12))
        P = np.concatenate(P)
        F = np.concatenate(F)
        # две гипотезы нормировки: линейно в Гц и в логарифме
        kh, bh = np.polyfit(P, F, 1)                 # F = sigma*P + mu
        kl, bl = np.polyfit(P, np.log(F), 1)
        rh = np.corrcoef(P, F)[0, 1]
        rl = np.corrcoef(P, np.log(F))[0, 1]
        print(f"\n[{v}] символов {len(P)}; F0 {np.median(F):.0f} Гц (5–95%: "
              f"{np.percentile(F, 5):.0f}–{np.percentile(F, 95):.0f})")
        print(f"  Гц:  F0 = {kh:6.1f}·p + {bh:6.1f}   r={rh:.3f}   mu/sigma={bh / kh:.2f} "
              f"(mean_std_coef {msc[vi]:.2f})")
        print(f"  log: lnF0 = {kl:.3f}·p + {bl:.3f}  r={rl:.3f}   "
              f"(= {kl / np.log(2) * 12:.1f} полутона на единицу p)")
        print(f"  pitch=0 у {zero_total} символов, из них звонких {zero_voiced}")
        for delta in sorted(shifts):
            hz = np.median([s[0] for s in shifts[delta]])
            st = np.median([s[1] for s in shifts[delta]])
            print(f"  δ={delta:+.1f}: ΔF0 {hz:+6.1f} Гц ({st:+5.2f} полутона); "
                  f"прогноз Гц-гипотезы {delta * kh:+6.1f} Гц")


# ---------------------------------------------------------------- align
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


def cmd_align(a):
    import warnings
    warnings.filterwarnings("ignore")
    _, pk, tm = load()
    torch.set_grad_enabled(False)
    rng = np.random.RandomState(3)
    texts = phrases(a.n)
    ref_voice = "aidar"
    cases = [("aidar", "темп 1.25", 1.25, 0.0), ("kseniya", "темп 1.0", 1.0, 0.0),
             ("kseniya", "темп 0.8", 0.8, 0.0), ("baya", "разброс ±30%", 1.0, 0.3),
             ("eugene", "темп 1.2 + разброс ±30%", 1.2, 0.3)]
    print(f"== опора {ref_voice}, фраз {len(texts)}; ошибка границ символов в кадрах (12.5 мс)")
    for tv, name, rate, jit in cases:
        errs, big, zero_hit = [], 0, 0
        for t in texts:
            seq, sp_r = encode(pk, t, ref_voice)
            mask, d_ref, p_ref = predict(tm, seq, sp_r)
            mel_r, w_r = synth(tm, seq, sp_r, mask, d_ref, p_ref)
            _, sp_t = encode(pk, t, tv)
            _, d_t, p_t = predict(tm, seq, sp_t)
            scale = rate * (1 + jit * (2 * rng.rand(d_t.shape[1]) - 1)) if jit else rate
            d_true = torch.round(d_t * torch.as_tensor(scale, dtype=torch.float32))
            mel_t, w_t = synth(tm, seq, sp_t, mask, d_true, p_t)
            fr = feats(w_r.numpy(), mel_r.shape[-1])
            ft = feats(w_t.numpy(), mel_t.shape[-1])
            d_est = dtw_durs(fr, d_ref[0].numpy(), ft)
            cb_t = np.cumsum(d_true[0].numpy())
            cb_e = np.cumsum(d_est)
            e = np.abs(cb_e - cb_t)[:-1]
            errs.append(e)
            big += int((e > 2).sum())
            zero_hit += int(((d_est == 0) & (d_true[0].numpy() > 0)).sum())
        e = np.concatenate(errs)
        print(f"  {tv:8s} {name:24s} среднее {e.mean():.2f}  медиана {np.median(e):.1f}  "
              f"90% {np.percentile(e, 90):.1f}  >2 кадров {100 * big / len(e):.1f}%  "
              f"потерянных символов {zero_hit}")


# ---------------------------------------------------------------- space
def norm_text(s):
    s = s.lower().replace("ё", "е").replace("+", "")
    return re.sub(r"\s+", " ", re.sub(r"[^а-я ]", " ", s)).strip()


def cer(ref, hyp):
    r, h = norm_text(ref), norm_text(hyp)
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[-1] / max(1, len(r))


def cmd_space(a):
    import sys
    import warnings
    import torchaudio
    warnings.filterwarnings("ignore")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from asr_scan import transcribe_pcm
    model, pk, tm = load()
    torch.set_grad_enabled(False)
    tabs = [tm.tacotron.speaker_embedding.weight,
            tm.dur_predictor.dur_pred.speaker_embedding.weight,
            tm.pitch_predictor.pitch_pred.speaker_embedding.weight]
    orig = [t.clone() for t in tabs]
    slot = 4                                    # строка xenia — место для чужого голоса
    texts = phrases(20)[:a.n]
    odir = os.path.join(a.out, "space")
    os.makedirs(odir, exist_ok=True)
    stock_x = model.apply_tts(text=texts[0], speaker="xenia", sample_rate=SR)

    def set_voice(vecs):
        for t, v in zip(tabs, vecs):
            t[slot] = v

    def run(name, vecs=None, speaker="xenia"):
        if vecs is not None:
            set_voice(vecs)
        cs, f0s = [], []
        for i, t in enumerate(texts):
            w = model.apply_tts(text=t, speaker=speaker, sample_rate=SR)
            x16 = torchaudio.functional.resample(w, SR, 16000).numpy()
            pcm = np.clip(np.round(x16 * 32767), -32768, 32767).astype("<i2").tobytes()
            hyp = transcribe_pcm(pcm, a.asr_host, a.asr_port)
            cs.append(cer(t, hyp))
            if i < 3:
                write16(os.path.join(odir, f"{name}_{i}.wav"), w.numpy())
            if i < 4:
                f = f0_track(w.numpy(), len(w) // HOP)
                f0s.append(np.nanmedian(f))
        for t, o in zip(tabs, orig):
            t.copy_(o)
        print(f"  {name:28s} CER {100 * np.mean(cs):5.1f}%  (макс {100 * np.max(cs):5.1f}%)  "
              f"F0 {np.nanmedian(f0s):5.0f} Гц", flush=True)
        return np.mean(cs)

    print(f"== фраз {len(texts)}; CER gigaam, WAV — {odir}")
    print("-- стоковые голоса")
    for v in VOICES:
        run(v, speaker=v)
    ids = {v: i for i, v in enumerate(VOICES)}
    print("-- интерполяция (все три таблицы), t — доля второго голоса")
    for va, vb in (("aidar", "baya"), ("kseniya", "eugene"), ("aidar", "xenia")):
        for t in (0.25, 0.5, 0.75):
            run(f"{va}-{vb} t={t}", [(1 - t) * o[ids[va]] + t * o[ids[vb]] for o in orig])
    print("-- экстраполяция")
    for va, vb, t in (("aidar", "baya", 1.5), ("aidar", "baya", -0.5), ("kseniya", "eugene", 1.5)):
        run(f"{va}-{vb} t={t}", [(1 - t) * o[ids[va]] + t * o[ids[vb]] for o in orig])
    print("-- среднее пяти и случайные выпуклые смеси")
    run("среднее пяти", [o.mean(0) for o in orig])
    g = torch.Generator().manual_seed(5)
    for k in range(3):
        wts = torch.distributions.Dirichlet(torch.ones(5)).sample()
        run(f"смесь {k} [{','.join(f'{x:.2f}' for x in wts.tolist())}]",
            [(wts[:, None] * o).sum(0) for o in orig])
    print("-- случайные векторы с нормой как у стоковых (вне оболочки пяти голосов)")
    for k in range(4):
        vecs = []
        for o in orig:
            r = torch.randn(o.shape[1], generator=g)
            vecs.append(r / r.norm() * o.norm(dim=1).mean())
        run(f"случайный {k}", vecs)

    # поставка своего голоса: строка в таблице -> jit.save -> jit.load -> тот же звук
    print("\n== сохранение голоса")
    vecs = [0.5 * o[0] + 0.5 * o[1] for o in orig]
    set_voice(vecs)
    seq, sp = encode(pk, texts[0], "xenia")
    mask, dur, pitch = predict(tm, seq, sp)
    _, w1 = synth(tm, seq, sp, mask, dur, pitch)
    path = os.path.join(a.out, "voice_test.jit")
    torch.jit.save(tm, path)
    tm2 = torch.jit.load(path, map_location="cpu").eval()
    mask2, dur2, pitch2 = predict(tm2, seq, sp)
    _, w2 = synth(tm2, seq, sp, mask2, dur2, pitch2)
    print(f"  jit.save/load: звук совпадает {torch.equal(w1, w2)}; файл "
          f"{os.path.getsize(path) / 1e6:.1f} МБ")
    voice = torch.stack([vecs[0], torch.cat([vecs[1], vecs[2]])])      # формат load_random_voice
    vpath = os.path.join(a.out, "voice_test.pt")
    torch.save(voice, vpath)
    print(f"  голос отдельным файлом [2,128] как у load_random_voice: {os.path.getsize(vpath)} байт")
    for t, o in zip(tabs, orig):
        t.copy_(o)
    os.remove(path)
    back = model.apply_tts(text=texts[0], speaker="xenia", sample_rate=SR)
    print(f"  стоковая xenia восстановлена: {torch.equal(back, stock_x)}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["src", "mel", "pitch", "align", "space"])
    ap.add_argument("--out", default="/w/probe2")
    ap.add_argument("--n", type=int, default=60, help="фраз на голос")
    ap.add_argument("--asr-host", default="127.0.0.1")
    ap.add_argument("--asr-port", type=int, default=10407, help="wyoming gigaam (asr_scan.py)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
