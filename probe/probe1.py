#!/usr/bin/env python3
"""probe1.py — вскрытие Silero TTS v5_5_ru: что внутри и учится ли (SILERO.md).

Только чтение: модель грузится из torch hub cache, ничего не сохраняется.
Запуск (сервер, CPU, чтобы не мешать обучению на GPU):
  docker run --rm -v ~/linkplay/tmp/silero/hub:/root/.cache/torch/hub \
    -v ~/linkplay/silero:/app -v ~/linkplay/tmp/silero:/w -w /app \
    silero_ru python -m probe.probe1 --out /w/probe
(кэш hub — копия из kws_lab: docker cp kws_lab:/root/.cache/torch/hub ~/linkplay/tmp/silero/)

Печатает: говорящих и их id, алфавит, мел/STFT-параметры, дерево подмодулей
с числом параметров, таблицы эмбеддингов голоса; затем проверяет градиент:
  1) обычный путь (длительности и pitch предсказаны) — мел и звук;
  2) учительский путь (gt_durs/gt_pitch подставлены) — доходит ли .grad до
     эмбеддингов голоса и до весов декодера через мел и через вокодер.
"""
import argparse
import os
import time

import numpy as np
import torch

from voice.audio import write16
from voice.model import HUB
from voice.paths import data


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", default="v5_5_ru")
    ap.add_argument("--voice", default="aidar")
    ap.add_argument("--text", default="Таймер на пять минут поставлен.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(4)

    model, _ = torch.hub.load(HUB, "silero_tts", language="ru", speaker=a.speaker)
    print(f"== обёртка: {type(model).__name__}, пакетов {len(model.packages)}")
    for pi, pk in enumerate(model.packages):
        print(f"  пакет {pi}: {type(pk).__name__}, jit-моделей {len(pk.models)}")
        print(f"    speaker_to_ids: {pk.speaker_to_ids}")
        print(f"    символов {len(pk.symbols)}: {pk.symbols!r}")
        print(f"    phons={pk.phons} window={pk.window} emb_dim={pk.emb_dim}"
              f" accentor={type(pk.accentor).__name__ if pk.accentor else None}")

    pk = model.packages[0]
    tts = pk.models[0]
    print(f"\n== jit-модель: {tts.original_name}; атрибуты верхнего уровня:")
    for name, sub in tts.named_children():
        print(f"  {name:16s} {sub.original_name:28s} {nparams(sub)/1e6:8.3f} M")
    print(f"  ВСЕГО {nparams(tts)/1e6:.3f} M параметров, "
          f"буферов {sum(b.numel() for b in tts.buffers())}")
    print("  requires_grad у параметров:", {p.requires_grad for p in tts.parameters()})

    tm = tts.tts_model if hasattr(tts, "tts_model") else tts
    print(f"\n== внутри {tm.original_name}:")
    for name, sub in tm.named_children():
        print(f"  {name:16s} {sub.original_name:28s} {nparams(sub)/1e6:8.3f} M")
        for n2, s2 in sub.named_children():
            print(f"      {n2:22s} {s2.original_name:26s} {nparams(s2)/1e6:8.3f} M")
    for attr in ("punkt_symbols", "point_id", "comma_ids", "mean_std_coef", "sil_value"):
        if hasattr(tm, attr):
            print(f"  {attr} = {getattr(tm, attr)}")
    tac = tm.tacotron
    for attr in ("n_mel_channels", "encoder_type", "pitch_strength", "energy_strength",
                 "pitch_emb_type", "padding_value", "log_dur"):
        print(f"  tacotron.{attr} = {getattr(tac, attr)}")
    ist = tm.vocoder.head.istft
    print(f"  vocoder ISTFT: n_fft={ist.n_fft} hop={ist.hop_length} win={ist.win_length}"
          f" padding={ist.padding}; вход вокодера {tm.vocoder.backbone.embed.weight.shape}")

    print("\n== эмбеддинги голоса (строка = говорящий):")
    embs = {
        "mel   tacotron.speaker_embedding": tac.speaker_embedding.weight,
        "dur   dur_pred.speaker_embedding": tm.dur_predictor.dur_pred.speaker_embedding.weight,
        "pitch pitch_pred.speaker_embedding": tm.pitch_predictor.pitch_pred.speaker_embedding.weight,
    }
    for k, w in embs.items():
        norms = ", ".join(f"{v:.2f}" for v in w.norm(dim=1).tolist())
        print(f"  {k}: {tuple(w.shape)}  нормы строк [{norms}]")
    w = tac.speaker_embedding.weight
    cs = torch.nn.functional.cosine_similarity(w[:, None], w[None], dim=-1)
    print("  косинус мел-эмбеддингов:\n" + "\n".join(
        "    " + " ".join(f"{v:5.2f}" for v in row) for row in cs.tolist()))

    # ---- обычный синтез: эталон до любых манипуляций
    torch.set_grad_enabled(False)
    t0 = time.time()
    ref = model.apply_tts(text=a.text, speaker=a.voice, sample_rate=48000)
    print(f"\n== синтез «{a.text}» ({a.voice}): {len(ref)/48000:.2f} с звука "
          f"за {time.time()-t0:.2f} с CPU")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        write16(os.path.join(a.out, f"stock_{a.voice}.wav"), ref.numpy(), 48000)

    # ---- вход модели так же, как его строит apply_tts
    sp_ids, _ = pk.get_speakers(a.voice)
    sentences, _, breaks, rates, pitches, sp = pk.prepare_tts_model_input(
        a.text, ssml=False, speaker_ids=sp_ids, lang=None)
    seq, symb_durs, durs_rate, pitch_coefs = pk.merge_batch_model(sentences, breaks, rates, pitches)
    print(f"  токенов {seq.shape[1]}: "
          + "".join(pk.symbols[i] if i < len(pk.symbols) else "?" for i in seq[0].tolist()))

    mask = torch.zeros(1, seq.shape[1], dtype=torch.bool)
    log_dur = tm.dur_predictor(seq, sp, mask, 1.0, None)
    dur = torch.round(torch.clamp(torch.exp(log_dur) - 1, min=0))
    pitch = tm.pitch_predictor(seq, sp, mask, None, None)
    print(f"  dur_predictor -> {tuple(log_dur.shape)}, сумма кадров {int(dur.sum())} "
          f"(= {dur.sum().item()*pk.window:.2f} с при окне {pk.window})")
    print(f"  pitch_predictor -> {tuple(pitch.shape)}, "
          f"min/mean/max {pitch.min():.2f}/{pitch.mean():.2f}/{pitch.max():.2f}")
    mel = tac(seq, sp, mask, dur, pitch, None)
    print(f"  tacotron -> мел {tuple(mel.shape)}  min/mean/max "
          f"{mel.min():.2f}/{mel.mean():.2f}/{mel.max():.2f}")
    wav = tm.vocoder(mel, 48000, 0.0, True)
    print(f"  vocoder -> {tuple(wav.shape)} = {wav.shape[-1]/48000:.2f} с; "
          f"сэмплов на кадр {wav.shape[-1]/mel.shape[-1]:.1f}")
    mel0, wav0 = mel.clone(), wav.clone()
    out, dur_hat = tm(seq, sp, 48000, symb_durs, durs_rate, pitch_coefs)
    print(f"  forward(gt нет) -> звук {tuple(out.shape)}; dur_hat {tuple(dur_hat.shape)}")
    out2, _ = tm(seq, sp, 48000, None, None, None, dur_hat.clone(), pitch.clone())
    d = (out2 - out).abs().max().item() if out2.shape == out.shape else float("nan")
    print(f"  forward(gt_durs=dur_hat, gt_pitch=pitch) -> {tuple(out2.shape)}, "
          f"макс. отличие от обычного {d:.4f}")

    # ---- проверка градиента: учительский путь
    print("\n== проверка градиента")
    torch.set_grad_enabled(True)
    tm.train(False)
    targets = {
        "mel-эмбеддинг": tac.speaker_embedding.weight,
        "декодер (lin)": tac.lin.weight,
        "энкодер (1-й параметр)": next(tac.encoder.parameters()),
        "вокодер (embed)": tm.vocoder.backbone.embed.weight,
        "dur-эмбеддинг": tm.dur_predictor.dur_pred.speaker_embedding.weight,
        "pitch-эмбеддинг": tm.pitch_predictor.pitch_pred.speaker_embedding.weight,
    }
    for p in targets.values():
        p.requires_grad_(True)
        p.grad = None

    def report(tag):
        for k, p in targets.items():
            g = p.grad
            if g is None:
                print(f"  [{tag}] {k:24s} grad=None")
            else:
                nz = int((g.abs().sum(dim=-1) > 0).sum()) if g.dim() == 2 else -1
                print(f"  [{tag}] {k:24s} |grad|={g.norm():.3e}"
                      + (f"  ненулевых строк {nz}/{g.shape[0]}" if "эмбеддинг" in k else ""))
            p.grad = None

    try:
        mel = tac(seq, sp, mask, dur.detach(), pitch.detach(), None)
        (mel ** 2).mean().backward()
        report("мел")
    except Exception as e:                            # noqa: BLE001
        print(f"  [мел] ОШИБКА backward: {type(e).__name__}: {str(e)[:300]}")
    try:
        mel = tac(seq, sp, mask, dur.detach(), pitch.detach(), None)
        wav = tm.vocoder(mel, 48000, 0.0, True)
        spec = torch.stft(wav.reshape(1, -1), 2048, 600, window=torch.hann_window(2048),
                          return_complex=True).abs()
        torch.log(spec + 1e-5).mean().backward()
        report("звук")
    except Exception as e:                            # noqa: BLE001
        print(f"  [звук] ОШИБКА backward: {type(e).__name__}: {str(e)[:300]}")
    try:
        ld = tm.dur_predictor(seq, sp, mask, 1.0, None)
        pp = tm.pitch_predictor(seq, sp, mask, None, None)
        (ld.mean() + pp.mean()).backward()
        report("dur+pitch")
    except Exception as e:                            # noqa: BLE001
        print(f"  [dur+pitch] ОШИБКА backward: {type(e).__name__}: {str(e)[:300]}")
    for p in targets.values():
        p.requires_grad_(False)

    # ---- какой мел считает модель: подбор преобразования звук -> мел, при
    # котором мел от выхода вокодера совпадает с мелом tacotron. Без этого
    # из своих записей не посчитать учительский мел.
    import torchaudio
    print("\n== подбор мел-преобразования (n_fft 2400, hop 600, 192 полосы, ln, пол 1e-5)")
    res = []
    for fmin in (0, 20, 40, 50, 80):
        for fmax in (8000, 11025, 12000, 16000, 20000, 24000):
            for scale, norm in (("slaney", "slaney"), ("slaney", None), ("htk", None), ("htk", "slaney")):
                for power in (1.0, 2.0):
                    for center in (True, False):
                        tf = torchaudio.transforms.MelSpectrogram(
                            48000, n_fft=2400, win_length=2400, hop_length=600, f_min=fmin,
                            f_max=fmax, n_mels=192, power=power, norm=norm, mel_scale=scale,
                            center=center, pad_mode="reflect")
                        x = wav0.reshape(1, -1)
                        if not center:
                            x = torch.nn.functional.pad(x[None], (900, 900), mode="reflect")[0]
                        m = torch.log(torch.clamp(tf(x), min=1e-5))[0]
                        T = min(m.shape[-1], mel0.shape[-1])
                        dm = m[:, :T] - mel0[0, :, :T]
                        off = dm.mean().item()
                        res.append(((dm - off).abs().mean().item(), dm.abs().mean().item(), off,
                                    fmin, fmax, scale, norm, power, center))
    res.sort(key=lambda r: r[0])
    for r in res[:6]:
        print(f"  |d|={r[0]:.3f} (без сдвига {r[1]:.3f}, сдвиг {r[2]:+.2f})  fmin={r[3]} fmax={r[4]}"
              f" scale={r[5]} norm={r[6]} power={r[7]} center={r[8]}")
    print(f"  худший: |d|={res[-1][0]:.3f}; разброс самого мела tacotron (std) {mel0.std():.3f}")

    # ---- модель не тронута: повторный синтез совпадает с эталоном
    torch.set_grad_enabled(False)
    again = model.apply_tts(text=a.text, speaker=a.voice, sample_rate=48000)
    same = again.shape == ref.shape and torch.equal(again, ref)
    print(f"\n== повторный синтез совпадает с эталоном: {same}"
          + ("" if same else f" (макс. отличие {(again-ref).abs().max():.2e})"
             if again.shape == ref.shape else f" (длины {len(again)} vs {len(ref)})"))


if __name__ == "__main__":
    main()
