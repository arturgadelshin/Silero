"""model.py — сама Silero v5_5_ru: загрузка, кодирование текста, предсказание
длительностей и pitch, синтез, подмена строки голоса.

Разбор устройства модели — в SILERO.md. Коротко: tacotron (кодировщик текста +
декодер мела), vocoder, dur_predictor, pitch_predictor; голос — три строки
таблиц вложений в позиции --row (4 — место xenia).
"""
import os

import torch

HUB = "snakers4/silero-models"
PKG = "/root/.cache/torch/hub/snakers4_silero-models_master/src/silero/model/v5_5_ru.pt"
SR = 48000
HOP = 600
EMB = 128                      # размер строки голоса в tacotron; dur/pitch по 64
VOICES = ["aidar", "baya", "kseniya", "eugene", "xenia"]


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
    mask = torch.zeros(1, seq.shape[1], dtype=torch.bool, device=seq.device)
    ld = tm.dur_predictor(seq, sp, mask, 1.0, None)
    dur = torch.round(torch.clamp(torch.exp(ld) - 1, min=0))
    pitch = tm.pitch_predictor(seq, sp, mask, None, None)
    # как в forward: на пунктуации pitch обнуляется
    for i, c in enumerate(seq[0].tolist()):
        if c in tm.punkt_symbols:
            pitch[0, 0, i] = 0.0
    return mask, dur, pitch


def cap_pauses(tm, seq, dur, max_sec):
    """Потолок паузы на знаках препинания, секунды (0 — не трогать).

    dur_predictor отводит точке 71 кадр — 0.89 с. Между отдельными фразами
    так и надо, но внутри одной фразы (а туда точка попадает, когда короткий
    огрызок приклеен к соседу — см. split_text) это слышно как провал.
    Запятой и прочей мелочи потолок не мешает: им модель даёт 5 кадров.
    Исходный тензор не меняется.
    """
    if not max_sec:
        return dur
    lim = float(max(1, round(max_sec * SR / HOP)))
    dur = dur.clone()
    for i, c in enumerate(seq[0].tolist()):
        if c in tm.punkt_symbols and dur[..., i].item() > lim:
            dur[..., i] = lim
    return dur


def synth(tm, seq, sp, mask, dur, pitch):
    mel = tm.tacotron(seq, sp, mask, dur, pitch, None)
    wav = tm.vocoder(mel, SR, 0.0, True)
    return mel, wav.reshape(-1)


def voice_tensor(tm, row):
    """строки голоса в формате файла Silero: [2,128] = мел | dur+pitch"""
    P = dict(tm.named_parameters())
    mel = P["tacotron.speaker_embedding.weight"][row].detach().cpu()
    dur = P["dur_predictor.dur_pred.speaker_embedding.weight"][row].detach().cpu()
    pit = P["pitch_predictor.pitch_pred.speaker_embedding.weight"][row].detach().cpu()
    return torch.stack([mel, torch.cat([dur, pit])])


def set_voice(tm, row, v):
    P = dict(tm.named_parameters())
    with torch.no_grad():
        P["tacotron.speaker_embedding.weight"][row] = v[0].to(P["tacotron.speaker_embedding.weight"])
        P["dur_predictor.dur_pred.speaker_embedding.weight"][row] = v[1, :EMB // 2]
        P["pitch_predictor.pitch_pred.speaker_embedding.weight"][row] = v[1, EMB // 2:]


def load_state(tm, path, row, dev):
    """файл голоса [2,128] кладём в строку; целую модель ступени 2 грузим как jit"""
    try:
        return torch.jit.load(path, map_location=dev)
    except Exception:                                   # noqa: BLE001
        v = torch.load(path, map_location=dev)
        set_voice(tm, row, v if torch.is_tensor(v) else v["voice"])
        return tm
