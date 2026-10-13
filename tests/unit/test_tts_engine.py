"""Замок на одинаковый текст: синтез один на всю рассылку.

Движок целиком поднимать не нужно (модель — это секунды и 140 МБ весов),
поэтому Engine собирается через __new__, а вместо настоящего синтеза
подставляется счётчик с задержкой: ровно так ведёт себя рассылка на двести
колонок — все двести приходят раньше, чем первый успел положить ответ в кэш.
"""
import threading
import time
import types

import pytest

from service.tts import Engine, Pick


def engine(cache_mb=200.0, voices=("artur",)):
    e = Engine.__new__(Engine)
    e.SR = 48000
    e.a = types.SimpleNamespace(speaker="xenia", sr=48000, pause_max=0.5,
                                cache_mb=cache_mb)
    e.models = __import__("collections").OrderedDict(
        (v, f"модель {v}") for v in voices)
    e.voices = Engine.voice_map(e.models, ["aidar", "baya", "kseniya",
                                           "eugene", "xenia"], "xenia")
    e.lock = threading.Lock()
    e.cache = __import__("collections").OrderedDict()
    e.cache_bytes = 0
    e.inflight = {}
    e.stat = {"requests": 0, "speech_sec": 0.0, "synth_sec": 0.0,
              "cache_hits": 0, "started": time.time()}
    e.calls = []
    return e


def slow_render(e, delay=0.2, pcm=b"\0" * 100):
    def _render(text, v, accent, numbers, sr, gap, pause):
        e.calls.append((text, v.label, v.speaker))
        time.sleep(delay)
        return {"pcm": pcm, "sr": sr, "sec": 1.0, "text": text, "lost": [],
                "synth_sec": delay}
    e._render = _render


def storm(e, n, text="одна фраза на всех"):
    out, err = [], []
    def work():
        try:
            out.append(e.say(text))
        except Exception as exc:                        # noqa: BLE001
            err.append(exc)
    thr = [threading.Thread(target=work) for _ in range(n)]
    for t in thr:
        t.start()
    for t in thr:
        t.join()
    return out, err


def test_one_synthesis_for_a_broadcast():
    e = engine()
    slow_render(e)
    out, err = storm(e, 50)
    assert err == []
    assert len(e.calls) == 1                            # до замка было бы под сорок
    assert e.calls[0] == ("одна фраза на всех", "artur", "xenia")
    assert len(out) == 50 and all(r is out[0] for r in out)
    assert e.stat["requests"] == 50 and e.stat["cache_hits"] == 49
    assert e.stat["speech_sec"] == 1.0                  # речь считаем один раз


def test_different_texts_still_run_in_parallel():
    e = engine()
    slow_render(e, delay=0.2)
    out, err = [], []
    def work(k):
        out.append(e.say(f"фраза номер {k}"))
    thr = [threading.Thread(target=work, args=(k,)) for k in range(5)]
    t0 = time.time()
    for t in thr:
        t.start()
    for t in thr:
        t.join()
    # замок держит только одинаковый ключ: разные тексты идут разом
    assert time.time() - t0 < 0.6
    assert len(e.calls) == 5 and e.stat["cache_hits"] == 0


def test_waiters_get_the_answer_even_if_the_cache_evicted_it():
    # крошечный кэш выбрасывает ответ сразу же, ждущим он идёт напрямую
    e = engine(cache_mb=1e-6)
    slow_render(e)
    out, err = storm(e, 10)
    assert err == [] and len(e.calls) == 1
    assert all(r is out[0] for r in out) and e.cache == {}


def test_failure_does_not_start_a_stampede():
    e = engine()
    def _render(*args):
        e.calls.append(args[0])                         # текст
        time.sleep(0.1)
        raise ValueError("битый текст")
    e._render = _render
    out, err = storm(e, 10)
    assert out == [] and len(err) == 10
    assert all(isinstance(x, ValueError) for x in err)
    assert len(e.calls) == 1                            # повторять за упавшим незачем
    # ключ убран: следующий запрос начнёт заново, а не повиснет на мёртвом Job
    assert e.inflight == {}
    with pytest.raises(ValueError):
        e.say("одна фраза на всех")
    assert len(e.calls) == 2


def test_voice_label_comes_from_the_weights_path():
    from service.tts import voice_label
    # так лежит голос в образе и в репозитории
    assert voice_label("/opt/voices/stalker/model_voice.pt", "xenia") == "stalker"
    assert voice_label("voices/artur/model_voice.pt", "xenia") == "artur"
    # служебный путь: брать нечего, остаётся строка модели
    assert voice_label("/w/train/model_voice.pt", "xenia") == "xenia"
    assert voice_label("model_voice.pt", "xenia") == "xenia"
    # морда кладёт голоса по идентификатору — он и станет подписью
    assert voice_label("/w/web/models/1a0ce5153aa68be/train/model_voice.pt",
                       "xenia") == "1a0ce5153aa68be"


def test_wyoming_info_lists_all_voices_with_own_first():
    from service.tts import Wyoming
    eng = types.SimpleNamespace(
        a=types.SimpleNamespace(speaker="xenia", voice_label="stalker"),
        models={"stalker": "м1"},
        pk=types.SimpleNamespace(speakers=["aidar", "baya", "kseniya",
                                           "eugene", "xenia"]))
    info = Wyoming.info(types.SimpleNamespace(eng=eng))
    voices = info["tts"][0]["voices"]
    # имена — строки модели: их HA возвращает в synthesize, менять нельзя
    assert [v["name"] for v in voices] == ["xenia", "aidar", "baya", "kseniya", "eugene"]
    assert voices[0]["description"] == "stalker"     # свой — первым и по имени
    assert all("стоковый" in v["description"] for v in voices[1:])
    assert "stalker" in info["tts"][0]["description"]


def test_wyoming_info_with_two_voices_lists_labels():
    from service.tts import Wyoming
    import collections
    models = collections.OrderedDict(artur="м1", stalker="м2")
    eng = types.SimpleNamespace(
        a=types.SimpleNamespace(speaker="xenia", voice_label="artur"),
        models=models,
        pk=types.SimpleNamespace(speakers=["aidar", "baya", "kseniya",
                                           "eugene", "xenia"]))
    info = Wyoming.info(types.SimpleNamespace(eng=eng))
    voices = info["tts"][0]["voices"]
    # обученные зовутся подписями: строкой xenia их не различить, она в обеих
    assert [v["name"] for v in voices] == ["artur", "stalker", "aidar", "baya",
                                           "kseniya", "eugene"]
    assert [v["description"] for v in voices[:2]] == ["artur", "stalker"]
    assert "artur, stalker" in info["tts"][0]["description"]


def test_voice_map_resolves_labels_rows_and_unknown_names():
    rows = ["aidar", "baya", "kseniya", "eugene", "xenia"]
    import collections
    v = Engine.voice_map(collections.OrderedDict(artur="м1", stalker="м2"),
                         rows, "xenia")
    assert v["artur"] == Pick("artur", "artur", "xenia", "м1")
    assert v["stalker"] == Pick("stalker", "stalker", "xenia", "м2")
    # имя строки обученного голоса оставлено рабочим: его помнит настроенный HA
    assert v["xenia"] == v["artur"]
    # стоковые — из модели по умолчанию
    assert v["eugene"] == Pick("eugene", "artur", "eugene", "м1")
    e = engine(voices=("artur", "stalker"))
    assert e.pick("stalker").tm == "модель stalker"
    assert e.pick("нет такого") == e.pick("artur")      # незнакомое имя — первый
    assert e.pick(None) == e.pick("artur")


def test_two_names_of_one_voice_share_the_cache():
    e = engine(voices=("artur", "stalker"))
    slow_render(e)
    assert e.say("одна фраза", "xenia") is e.say("одна фраза", "artur")
    assert len(e.calls) == 1
    # а другой голос считается заново
    e.say("одна фраза", "stalker")
    assert [c[1] for c in e.calls] == ["artur", "stalker"]


def test_model_args_takes_paths_and_labels():
    from service.tts import model_args
    assert model_args(["voices/artur/model_voice.pt"]) == [
        ("artur", "voices/artur/model_voice.pt")]
    assert model_args(["/opt/voices/stalker/model_voice.pt",
                       "свой=/w/train/model_voice.pt"]) == [
        ("stalker", "/opt/voices/stalker/model_voice.pt"),
        ("свой", "/w/train/model_voice.pt")]
    # служебный путь без подписи: остаётся строка модели
    assert model_args(["/w/train/model_voice.pt"]) == [
        ("xenia", "/w/train/model_voice.pt")]
    with pytest.raises(SystemExit):
        model_args(["voices/artur/model_voice.pt", "artur=/w/train/model_voice.pt"])
