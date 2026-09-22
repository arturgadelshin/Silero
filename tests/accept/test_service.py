"""Живой сервис: что обещано в SILERO.md, то и отвечает."""
import io
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import wave


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=30) as r:
        return r.status, r.read(), r.headers


def say(url, text, **q):
    q["text"] = text
    return get(url, "/say?" + urllib.parse.urlencode(q, encoding="utf-8"))


def test_health_reports_a_warm_engine(url):
    st, body, _ = get(url, "/health")
    d = json.loads(body)
    # rtf на свежем сервисе ноль: счётчики обнуляются после прогрева
    assert st == 200 and d["speaker"] and d["uptime_sec"] > 0 and d["rtf"] >= 0


def test_say_returns_playable_wav(url):
    st, body, hdr = say(url, "Проверка связи.")
    assert st == 200 and hdr["Content-Type"] == "audio/wav"
    with wave.open(io.BytesIO(body), "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        assert w.getnframes() / w.getframerate() > 0.5


def test_rate_is_honoured(url):
    _, body, _ = say(url, "Проверка частоты.", sr=24000)
    with wave.open(io.BytesIO(body), "rb") as w:
        assert w.getframerate() == 24000


def test_numbers_are_spoken_not_dropped(url):
    _, body, _ = say(url, "Сейчас 15:30.", format="json")
    d = json.loads(body)
    assert "пятнадцать тридцать" in d["text"].replace("+", "")
    assert d["lost"] == []


def test_cache_is_instant_on_repeat(url):
    phrase = f"Проверка кэша {time.time():.0f}."
    say(url, phrase)
    t0 = time.time()
    say(url, phrase)
    assert time.time() - t0 < 0.1


def test_latency_is_under_a_second(url):
    t0 = time.time()
    say(url, f"Короткая фраза для замера {time.time():.0f}.")
    assert time.time() - t0 < 1.0


def test_text_without_cyrillic_is_a_clear_error(url):
    try:
        say(url, "hello world")
        assert False, "ожидали 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400 and "русских" in e.read().decode()


def test_wyoming_speaks_the_protocol(wyoming):
    s = socket.create_connection(wyoming, timeout=30)
    s.sendall(json.dumps({"type": "describe"}).encode() + b"\n")
    rd = s.makefile("rb")
    hdr = json.loads(rd.readline())
    data = json.loads(rd.read(hdr["data_length"]))
    s.close()
    assert hdr["type"] == "info"
    assert data["tts"][0]["voices"]


def test_pause_query_overrides_the_cap(url):
    # короткий хвост клеится к длинной фразе, и точка внутри неё должна
    # укладываться в потолок: без потолка звука заметно больше
    t = "Довольно длинное предложение про всё на свете, чтобы было что клеить. Ага."
    n = [len(say(url, t, pause=p)[1]) for p in ("0", "0.5")]
    assert n[0] > n[1]
