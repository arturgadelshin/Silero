#!/usr/bin/env python3
"""bench.py — сколько одновременных запросов тянет один сервис.

Вопрос, ради которого писалось: нужно ли поднимать пять процессов на пять
колонок, как у piper, или хватает одного. Сервис многопоточный (HTTP —
ThreadingHTTPServer, wyoming — ThreadingTCPServer), одновременность ограничена
семафором `--workers`, а внутри torch отпускает GIL, так что потоки правда
считают параллельно.

Меряются оба входа: HTTP (`format=json`, чтобы не гонять WAV по сети — синтез
при этом настоящий) и wyoming, которым ходит Home Assistant. У wyoming свой
оверхед: звук идёт чанками по секунде, и ресемпл 48000 -> 22050 на каждую фразу.

Тексты берутся из `voice.phrases` — те же, что у приёмки и вскрытия. Они
детерминированы, поэтому второй прогон с теми же N попал бы в кэш сервиса;
чтобы этого не было, у каждого прогона свой `--offset`, а в отчёте печатается
число попаданий в кэш за прогон (по /health). Ненулевое — значит замер испорчен.

Ключ `--same` — про рассылку: когда автоматизация оповещает двести колонок
одной фразой, синтез нужен один раз, остальные берут готовое из кэша сервиса.
Это совсем другая нагрузка, чем двести разных текстов, и мерить её надо
отдельно.

  python -m tools.bench --url http://127.0.0.1:10512 --n 1,5,50
  python -m tools.bench --wyoming 127.0.0.1:10513 --n 5,50
  python -m tools.bench --url http://127.0.0.1:10512 --n 200 --same
"""
import argparse
import json
import socket
import statistics
import threading
import time
import urllib.parse
import urllib.request

from voice.phrases import phrases


def http_once(url, text, timeout):
    """один запрос: возвращает задержку клиента, секунды речи и время синтеза"""
    q = urllib.parse.urlencode({"text": text, "format": "json"})
    t0 = time.time()
    with urllib.request.urlopen(f"{url}/say?{q}", timeout=timeout) as r:
        d = json.loads(r.read())
    return time.time() - t0, d["sec"], d["synth_sec"]


def wy_send(sock, typ, data, payload=None):
    hdr = {"type": typ, "data_length": None, "payload_length": None}
    blob = json.dumps(data, ensure_ascii=False).encode()
    hdr["data_length"] = len(blob)
    if payload is None:
        hdr.pop("payload_length")
    sock.sendall(json.dumps(hdr, ensure_ascii=False).encode() + b"\n" + blob)


def wy_once(host, port, text, timeout):
    """синтез по протоколу wyoming: считаем байты звука до audio-stop"""
    t0 = time.time()
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.settimeout(timeout)
        wy_send(s, "synthesize", {"text": text})
        f = s.makefile("rb")
        sr, got = 22050, 0
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("wyoming закрыл соединение до audio-stop")
            hdr = json.loads(line)
            n = hdr.get("data_length") or 0
            data = json.loads(f.read(n)) if n else {}
            m = hdr.get("payload_length") or 0
            if m:
                f.read(m)
                got += m
            if hdr.get("type") == "audio-start":
                sr = data.get("rate", sr)
            elif hdr.get("type") == "audio-stop":
                break
    return time.time() - t0, got / (sr * 2), None


def health(url):
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=10) as r:
            return json.loads(r.read())
    except OSError:
        return None


def run(n, once, texts):
    """n запросов разом; возвращает стену, задержки и секунды речи"""
    lat, sec, err = [], [], []
    lock = threading.Lock()

    def work(t):
        try:
            l, c, _ = once(t)
        except Exception as exc:                        # noqa: BLE001
            with lock:
                err.append(f"{type(exc).__name__}: {exc}")
            return
        with lock:
            lat.append(l)
            sec.append(c)

    thr = [threading.Thread(target=work, args=(t,)) for t in texts]
    t0 = time.time()
    for t in thr:
        t.start()
    for t in thr:
        t.join()
    return time.time() - t0, lat, sec, err


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))] if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:10500", help="HTTP сервиса")
    ap.add_argument("--wyoming", default="", help="host:port вместо HTTP")
    ap.add_argument("--n", default="1,5,50", help="сколько запросов разом, через запятую")
    ap.add_argument("--offset", type=int, default=0, help="сдвиг по списку фраз (мимо кэша)")
    ap.add_argument("--same", action="store_true",
                    help="всем один текст: рассылка на много колонок, синтез один")
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args()

    if a.wyoming:
        host, _, port = a.wyoming.partition(":")
        once = lambda t: wy_once(host, int(port or 10501), t, a.timeout)   # noqa: E731
        where = f"wyoming {a.wyoming}"
    else:
        once = lambda t: http_once(a.url.rstrip("/"), t, a.timeout)        # noqa: E731
        where = a.url

    ns = [int(s) for s in a.n.split(",") if s.strip()]
    pool = phrases(a.offset + sum(ns) + 1)
    print(f"{where}: пачками по {ns}, фразы со сдвига {a.offset}"
          + (", один текст на всех (рассылка)" if a.same else ""), flush=True)
    print(f"{'N':>4} {'стена':>8} {'медиана':>9} {'p95':>8} {'речи':>9} "
          f"{'быстрее реального':>18} {'кэш':>5}", flush=True)

    at = a.offset
    for n in ns:
        # при --same первый запрос считает, остальные 199 ждут его и берут кэш,
        # поэтому фраза для каждой пачки своя: иначе вторая пачка вся из кэша
        texts = [pool[at]] * n if a.same else pool[at:at + n]
        at += 1 if a.same else n
        h0 = health(a.url)
        wall, lat, sec, err = run(n, once, texts)
        h1 = health(a.url)
        hits = (h1["cache_hits"] - h0["cache_hits"]) if h0 and h1 else -1
        speech = sum(sec)
        print(f"{n:>4} {wall:>7.2f}с {statistics.median(lat) if lat else 0:>8.2f}с "
              f"{p95(lat):>7.2f}с {speech:>8.1f}с {speech / max(wall, 1e-9):>17.1f}× "
              f"{hits:>5}", flush=True)
        if err:
            print(f"     ошибок {len(err)}: {err[0]}", flush=True)


if __name__ == "__main__":
    main()
