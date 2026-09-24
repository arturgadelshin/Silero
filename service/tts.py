#!/usr/bin/env python3
"""tts.py — всегда прогретый TTS своим голосом: HTTP и wyoming.

Всё дорогое делается один раз на старте: модель (2 с), разметчик ударений
(silero-stress 4 с, RUAccent 26 с) и прогрев jit (первые два синтеза идут по
секунде, третий уже 24 мс). Дальше на запрос остаётся разметка (6 мс) и
синтез (RTF 0.05), то есть фраза на три секунды готова примерно за 0.15 с.

Два входа в один и тот же движок:
  HTTP      POST /say (тело — текст или JSON) и GET /say?text=… -> WAV;
            GET /health, GET /voices
  wyoming   тот же протокол, что у wyoming-piper: describe -> info,
            synthesize -> audio-start, audio-chunk…, audio-stop. Прописывается
            в Home Assistant как ещё один TTS.

Синтез однопоточный (torch.set_num_threads в этом образе не действует), но
потоки Python работают параллельно: тяжёлые куски внутри torch отпускают GIL.
Одновременных синтезов не больше `--workers`, остальные ждут в очереди.
Одинаковый текст в один момент считается один раз: см. Engine.say.

  docker run -d --restart unless-stopped --name silero_tts --network host \
    -v ~/linkplay/tmp/silero/hub:/root/.cache/torch/hub \
    -v ~/linkplay/tmp/silero/ruaccent:/ruaccent \
    -v ~/linkplay/silero:/app -v ~/linkplay/tmp/silero:/w -w /app \
    silero_ru python -m service.tts
"""
import argparse
import json
import os
import re
import socketserver
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch

from voice.audio import pcm_bytes, resample, wav_from_pcm
from voice.model import (SR, cap_pauses, encode, load, load_state, predict,
                         synth)
from voice.paths import data
from voice.text import (accentor, outside, prepare, split_text,
                        utf8_or_cp1251)

VERSION = "1.0"


class Job:
    """идущий прямо сейчас синтез одной фразы: на нём ждут все, кто попросил
    то же самое, пока он не доложит результат (res) или ошибку (err)"""
    __slots__ = ("done", "res", "err")

    def __init__(self):
        self.done = threading.Event()
        self.res = None
        self.err = None


class Engine:
    def __init__(self, a):
        self.SR = SR
        t0 = time.time()
        _, self.pk, self.tm = load()
        torch.set_grad_enabled(False)
        self.tm = load_state(self.tm, a.model, a.row, "cpu")
        self.alphabet = set(self.pk.symbols[3:])
        t_model = time.time() - t0

        self.acc = None
        t_acc = 0.0
        if not a.no_accent:
            t0 = time.time()
            self.acc = accentor(a.accent_engine, a.accent_model, a.ruaccent_dir)
            t_acc = time.time() - t0

        self.a = a
        self.sem = threading.Semaphore(a.workers)
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.inflight = {}
        self.stat = {"requests": 0, "speech_sec": 0.0, "synth_sec": 0.0,
                     "cache_hits": 0, "started": time.time()}
        t0 = time.time()
        # прогрев обязательно в отдельном потоке: оптимизация jit срабатывает на
        # третьем прогоне и не переносится из главного потока — иначе первый
        # настоящий запрос платит 1.3 с вместо 0.17
        # разметку греем тем же вызовом: первый вызов разметчика дороже
        # остальных (модель омографов), и это тоже секунда на первом запросе
        def warm():
            for k in range(3):
                self.say(f"Прогрев голоса, замок и замок, попытка {k + 1}.",
                         a.speaker, sr=self.SR)
        w = threading.Thread(target=warm)
        w.start()
        w.join()
        self.stat.update(requests=0, speech_sec=0.0, synth_sec=0.0, cache_hits=0)
        print(f"готов: модель {t_model:.1f} с, {a.accent_engine} {t_acc:.1f} с, "
              f"прогрев {time.time() - t0:.1f} с; голос {a.speaker}, "
              f"одновременно {a.workers}", flush=True)

    # ---------------------------------------------------------------- текст
    def mark(self, s, accent=True, numbers=True):
        s = prepare(s, self.acc if accent else None, numbers)
        return s, outside(s, self.alphabet)

    # ---------------------------------------------------------------- звук
    def _one(self, text, speaker, pause):
        # режим grad в torch потоко-локальный, а запросы приходят в новых
        # потоках: без no_grad здесь тензоры требуют градиент и numpy() падает
        with torch.no_grad():
            seq, sp = encode(self.pk, text, speaker)
            mask, dur, pitch = predict(self.tm, seq, sp)
            dur = cap_pauses(self.tm, seq, dur, pause)
            _, wav = synth(self.tm, seq, sp, mask, dur, pitch)
            return wav.numpy()

    def _render(self, text, speaker, accent, numbers, sr, gap, pause):
        t0 = time.time()
        parts, marks, lost = [], [], []
        for i, s in enumerate(split_text(text)):
            m, bad = self.mark(s, accent, numbers)
            marks.append(m)
            lost += bad
            with self.sem:
                w = self._one(m, speaker, pause)
            if i:
                parts.append(np.zeros(int(gap * self.SR), dtype=np.float32))
            parts.append(w)
        x = np.concatenate(parts)
        if sr != self.SR:
            x = resample(x, self.SR, sr)
        return {"pcm": pcm_bytes(x), "sr": sr, "sec": len(x) / sr,
                "text": " ".join(marks), "lost": sorted(set(lost)),
                "synth_sec": time.time() - t0}

    def say(self, text, speaker=None, accent=True, numbers=True, sr=None, gap=0.25,
            pause=None):
        """Готовый ответ из кэша, ожидание чужого синтеза той же фразы — или
        свой синтез.

        Среднее звено — не украшение. Кэш заполняется только после синтеза,
        поэтому рассылка на двести колонок одной фразой (а автоматизации
        оповещают именно так) приходила одним мгновенным залпом, в кэш никто
        не попадал, и сорок с лишним потоков считали одно и то же. Замеры на
        72 ядрах: из 200 одинаковых запросов кэшем отвечали 158 (HTTP) и 165
        (wyoming), остальные синтезировали заново. Теперь первый считает,
        прочие ждут его на Job и забирают тот же самый ответ.
        """
        speaker = speaker or self.a.speaker
        sr = sr or self.a.sr
        pause = self.a.pause_max if pause is None else pause
        key = (text, speaker, accent, numbers, sr, pause)
        with self.lock:
            hit = self.cache.get(key)
            if hit is not None:
                self.cache.move_to_end(key)
                self.stat["cache_hits"] += 1
                self.stat["requests"] += 1
                return hit
            job = self.inflight.get(key)
            mine = job is None
            if mine:
                job = self.inflight[key] = Job()
        if not mine:
            job.done.wait()
            if job.err is not None:                    # считавший упал — падаем с ним,
                raise job.err                          # чтобы не ломиться толпой следом
            with self.lock:
                self.stat["cache_hits"] += 1
                self.stat["requests"] += 1
            return job.res
        try:
            res = self._render(text, speaker, accent, numbers, sr, gap, pause)
        except BaseException as exc:                   # noqa: BLE001
            job.err = exc
            with self.lock:
                self.inflight.pop(key, None)
            job.done.set()
            raise
        with self.lock:
            self.stat["requests"] += 1
            self.stat["speech_sec"] += res["sec"]
            self.stat["synth_sec"] += res["synth_sec"]
            self.cache[key] = res
            self.cache_bytes += len(res["pcm"])
            while self.cache_bytes > self.a.cache_mb * 1e6 and self.cache:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= len(old["pcm"])
            self.inflight.pop(key, None)
        # ответ отдаём ждущим сами: к моменту их пробуждения он может быть уже
        # вытеснен из кэша, если фразы длинные, а --cache-mb маленький
        job.res = res
        job.done.set()
        return res


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    eng = None

    def log_message(self, *args):                      # свой лог, короче стандартного
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _say(self, text, q):
        if not text.strip():
            return self._json({"error": "пустой текст"}, 400)
        if not re.search(r"[а-яёА-ЯЁ]", text):
            return self._json({"error": "в тексте нет русских букв — похоже на "
                                        "битую кодировку; шлите UTF-8",
                               "получено": text[:80]}, 400)
        try:
            r = self.eng.say(text,
                             speaker=(q.get("speaker") or [None])[0],
                             accent=(q.get("accent", ["1"])[0] != "0"),
                             numbers=(q.get("numbers", ["1"])[0] != "0"),
                             sr=int((q.get("sr") or [self.eng.a.sr])[0]),
                             pause=float((q.get("pause") or
                                          [self.eng.a.pause_max])[0]))
        except Exception as exc:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"упал запрос {self.path!r} от {self.client_address[0]}", flush=True)
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        if (q.get("format") or ["wav"])[0] == "json":
            return self._json({k: v for k, v in r.items() if k != "pcm"})
        b = wav_from_pcm(r["pcm"], r["sr"])
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("X-Text", r["text"].encode("utf-8").decode("latin1", "replace"))
        self.send_header("X-Synth-Sec", f"{r['synth_sec']:.3f}")
        self.end_headers()
        self.wfile.write(b)
        print(f"http {r['sec']:.2f} с речи за {r['synth_sec']:.2f} с"
              + (f", ВНЕ АЛФАВИТА {r['lost']}" if r["lost"] else ""), flush=True)

    def _query(self, raw):
        """разбор с терпимостью к cp1251: http.server отдаёт строку запроса
        как latin-1, поэтому возвращаем байты и декодируем сами"""
        out = {}
        for k, v in parse_qs(raw, encoding="latin-1").items():
            out[k] = [utf8_or_cp1251(s.encode("latin-1", "replace")) for s in v]
        return out

    def do_GET(self):
        u = urlparse(self.path)
        q = self._query(u.query)
        if u.path == "/health":
            s = dict(self.eng.stat)
            s["uptime_sec"] = round(time.time() - s.pop("started"), 1)
            s["rtf"] = round(s["synth_sec"] / max(s["speech_sec"], 1e-9), 3)
            s["speaker"] = self.eng.a.speaker
            s["cache_mb"] = round(self.eng.cache_bytes / 1e6, 1)
            return self._json(s)
        if u.path == "/voices":
            return self._json({"speakers": list(self.eng.pk.speakers),
                               "own": self.eng.a.speaker, "sr": self.eng.a.sr})
        if u.path == "/say":
            return self._say((q.get("text") or [""])[0], q)
        self._json({"error": "нет такого пути", "пути": ["/say", "/health", "/voices"]}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/say":
            return self._json({"error": "нет такого пути"}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        body = utf8_or_cp1251(self.rfile.read(n))
        q = self._query(u.query)
        if body.lstrip().startswith("{"):
            try:
                d = json.loads(body)
            except ValueError as exc:
                return self._json({"error": f"JSON: {exc}"}, 400)
            body = d.pop("text", "")
            for k, v in d.items():
                q[k] = [str(v)]
        self._say(body, q)


# ---------------------------------------------------------------- wyoming
class Wyoming(socketserver.StreamRequestHandler):
    eng = None

    def send(self, typ, data=None, payload=None):
        hdr = {"type": typ}
        blob = json.dumps(data or {}, ensure_ascii=False).encode()
        hdr["data_length"] = len(blob)
        if payload is not None:
            hdr["payload_length"] = len(payload)
        self.wfile.write(json.dumps(hdr, ensure_ascii=False).encode() + b"\n")
        self.wfile.write(blob)
        if payload is not None:
            self.wfile.write(payload)
        self.wfile.flush()

    def info(self):
        att = {"name": "silero + f5 clone", "url": "https://github.com/snakers4/silero-models"}
        voice = {"name": self.eng.a.speaker, "description": "свой голос",
                 "attribution": att, "installed": True, "version": VERSION,
                 "languages": ["ru"], "speakers": None}
        return {"asr": [], "wake": [], "handle": [], "intent": [], "mic": [], "snd": [],
                "satellite": None,
                "tts": [{"name": "silero-own", "description": "Silero v5_5_ru, свой голос",
                         "attribution": att, "installed": True, "version": VERSION,
                         "voices": [voice], "supports_synthesize_streaming": False}]}

    def handle(self):
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                hdr = json.loads(line)
            except ValueError:
                return
            data = {}
            n = hdr.get("data_length") or 0
            if n:
                data = json.loads(self.rfile.read(n))
            m = hdr.get("payload_length") or 0
            if m:
                self.rfile.read(m)
            typ = hdr.get("type")
            if typ == "describe":
                self.send("info", self.info())
            elif typ == "synthesize":
                self.synthesize(data)

    def synthesize(self, data):
        text = (data.get("text") or "").strip()
        voice = (data.get("voice") or {}).get("name") or self.eng.a.speaker
        if voice not in self.eng.pk.speakers:
            voice = self.eng.a.speaker
        sr = self.eng.a.wyoming_sr
        t0 = time.time()
        r = self.eng.say(text, speaker=voice, sr=sr)
        self.send("audio-start", {"rate": sr, "width": 2, "channels": 1, "timestamp": 0})
        step = sr * 2                                   # по секунде звука на событие
        pcm = r["pcm"]
        for k in range(0, len(pcm), step):
            self.send("audio-chunk", {"rate": sr, "width": 2, "channels": 1,
                                      "timestamp": k // (sr * 2)}, pcm[k:k + step])
        self.send("audio-stop", {"timestamp": len(pcm) // (sr * 2)})
        print(f"wyoming {r['sec']:.2f} с речи за {time.time() - t0:.2f} с"
              + (f", ВНЕ АЛФАВИТА {r['lost']}" if r["lost"] else ""), flush=True)


class TCP(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=data("train/model_voice.pt"))
    ap.add_argument("--speaker", default="xenia", help="строка обученного голоса")
    ap.add_argument("--row", type=int, default=4)
    ap.add_argument("--sr", type=int, default=48000, help="частота ответа HTTP")
    ap.add_argument("--wyoming-sr", type=int, default=22050)
    ap.add_argument("--http-port", type=int, default=10500)
    ap.add_argument("--wyoming-port", type=int, default=10501)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--workers", type=int, default=4, help="одновременных синтезов")
    ap.add_argument("--cache-mb", type=float, default=200.0)
    ap.add_argument("--pause-max", type=float, default=0.5,
                    help="потолок паузы на точке внутри фразы, с (0 — как у модели)")
    ap.add_argument("--ruaccent-dir", default="/ruaccent")
    ap.add_argument("--accent-engine", default="silero", choices=["silero", "ruaccent"],
                    help="разметчик ударений: silero-stress или RUAccent")
    ap.add_argument("--accent-model", default="turbo3.1", help="модель омографов RUAccent")
    ap.add_argument("--no-accent", action="store_true", help="без RUAccent (старт за 2 с)")
    a = ap.parse_args()

    eng = Engine(a)
    Handler.eng = eng
    Wyoming.eng = eng
    tcp = TCP((a.host, a.wyoming_port), Wyoming)
    threading.Thread(target=tcp.serve_forever, daemon=True).start()
    http = ThreadingHTTPServer((a.host, a.http_port), Handler)
    print(f"http://{a.host}:{a.http_port}/say  wyoming {a.host}:{a.wyoming_port}", flush=True)
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
