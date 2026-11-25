#!/usr/bin/env python3
"""tts.py — всегда прогретый TTS своим голосом: HTTP и wyoming.

Всё дорогое делается один раз на старте: модель (2 с), разметчик ударений
(silero-stress 4 с, RUAccent 26 с) и прогрев jit (первые два синтеза идут по
секунде, третий уже 24 мс). Дальше на запрос остаётся разметка (6 мс) и
синтез (RTF 0.05), то есть фраза на три секунды готова примерно за 0.15 с.

Лишнее из хаб-пакета отдаётся сразу после загрузки голосов — встроенный
акцентор и базовая модель, если все голоса целые jit (на слабом хосте это
минус 0.2 ГБ RSS). Серверы поднимаются раньше прогрева: сервис начинает
отвечать, пока греется джит.

Голосов может быть несколько: `--model` повторяется, и каждая модель получает
свою подпись (`--model voices/artur/model_voice.pt --model voices/stalker/…`
или `подпись=путь`). В Home Assistant они приходят одной сущностью TTS, но
разными голосами в списке — переключение прямо в ассистенте. Стоковые голоса
берутся из первой модели; они в ней уже не стоковые, общие слои ступени 2
учатся для всех строк сразу (см. SILERO.md).

Два входа в один и тот же движок:
  HTTP      POST /say (тело — текст или JSON) и GET /say?text=… -> WAV;
            GET /health, GET /voices
  wyoming   тот же протокол, что у wyoming-piper: describe -> info,
            synthesize -> audio-start, audio-chunk…, audio-stop — по
            предложениям, как piper: первый кусок уходит, пока считается
            второй. Прописывается в Home Assistant как ещё один TTS.

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
import ctypes
import gc
import json
import os
import re
import socketserver
import threading
import time
from collections import OrderedDict, namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch

from voice.audio import loudness_normalize, pcm_bytes, resample, wav_from_pcm
from voice.model import (SR, cap_pauses, encode, load, load_state, predict,
                         synth)
from voice.paths import data
from voice.text import (accentor, outside, prepare, split_text,
                        utf8_or_cp1251)
from voice.vocoder_ort import VocoderORT

VERSION = "1.0"


SERVICE = [".", "..", "/", "w", "opt", "tmp", "home", "app", "voice", "voices",
           "models", "model", "train", "silero"]


Pick = namedtuple("Pick", "name label speaker tm")
"""выбранный голос: имя, под которым его просят; подпись обученной модели,
к которой он относится; строка внутри этой модели; сама модель"""


def model_args(items, speaker="xenia"):
    """`--model` принимается несколько раз: `путь` или `подпись=путь`.

    Без подписи её даёт имя каталога весов (`voices/artur` -> «artur»), как и
    было с одним голосом. Подписи обязаны быть разными: именно они попадают в
    Home Assistant как имена голосов.
    """
    out = []
    for it in items:
        label, sep, path = it.partition("=")
        if not sep or len(label) == 1:                  # «C:\...» — это не подпись
            label, path = "", it
        label = label or voice_label(path, speaker)
        if label in [l for l, _ in out]:
            raise SystemExit(f"две модели с одной подписью: {label}")
        out.append((label, path))
    return out


def voice_label(path, speaker):
    """Подпись голоса в Home Assistant.

    Раньше в info() стояло жёсткое «свой голос», и два инстанса — artur дома
    и stalker на другом сервере — в списке HA было не различить. Берём имя из
    пути к весам: `voices/stalker/model_voice.pt` -> «stalker». Если путь
    служебный (`/w/train/model_voice.pt`), брать нечего — остаётся имя строки
    модели, и тогда подпись задают ключом --voice-label.
    """
    parts = os.path.normpath(path).replace("\\", "/").split("/")[:-1]
    for d in reversed(parts):
        if d and d.lower() not in SERVICE:
            return d
    return speaker


def release_memory():
    """Отдать ОС пустую память: gc, затем malloc_trim по всем кучам glibc.
    Освобождённые веса без trim так и висят в куче — на слабом хосте это
    разница в сотни мегабайт RSS. Не glibc (musl) — просто ничего не делает."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


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
        hub = self.tm
        self.models = OrderedDict()
        for label, path in a.models:
            tm = load_state(hub, path, a.row, "cpu")
            if tm is hub and len(a.models) > 1:
                raise SystemExit(f"{path}: это строка голоса [2,128], она ложится "
                                 "в общую модель. Несколькими голосами работают "
                                 "только целые модели ступени 2 (jit)")
            self.models[label] = tm
        self.tm = next(iter(self.models.values()))      # голос по умолчанию
        self.voices = self.voice_map(self.models, self.pk.speakers, a.speaker)
        self.alphabet = set(self.pk.symbols[3:])
        t_model = time.time() - t0

        # Лишнее из хаб-пакета отдаём сразу, до загрузки разметчика: встроенный
        # акцентор (квантованный BERT, ~86 МБ — сервис размечает внешним
        # silero-stress) и базовую модель, когда все голоса пришли целыми jit
        # и сами всё умеют (~120 МБ). Строка-файл [2,128] — не тот случай:
        # она живёт внутри базовой модели, условие это проверяет. gc здесь, а
        # malloc_trim один раз в конце прогрева: ступенчатые освобождения с
        # промежуточными trim отдают вдвое меньше (754 против 663 МБ на замере).
        if all(m is not hub for m in self.models.values()):
            self.pk.accentor = None
            self.pk.models = None
        del hub
        gc.collect()

        # Вокодер через ONNX Runtime (int8-спина, динамическое квантование
        # MatMul — вдвое быстрее джита, CER тот же; см. voice/vocoder_ort.py).
        # vocoder_dyn.onnx ищется рядом с весами голоса: запечён в образ —
        # работает, не запечён или нет onnxruntime — голос остаётся на джите.
        # --vocoder ort требует файл и падает: промах не должен быть тихим.
        self.voc = {}
        mode = getattr(a, "vocoder", "auto")
        if mode != "jit":
            threads = int(os.environ.get("SILERO_THREADS", "8"))
            for label, path in a.models:
                p = os.path.join(os.path.dirname(path), "vocoder_dyn.onnx")
                if not os.path.exists(p):
                    if mode == "ort":
                        raise SystemExit(f"{p}: нет int8-вокодера "
                                         f"(см. проект silero-int8)")
                    continue
                try:
                    self.voc[label] = VocoderORT(p, self.models[label], threads)
                    print(f"{label}: вокодер ORT int8", flush=True)
                except Exception as exc:               # noqa: BLE001
                    if mode == "ort":
                        raise
                    print(f"{label}: int8-вокодер не поднялся "
                          f"({type(exc).__name__}: {exc}), остаюсь на джите",
                          flush=True)

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
        print(f"готов: модели {t_model:.1f} с, {a.accent_engine} {t_acc:.1f} с; "
              f"голос{'а' if len(self.models) > 1 else ''} "
              f"{', '.join(self.models)} (строка {a.speaker}), "
              f"одновременно {a.workers}", flush=True)

    def warm(self):
        """Прогрев джита и разметчика. main() зовёт это в фоне уже после
        старта серверов: сервис отвечает на десятки секунд раньше, а запрос,
        пришедший до конца прогрева, платит неоптимизированный джит.

        Поток обязателен: оптимизация джита срабатывает на третьем прогоне и
        не переносится из главного потока — иначе первый настоящий запрос
        платит 1.3 с вместо 0.17. Разметку греем тем же вызовом: первый вызов
        разметчика дороже остальных (модель омографов), и это тоже секунда на
        первом запросе. Греем каждую модель: оптимизация джита своя у каждой,
        иначе первый запрос вторым голосом платит секунду."""
        t0 = time.time()
        for label in self.models:
            for k in range(3):
                self.say(f"Прогрев голоса {label}, замок и замок, "
                         f"попытка {k + 1}.", label, sr=self.SR)
        self.stat.update(requests=0, speech_sec=0.0, synth_sec=0.0, cache_hits=0)
        release_memory()
        print(f"прогрев {time.time() - t0:.1f} с (сервис уже отвечает)", flush=True)

    @staticmethod
    def voice_map(models, speakers, own_row):
        """Имя голоса -> что синтезировать.

        Обученные голоса зовутся своими подписями (artur, stalker), стоковые —
        строками модели. Имя строки обученного голоса (xenia) оставлено
        рабочим: его мог запомнить настроенный Home Assistant. Стоковые берутся
        из модели голоса по умолчанию — они в ней всё равно уже не стоковые,
        общие слои на ступени 2 учатся для всех строк сразу.
        """
        v = OrderedDict()
        for label, tm in models.items():
            v[label] = Pick(label, label, own_row, tm)
        first = next(iter(v.values()))
        v.setdefault(own_row, first)
        for s in speakers:
            if s != own_row:
                v.setdefault(s, Pick(s, first.label, s, first.tm))
        return v

    def pick(self, name=None):
        return self.voices.get(name) or next(iter(self.voices.values()))

    # ---------------------------------------------------------------- текст
    def mark(self, s, accent=True, numbers=True):
        s = prepare(s, self.acc if accent else None, numbers)
        return s, outside(s, self.alphabet)

    # ---------------------------------------------------------------- звук
    def _one(self, text, v, pause):
        # режим grad в torch потоко-локальный, а запросы приходят в новых
        # потоках: без no_grad здесь тензоры требуют градиент и numpy() падает
        with torch.no_grad():
            seq, sp = encode(self.pk, text, v.speaker)
            mask, dur, pitch = predict(v.tm, seq, sp)
            dur = cap_pauses(v.tm, seq, dur, pause)
            _, wav = synth(v.tm, seq, sp, mask, dur, pitch,
                           self.voc.get(v.label))
            return wav.numpy()

    def _render(self, text, v, accent, numbers, sr, gap, pause, out=None):
        """Куски из split_text считаются по очереди, и каждый готовый сразу
        уходит в out — байты pcm в целевой частоте, с паузой после него.
        Так wyoming-клиент слышит первое предложение, пока считается второе.

        Громкость поэтому нормируется на кусок, а не на весь ответ: иначе
        первый ждал бы всю фразу. Эталон — записи диктора — нормирован так
        же, фраза за фразой (loudness_normalize), и куски одного ответа
        звучат на одном уровне. Возвращается весь ответ: он идёт в кэш,
        ждущим ту же фразу и в HTTP."""
        t0 = time.time()
        chunks = split_text(text)
        pcm, marks, lost = [], [], []
        t_first = None
        for i, s in enumerate(chunks):
            m, bad = self.mark(s, accent, numbers)
            marks.append(m)
            lost += bad
            with self.sem:
                w = self._one(m, v, pause)
            if self.a.loudness is not None:
                # до ресемпла: K-взвешивание посчитано под 48 кГц, а ресемпл
                # после ограничителя уводит пик не дальше его запаса в 0.5 дБ
                w = loudness_normalize(w, self.SR, self.a.loudness, self.a.peak_dbfs)
            if i + 1 < len(chunks):
                w = np.concatenate([w, np.zeros(int(gap * self.SR), dtype=np.float32)])
            if sr != self.SR:
                w = resample(w, self.SR, sr)
            b = pcm_bytes(w)
            pcm.append(b)
            if t_first is None:
                t_first = time.time() - t0
            if out is not None:
                out(b)
        x = b"".join(pcm)
        return {"pcm": x, "sr": sr, "sec": len(x) / 2 / sr,
                "text": " ".join(marks), "lost": sorted(set(lost)),
                "synth_sec": time.time() - t0,
                "first_sec": t_first if len(chunks) > 1 else None}

    def say(self, text, speaker=None, accent=True, numbers=True, sr=None, gap=0.25,
            pause=None, out=None):
        """Готовый ответ из кэша, ожидание чужого синтеза той же фразы — или
        свой синтез.

        Среднее звено — не украшение. Кэш заполняется только после синтеза,
        поэтому рассылка на двести колонок одной фразой (а автоматизации
        оповещают именно так) приходила одним мгновенным залпом, в кэш никто
        не попадал, и сорок с лишним потоков считали одно и то же. Замеры на
        72 ядрах: из 200 одинаковых запросов кэшем отвечали 158 (HTTP) и 165
        (wyoming), остальные синтезировали заново. Теперь первый считает,
        прочие ждут его на Job и забирают тот же самый ответ.

        out — куски своего синтеза уходят в него по мере готовности (см.
        _render); из кэша и с чужого Job ответ приходит целиком — он уже есть.
        """
        v = self.pick(speaker)
        sr = sr or self.a.sr
        pause = self.a.pause_max if pause is None else pause
        # ключ по подписи модели и строке в ней, а не по запрошенному имени:
        # «xenia» и «artur» — один и тот же синтез, кэш у них общий
        key = (text, v.label, v.speaker, accent, numbers, sr, pause)
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
            res = self._render(text, v, accent, numbers, sr, gap, pause, out)
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
            s["voice"] = next(iter(self.eng.models))
            s["voices"] = list(self.eng.models)
            s["cache_mb"] = round(self.eng.cache_bytes / 1e6, 1)
            s["loudness"] = self.eng.a.loudness
            return self._json(s)
        if u.path == "/voices":
            return self._json({"names": list(self.eng.voices),
                               "own": list(self.eng.models),
                               "speakers": list(self.eng.pk.speakers),
                               "voice": next(iter(self.eng.models)),
                               "row": self.eng.a.speaker,
                               "sr": self.eng.a.sr})
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
        """Обученные голоса и четыре стоковых Silero.

        С одним голосом список как был: имя — строка модели (xenia), подпись
        из --voice-label в описании. Менять это нельзя — имя HA возвращает в
        synthesize, и у настроенного HA выбранный голос отвалился бы.

        С несколькими моделями строкой голоса их не различить (во всех обучена
        одна и та же, xenia), поэтому именами становятся подписи: artur,
        stalker. В списке ассистента они и видны, переключение — там же. Имя
        xenia по-прежнему принимается, просто больше не показывается.
        """
        att = {"name": "silero + f5 clone", "url": "https://github.com/snakers4/silero-models"}
        own = list(self.eng.models)
        row = self.eng.a.speaker
        if len(own) > 1:
            mine = [(label, label) for label in own]
        else:
            mine = [(row, own[0])]
        stock = [(s, f"{s}, стоковый Silero")
                 for s in self.eng.pk.speakers if s != row]
        voices = [{"name": name, "description": desc,
                   "attribution": att, "installed": True, "version": VERSION,
                   "languages": ["ru"], "speakers": None}
                  for name, desc in mine + stock]
        return {"asr": [], "wake": [], "handle": [], "intent": [], "mic": [], "snd": [],
                "satellite": None,
                "tts": [{"name": "silero-own",
                         "description": "Silero v5_5_ru, голос"
                                        f"{'а' if len(own) > 1 else ''} "
                                        f"{', '.join(own)}",
                         "attribution": att, "installed": True, "version": VERSION,
                         "voices": voices, "supports_synthesize_streaming": False}]}

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
        """Звук уходит по предложениям, как у wyoming-piper: audio-start —
        с первым готовым куском, дальше каждый, как посчитался, audio-stop в
        конце. Home Assistant начинает играть первое предложение, не дожидаясь
        остальных. Из кэша фраза уже есть и уходит одним залпом."""
        text = (data.get("text") or "").strip()
        voice = (data.get("voice") or {}).get("name")
        sr = self.eng.a.wyoming_sr
        t0 = time.time()
        step = sr * 2                                   # по секунде звука на событие
        fmt = {"rate": sr, "width": 2, "channels": 1}
        pos, started = 0, False                         # ушло звука, байт

        def out(b):
            nonlocal pos, started
            if not started:                             # до этого клиент ничего не ждёт
                self.send("audio-start", dict(fmt, timestamp=0))
                started = True
            for k in range(0, len(b), step):
                self.send("audio-chunk", dict(fmt, timestamp=(pos + k) // step),
                          b[k:k + step])
            pos += len(b)

        r = self.eng.say(text, speaker=voice, sr=sr, out=out)
        pcm = r["pcm"]
        if not started:                                 # из кэша или с чужого Job — целиком
            self.send("audio-start", dict(fmt, timestamp=0))
            for k in range(0, len(pcm), step):
                self.send("audio-chunk", dict(fmt, timestamp=k // step), pcm[k:k + step])
        self.send("audio-stop", {"timestamp": len(pcm) // step})
        # «первый кусок» имеет смысл только у стриминга: из кэша ответ цельный
        first = f", первый кусок через {r['first_sec']:.2f} с" \
            if started and r["first_sec"] else ""
        print(f"wyoming {r['sec']:.2f} с речи за {time.time() - t0:.2f} с" + first
              + (f", ВНЕ АЛФАВИТА {r['lost']}" if r["lost"] else ""), flush=True)


class TCP(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], metavar="[ПОДПИСЬ=]ПУТЬ",
                    help="веса голоса; можно несколько раз — тогда голоса "
                         "переключаются в самом ассистенте")
    ap.add_argument("--speaker", default="xenia", help="строка обученного голоса")
    ap.add_argument("--voice-label", default="", help="подпись единственного "
                    "голоса в HA; по умолчанию имя каталога весов "
                    "(voices/stalker -> stalker)")
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
    ap.add_argument("--loudness", type=float, default=None, metavar="LUFS",
                    help="нормировать каждое предложение ответа до этой "
                         "громкости "
                    "(BS.1770, моно; у живых записей диктора −15.7…−16.0 на "
                    "канал — берите -15.8); без ключа звук уходит как "
                    "синтезировался")
    ap.add_argument("--peak-dbfs", type=float, default=-1.1,
                    help="потолок пиков при --loudness (у эталона −1.1)")
    ap.add_argument("--ruaccent-dir", default="/ruaccent")
    ap.add_argument("--accent-engine", default="silero", choices=["silero", "ruaccent"],
                    help="разметчик ударений: silero-stress или RUAccent")
    ap.add_argument("--accent-model", default="turbo3.1", help="модель омографов RUAccent")
    ap.add_argument("--no-accent", action="store_true", help="без RUAccent (старт за 2 с)")
    ap.add_argument("--vocoder", default="auto", choices=["auto", "ort", "jit"],
                    help="вокодер: auto — ONNX Runtime int8, если рядом с весами "
                         "голоса лежит vocoder_dyn.onnx, иначе джит; ort — "
                         "требовать int8 (нет файла — стартовать откажусь); "
                         "jit — всегда родной")
    a = ap.parse_args()
    a.model = a.model or [data("train/model_voice.pt")]
    a.models = model_args(a.model, speaker=a.speaker)
    if len(a.models) == 1 and a.voice_label:            # --voice-label как раньше
        a.models = [(a.voice_label, a.models[0][1])]
    a.voice_label = a.models[0][0]

    eng = Engine(a)
    Handler.eng = eng
    Wyoming.eng = eng
    tcp = TCP((a.host, a.wyoming_port), Wyoming)
    threading.Thread(target=tcp.serve_forever, daemon=True).start()
    http = ThreadingHTTPServer((a.host, a.http_port), Handler)
    print(f"http://{a.host}:{a.http_port}/say  wyoming {a.host}:{a.wyoming_port}", flush=True)
    # прогрев в фоне: серверы уже отвечают, запрос до конца прогрева платит
    # неоптимизированный джит (~1.3 с вместо 0.17 на длинной фразе)
    threading.Thread(target=eng.warm, daemon=True).start()
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
