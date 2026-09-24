"""jobs.py — оркестратор обучения: очередь задач, ступени train/* подпроцессами.

Код обучения не меняется: каждая ступень запускается как `python -m train.…`
с путями внутрь каталога модели (/w/web/models/<id>/…), а прогресс web-морда
вытаскивает из stdout — строки ступеней стабильны и печатаются с flush=True.

Режимы:
  probe  ~24 фразы, ступень 1 и jit-поставка голоса — черновой голос за ~3 мин;
  full   весь корпус RUSLAN, ступени 1–2, просодия, приёмка — ~27 мин.
Проба и полный прогон живут в одном каталоге модели: gen_f5 и prep_targets
пропускают готовое по манифестам, поэтому «проба -> полное» доозвучивает хвост.

GPU один, поэтому задачи идут по очереди; отмена убивает текущий подпроцесс.
Состояние задачи лежит в /w/web/jobs/<id>.json и переживает перезапуск
(задача в состоянии running помечается «прервано»).
"""
import collections
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

from voice.paths import data

from web import store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_KEEP = 500
TARGET_PARTS_FULL = 6
TARGET_PARTS_PROBE = 4
PROBE_TRAIN_ROWS = 24
PROBE_VAL_ROWS = 3

PREVIEW_PHRASES = [
    "Привет! Теперь я умею говорить этим голосом.",
    "Замок на двери и зАмок на горе, таймер на пять минут поставлен.",
]

# F0 стоковых голосов, SILERO.md «pitch»: стартовую строку ступени 1 берём
# ближайшую к μ записей — eugene (дефолт train.py, мужской 104 Гц) не даёт
# женскому голосу перебросить высоту: пространство двухкластерное
INIT_VOICES = {"aidar": 128.5, "baya": 247.3, "kseniya": 255.5,
               "eugene": 104.5, "xenia": 215.1}

_jobs = {}                                   # jid -> dict (живые задачи)
_lock = threading.Lock()
_queue = collections.deque()
_worker = None


# ---------------------------------------------------------------- разбор stdout
RE_F5 = re.compile(r"^(\d+)/(\d+) ")
RE_PART_DONE = re.compile(r"^часть (\d+)/(\d+): готово")
RE_PART_N = re.compile(r"^(\d+)/(\d+) готово")
RE_EPOCH = re.compile(r"^эпоха\s+(\d+):.*проверка мел ([\d.]+) dur ([\d.]+) pitch ([\d.]+)")
RE_STATS = re.compile(r"μ ([\d.]+) Гц, σ ([\d.]+) Гц, μ/σ ([\d.]+)")
RE_CHECK = re.compile(r"== свой голос: CER ([\d.]+) % \(макс ([\d.]+)\), F0 медиана (\d+) Гц")


def parse_line(step, line, total=0):
    """одна строка stdout ступени -> (прогресс 0..1 | None, деталь, метрики)"""
    m = RE_F5.match(line)
    if m and step == "f5":
        n, t = int(m.group(1)), int(m.group(2))
        return n / max(t, 1), f"{n}/{t}", None
    m = RE_PART_DONE.match(line)
    if m and step == "targets":
        return 1.0, f"часть {int(m.group(1)) + 1}/{m.group(2)}", None
    m = RE_PART_N.match(line)
    if m and step == "targets":
        n, t = int(m.group(1)), int(m.group(2))
        return n / max(t, 1), f"{n}/{t}", None
    m = RE_EPOCH.match(line)
    if m and step in ("stage1", "stage2"):
        ep = int(m.group(1))
        p = ep / total if total else None
        return p, f"эпоха {ep}, мел {m.group(2)}", {
            "epoch": ep, "mel": float(m.group(2)),
            "dur": float(m.group(3)), "pitch": float(m.group(4))}
    m = RE_STATS.search(line)
    if m and step == "stats":
        mu, sd, msc = (float(x) for x in m.groups())
        return 1.0, f"μ {mu:.0f} Гц, σ {sd:.1f} Гц", {
            "f0_mean": mu, "f0_std": sd, "msc": msc}
    m = RE_CHECK.match(line)
    if m and step == "check":
        return 1.0, "", {"cer": float(m.group(1)) / 100,
                         "cer_max": float(m.group(2)) / 100,
                         "f0": int(m.group(3))}
    return None, None, None


# ---------------------------------------------------------------- план ступеней
def probe_rows(rows, n_train=PROBE_TRAIN_ROWS, n_val=PROBE_VAL_ROWS):
    """первые n_train train-строк и n_val val-строк корпуса, в исходном порядке.

    val-строки в поднаборе обязательны: без них ранняя остановка ступени 1
    срабатывает на 7-й эпохе по нулевому скору и черновой голос выходит
    после одной эпохи.
    """
    want, tr, va = set(), 0, 0
    for r in rows:
        if r[1] == "train" and tr < n_train:
            want.add(r[0]); tr += 1
        elif r[1] == "val" and va < n_val:
            want.add(r[0]); va += 1
    out = [r for r in rows if r[0] in want]
    return out, tr, va


def write_probe_texts(mid):
    src = data("ruslan", "texts.tsv")
    with open(src, encoding="utf-8") as f:
        rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    out, tr, va = probe_rows(rows)
    path = os.path.join(store.model_dir(mid), "probe_texts.tsv")
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write("\t".join(r) + "\n")
    return path, tr, va


def nearest_voice(mu):
    """стоковый голос, ближайший к μ записей по F0"""
    return min(INIT_VOICES, key=lambda k: abs(INIT_VOICES[k] - mu))


def init_voice_for(M):
    """('xenia', 186.0) | ('eugene', None): старт ступени 1 по stats.pt.
    До stats (проба на пустом каталоге) — дефолт train.py."""
    p = os.path.join(M, "targets", "stats.pt")
    if not os.path.exists(p):
        return "eugene", None
    import torch
    st = torch.load(p, map_location="cpu")
    mu = float(st["f0_mean"])
    return nearest_voice(mu), mu


def support_voice(mid):
    """голос-опора DTW в prep_targets: ближайший к F0 записей.

    Дефолт ступени — eugene, его выбирали под artur (104 Гц, ближайший к 105
    Гц записи). Женской записи (~200 Гц) мужская опора шумит границы по MFCC:
    у lilya dur-лосс 0.104 против 0.075 у artur, и ступень 2 портила дикцию.
    F0 берём из meta (замер опоры при загрузке) или из stats прошлой ступени."""
    meta = store.model_meta(mid) or {}
    mu = meta.get("f0_med")
    if not mu:
        p = os.path.join(store.model_dir(mid), "targets", "stats.pt")
        if os.path.exists(p):
            import torch
            mu = float(torch.load(p, map_location="cpu")["f0_mean"])
    return nearest_voice(mu) if mu else "eugene"


def build_steps(mid, mode):
    M = store.model_dir(mid)
    ref = os.path.join(M, "ref", "ref.wav")
    ref_text = os.path.join(M, "ref", "ref_text.txt")
    support = support_voice(mid)

    def cmd(*argv):
        return list(argv)

    if mode == "probe":
        texts = os.path.join(M, "probe_texts.tsv")
        parts = TARGET_PARTS_PROBE
        steps = [
            ("f5", "Клонирование голоса (F5)", 0.35, 0,
             [cmd("-m", "train.gen_f5", "--texts", texts, "--ref", ref,
                  "--ref-text", ref_text, "--out", os.path.join(M, "f5"))]),
            ("targets", "Цели обучения", 0.10, 0,
             [cmd("-m", "train.prep_targets", "--f5", os.path.join(M, "f5"),
                  "--out", os.path.join(M, "targets"), "--voice", support,
                  "--part", f"{i}/{parts}")
              for i in range(parts)]),
            ("stats", "Статистика голоса", 0.05, 0,
             [cmd("-m", "train.prep_targets", "--f5", os.path.join(M, "f5"),
                  "--out", os.path.join(M, "targets"), "--stats")]),
            ("stage1", "Ступень 1: эмбеддинг голоса", 0.40, 30,
             [cmd("-m", "train.train", "--targets", os.path.join(M, "targets"),
                  "--out", os.path.join(M, "train"), "--stage", "1",
                  "--epochs", "30")]),
            ("finalize", "Поставка и проба звука", 0.10, 0, []),
        ]
        return steps

    steps = [
        ("f5", "Клонирование голоса (F5)", 0.42, 0,
         [cmd("-m", "train.gen_f5", "--texts", data("ruslan", "texts.tsv"),
              "--ref", ref, "--ref-text", ref_text, "--out", os.path.join(M, "f5"))]),
        ("targets", "Цели обучения", 0.08, 0,
         [cmd("-m", "train.prep_targets", "--f5", os.path.join(M, "f5"),
              "--out", os.path.join(M, "targets"), "--voice", support,
              "--part", f"{i}/{TARGET_PARTS_FULL}")
          for i in range(TARGET_PARTS_FULL)]),
        ("stats", "Статистика голоса", 0.01, 0,
         [cmd("-m", "train.prep_targets", "--f5", os.path.join(M, "f5"),
              "--out", os.path.join(M, "targets"), "--stats")]),
        ("stage1", "Ступень 1: эмбеддинг голоса", 0.13, 40,
         [cmd("-m", "train.train", "--targets", os.path.join(M, "targets"),
              "--out", os.path.join(M, "train"), "--stage", "1", "--epochs", "40")]),
        ("stage2", "Ступень 2: декодер и предсказатели", 0.22, 60,
         [cmd("-m", "train.train", "--targets", os.path.join(M, "targets"),
              "--out", os.path.join(M, "train"), "--stage", "2",
              "--resume", os.path.join(M, "train", "voice_stage1.pt"),
              "--epochs", "60")]),
        ("bake", "Просодия в веса", 0.03, 0,
         [cmd("-m", "train.bake_prosody", "--model",
              os.path.join(M, "train", "model_stage2.pt"),
              "--out", os.path.join(M, "train", "model_voice.pt"))]),
        ("check", "Приёмка (CER, F0)", 0.08, 0,
         # обязательно cpu: вокодер в jit не умеет комплексный тип на GPU
         # (грабля SILERO.md, «Результат ступени 2») — а контейнер морды
         # видит GPU, и без флага приёмка берет cuda; --out — чтобы wavs
         # приёмки не сваливались в общий /w/train
         [cmd("-m", "train.train", "--check",
              os.path.join(M, "train", "model_voice.pt"),
              "--targets", os.path.join(M, "targets"),
              "--out", os.path.join(M, "train"), "--device", "cpu")]),
        ("finalize", "Регистрация модели", 0.03, 0, []),
    ]
    return steps


# ---------------------------------------------------------------- состояние
def _job_path(jid):
    return os.path.join(_jobs_dir(), f"{jid}.json")


def _jobs_dir():
    p = data("web", "jobs")
    os.makedirs(p, exist_ok=True)
    return p


def _persist(job):
    with _lock:
        d = {k: v for k, v in job.items() if not k.startswith("_")}
        tmp = _job_path(job["id"]) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        os.replace(tmp, _job_path(job["id"]))


def _log(job, line):
    job["log"].append(line)
    if len(job["log"]) > LOG_KEEP:
        del job["log"][:-LOG_KEEP]


def _set_progress(job):
    done = sum(s["weight"] for s in job["steps"] if s["state"] == "done")
    run = next((s for s in job["steps"] if s["state"] == "running"), None)
    job["progress"] = min(1.0, done + (run["weight"] * run["progress"] if run else 0.0))


def public(job):
    """что уходит в API: без служебных полей"""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def init():
    """поднять после старта: задачу, застигнутую перезапуском, пометить"""
    os.makedirs(_jobs_dir(), exist_ok=True)
    for fn in os.listdir(_jobs_dir()):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(_jobs_dir(), fn)
        try:
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
        except (OSError, ValueError):
            continue
        if j.get("state") in ("queued", "running"):
            j["state"] = "error"
            j["error"] = "прервано перезапуском веб-морды"
            j["finished"] = time.time()
            with open(p, "w", encoding="utf-8") as f:
                json.dump(j, f, ensure_ascii=False, indent=1)


def list_jobs():
    out = []
    os.makedirs(_jobs_dir(), exist_ok=True)
    for fn in os.listdir(_jobs_dir()):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(_jobs_dir(), fn), encoding="utf-8") as f:
                out.append(json.load(f))
        except (OSError, ValueError):
            continue
    for j in _jobs.values():
        out = [x for x in out if x["id"] != j["id"]] + [public(j)]
    return sorted(out, key=lambda j: -j.get("created", 0))


def get_job(jid):
    with _lock:
        j = _jobs.get(jid)
        if j:
            return public(j)
    p = _job_path(jid)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return None


def cancel(jid):
    with _lock:
        j = _jobs.get(jid)
        if not j or j["state"] not in ("queued", "running"):
            return False
        j["_cancel"] = True
    return True


# ---------------------------------------------------------------- запуск
def submit(mid, mode):
    meta = store.model_meta(mid)
    if not meta:
        raise ValueError(f"нет модели {mid}")
    with _lock:
        running = [j for j in _jobs.values() if j["state"] in ("queued", "running")]
        if any(j["model_id"] == mid for j in running):
            raise ValueError("у этой модели уже идёт обучение")
    job = {"id": store.new_id(), "model_id": mid, "mode": mode,
           "name": meta["name"], "state": "queued", "created": time.time(),
           "progress": 0.0, "error": None, "log": [], "metrics": {},
           "steps": [{"name": n, "title": t, "weight": w, "total": tot,
                      "state": "pending", "progress": 0.0, "detail": ""}
                     for n, t, w, tot, _ in build_steps(mid, mode)]}
    with _lock:
        _jobs[job["id"]] = job
        _queue.append(job["id"])
        _start_worker()
    _persist(job)
    return public(job)


def _start_worker():
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_work, daemon=True)
        _worker.start()


def _work():
    # no_grad потоко-локален (грабля сервиса в SILERO.md): в этом потоке
    # никто его не выключал, а финализация синтезирует превью
    import torch
    torch.set_grad_enabled(False)
    while True:
        try:
            jid = _queue.popleft()
        except IndexError:
            return
        with _lock:
            job = _jobs.get(jid)
        if not job:
            continue
        if job.get("_cancel"):
            job["state"], job["error"] = "cancelled", "отменено"
            job["finished"] = time.time()
            _persist(job)
            continue
        _run(job)


class Cancelled(Exception):
    pass


def _kill(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _run(job):
    mid, mode = job["model_id"], job["mode"]
    steps = build_steps(mid, mode)
    job["state"], job["started"] = "running", time.time()
    _log(job, f"задача {mode} для «{job['name']}»")
    _persist(job)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
    try:
        for i, (name, title, weight, total, cmds) in enumerate(steps):
            if job.get("_cancel"):
                raise Cancelled()
            st = job["steps"][i]
            assert st["name"] == name
            st["state"] = "running"
            _set_progress(job)
            _log(job, f"[{name}] {title}")
            _persist(job)
            extra = []
            if name == "stage1":
                v, mu = init_voice_for(store.model_dir(mid))
                extra = ["--init-voice", v]
                _log(job, f"стартовая строка голоса: {v}"
                          + (f" (μ записей {mu:.0f} Гц)" if mu else ""))
            for k, cmd in enumerate(cmds):
                argv = cmd + extra
                detail = f"часть {k + 1}/{len(cmds)}" if len(cmds) > 1 else ""
                _log(job, "$ python " + " ".join(argv))
                proc = subprocess.Popen(
                    [sys.executable] + argv, cwd=ROOT, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    start_new_session=True)
                job["_proc"] = proc
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        _log(job, line)
                    p, det, metrics = parse_line(name, line, total)
                    if det is not None:
                        st["detail"] = (detail + " " + det).strip()
                    if p is not None:
                        if name == "targets":        # прогресс по частям
                            st["progress"] = max(st["progress"],
                                                 (k + p) / len(cmds))
                        else:
                            st["progress"] = max(st["progress"], p)
                    if metrics:
                        job["metrics"].update(metrics)
                    _set_progress(job)
                    if job.get("_cancel"):
                        _kill(proc)
                        raise Cancelled()
                rc = proc.wait()
                job.pop("_proc", None)
                if rc != 0:
                    raise RuntimeError(f"ступень {name} упала, код {rc}")
                if job.get("_cancel"):
                    raise Cancelled()
            st["state"], st["progress"] = "done", 1.0
            _set_progress(job)
            _persist(job)
        _finalize(job)
        job["state"], job["progress"] = "done", 1.0
        job["finished"] = time.time()
        _log(job, "готово")
    except Cancelled:
        job["state"] = "cancelled"
        job["error"] = "отменено"
        job["finished"] = time.time()
        _log(job, "отменено пользователем")
    except Exception as exc:                          # noqa: BLE001
        job["state"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["finished"] = time.time()
        _log(job, f"ОШИБКА {job['error']}")
        for st in job["steps"]:
            if st["state"] == "running":
                st["state"] = "error"
    finally:
        proc = job.pop("_proc", None)
        if proc is not None and proc.poll() is None:
            _kill(proc)
        _persist(job)


def _finalize(job):
    """регистрация результата: jit-поставка пробы или метрики полного прогона"""
    mid, mode = job["model_id"], job["mode"]
    M = store.model_dir(mid)
    if mode == "probe":
        _bake_probe(mid)
        store.save_meta(mid, kind="probe", file="train/model_probe.pt",
                        finished_at=time.time())
        _log(job, "голос-черновик собран")
    else:
        store.save_meta(mid, kind="full", file="train/model_voice.pt",
                        metrics={"cer": job["metrics"].get("cer"),
                                 "cer_max": job["metrics"].get("cer_max"),
                                 "f0": job["metrics"].get("f0")},
                        finished_at=time.time())
    m = job["metrics"]
    if "msc" in m:
        store.save_meta(mid, msc=m["msc"], f0_mean=m.get("f0_mean"),
                        f0_std=m.get("f0_std"))
    _preview(mid)
    _log(job, "превью озвучено")


def _bake_probe(mid):
    """файл голоса [2,128] -> целый jit со строкой 4, единый формат для движка.

    Базовая модель общая на процесс (web.engine.base), поэтому строку 4 после
    сохранения возвращаем на место — иначе сломается стоковая xenia у движка.
    """
    import torch
    from voice.model import set_voice, voice_tensor
    from web.engine import base
    _, tm = base()
    M = store.model_dir(mid)
    v = torch.load(os.path.join(M, "train", "voice_stage1.pt"), map_location="cpu")
    stock = voice_tensor(tm, 4)
    set_voice(tm, 4, v if torch.is_tensor(v) else v["voice"])
    torch.jit.save(tm, os.path.join(M, "train", "model_probe.pt"))
    set_voice(tm, 4, stock)


def _preview(mid):
    """пара фраз новым голосом — слушать прямо в списке моделей"""
    import numpy as np
    import torch
    from voice.audio import write16
    from voice.model import SR, encode, load_state, predict, synth
    from voice.text import prepare
    from web.engine import base
    from web.refprep import accent
    meta = store.model_meta(mid)
    pk, tm = base()
    tm = load_state(tm, os.path.join(store.model_dir(mid), meta["file"]), 4, "cpu")
    parts = []
    with torch.no_grad():
        for i, phrase in enumerate(PREVIEW_PHRASES):
            if i:
                parts.append(np.zeros(int(0.25 * SR), dtype=np.float32))
            seq, sp = encode(pk, prepare(phrase, accent(), True), "xenia")
            mask, dur, pitch = predict(tm, seq, sp)
            _, wav = synth(tm, seq, sp, mask, dur, pitch)
            parts.append(wav.numpy())
    write16(os.path.join(store.model_dir(mid), "preview.wav"),
            np.concatenate(parts), SR)
