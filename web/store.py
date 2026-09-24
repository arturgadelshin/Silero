"""store.py — реестр моделей, история генерации и загрузки: только файлы и JSON.

Тяжёлого здесь нет сознательно: слой читает и пишет meta.json / index.json,
а engine и jobs лишь пользуются этими функциями. Поэтому unit-тесты на store
идут без модели и без GPU.

Каталоги (корень задаётся SILERO_DATA, как у всего остального):

    web/models/<id>/   meta.json + ref/ + f5/ + targets/ + train/ + preview.wav
    web/history/       index.json + <имя голоса>_<дата>.wav
    web/uploads/<id>/  исходник загрузки и приготовленная опора ref.wav
"""
import json
import os
import re
import shutil
import threading
import time

from voice.paths import data

LOCK = threading.Lock()


def _dir(*parts):
    p = data("web", *parts)
    os.makedirs(p, exist_ok=True)
    return p


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- модели
def new_id():
    return f"{int(time.time() * 1000):x}{os.urandom(2).hex()}"


def model_dir(mid):
    p = os.path.join(_dir("models"), mid)
    os.makedirs(p, exist_ok=True)
    return p


def model_meta(mid):
    # чтение каталог не создаёт: иначе каждый 404 оставлял бы пустой dir
    return _read_json(os.path.join(_dir("models"), mid, "meta.json"))


def save_meta(mid, **fields):
    with LOCK:
        m = model_meta(mid) or {"id": mid, "name": mid, "kind": "probe",
                                 "created": time.time()}
        m.update(fields)
        _write_json(os.path.join(model_dir(mid), "meta.json"), m)
        return m


def list_models():
    root = _dir("models")
    out = []
    for mid in os.listdir(root):
        if not os.path.isdir(os.path.join(root, mid)):
            continue
        m = model_meta(mid)
        if m:
            out.append(m)
    return sorted(out, key=lambda m: -m.get("created", 0))


def create_model(name, kind):
    """каталог модели с meta.json; имя задают сразу при обучении"""
    name = (name or "").strip()
    if not name:
        raise ValueError("задайте имя модели")
    if len(name) > 60:
        raise ValueError("имя длиннее 60 знаков")
    mid = new_id()
    save_meta(mid, name=name, kind=kind)
    return mid


def rename_model(mid, name):
    if not model_meta(mid):
        return None
    name = (name or "").strip()
    if not name:
        raise ValueError("имя не может быть пустым")
    if len(name) > 60:
        raise ValueError("имя длиннее 60 знаков")
    return save_meta(mid, name=name)


def delete_model(mid):
    with LOCK:
        p = os.path.join(data("web", "models"), mid)
        if os.path.isdir(p):
            shutil.rmtree(p)


# ---------------------------------------------------------------- история
def _safe(s):
    """имя файла из имени голоса: буквы/цифры/дефис, остальное в «_»"""
    s = re.sub(r"[^\w\-]+", "_", (s or "voice").strip(), flags=re.UNICODE)
    return s.strip("_")[:40] or "voice"


def history_index():
    return _read_json(os.path.join(_dir("history"), "index.json")) or []


def add_history(entry, wav):
    """записать wav в историю и вернуть запись с именем файла"""
    ts = time.localtime(entry["ts"])
    base = f"{_safe(entry['voice_name'])}_{time.strftime('%Y%m%d-%H%M%S', ts)}"
    root = _dir("history")
    name, k = base + ".wav", 0
    while os.path.exists(os.path.join(root, name)):
        k += 1
        name = f"{base}_{k}.wav"
    with open(os.path.join(root, name), "wb") as f:
        f.write(wav)
    entry = dict(entry, file=name, id=new_id())
    with LOCK:
        idx = history_index()
        idx.insert(0, entry)
        _write_json(os.path.join(root, "index.json"), idx)
    return entry


def history_path(entry):
    return os.path.join(_dir("history"), entry["file"])


def delete_history(hid):
    with LOCK:
        idx = history_index()
        keep = [e for e in idx if e["id"] != hid]
        for e in idx:
            if e["id"] == hid:
                try:
                    os.remove(history_path(e))
                except OSError:
                    pass
        _write_json(os.path.join(_dir("history"), "index.json"), keep)


# ---------------------------------------------------------------- загрузки
def upload_dir(uid):
    p = os.path.join(_dir("uploads"), uid)
    os.makedirs(p, exist_ok=True)
    return p


def upload_meta(uid):
    return _read_json(os.path.join(_dir("uploads"), uid, "meta.json"))


def save_upload_meta(uid, **fields):
    with LOCK:
        m = upload_meta(uid) or {"id": uid, "created": time.time()}
        m.update(fields)
        _write_json(os.path.join(upload_dir(uid), "meta.json"), m)
        return m
