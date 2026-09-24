"""app.py — HTTP API веб-морды и раздача статики. Запуск: python -m web.app.

Порт по умолчанию 10580 (10500/10501 заняты сервисом, 10510 брал проверочный
прогон). Сервис не трогаем: генерация идёт своим движком (web.engine), а
обучение — подпроцессами train/* (web.jobs).
"""
import argparse
import asyncio
import json
import os
import re
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web import jobs, store
from web.engine import DEFAULTS, STOCK, Engine
from web.refprep import mark

app = FastAPI(title="silero web", docs_url=None, redoc_url=None)
ENG = None


@app.on_event("startup")
def _startup():
    global ENG
    jobs.init()
    ENG = Engine()


def engine() -> Engine:
    if ENG is None:
        raise HTTPException(503, "движок ещё грузится")
    return ENG


def _latin(s):
    """заголовки HTTP — latin-1, как X-Text у сервиса"""
    return s.encode("utf-8").decode("latin1", "replace")


# ---------------------------------------------------------------- модели
@app.get("/api/models")
def models_list():
    out = []
    for m in store.list_models():
        f = m.get("file")
        m = dict(m)
        m["ready"] = bool(f) and os.path.exists(
            os.path.join(store.model_dir(m["id"]), f))
        m["has_preview"] = os.path.exists(
            os.path.join(store.model_dir(m["id"]), "preview.wav"))
        out.append(m)
    return {"models": out, "stock": STOCK, "defaults": DEFAULTS}


class RenameReq(BaseModel):
    name: str


@app.patch("/api/models/{mid}")
def model_rename(mid: str, req: RenameReq):
    try:
        m = store.rename_model(mid, req.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not m:
        raise HTTPException(404, "модель не найдена")
    return m


@app.delete("/api/models/{mid}")
def model_delete(mid: str):
    if not store.model_meta(mid):
        raise HTTPException(404, "модель не найдена")
    for j in jobs.list_jobs():
        if j["model_id"] == mid and j["state"] in ("queued", "running"):
            raise HTTPException(409, "сначала отмените идущее обучение")
    store.delete_model(mid)
    return {"ok": True}


@app.get("/api/models/{mid}/preview")
def model_preview(mid: str):
    p = os.path.join(store.model_dir(mid), "preview.wav")
    if not os.path.exists(p):
        raise HTTPException(404, "превью нет")
    return FileResponse(p, media_type="audio/wav")


# ---------------------------------------------------------------- опора
class MarkReq(BaseModel):
    text: str


@app.post("/api/mark")
def do_mark(req: MarkReq):
    if not req.text.strip():
        raise HTTPException(400, "пустой текст")
    return {"marked": mark(req.text)}


@app.post("/api/ref/prepare")
def ref_prepare(file: UploadFile = File(...), text: str = Form(...)):
    if not (text or "").strip():
        raise HTTPException(400, "введите текст опорной фразы")
    uid = store.new_id()
    d = store.upload_dir(uid)
    ext = os.path.splitext(file.filename or "")[1][:8] or ".bin"
    raw = os.path.join(d, "input" + ext)
    with open(raw, "wb") as f:
        f.write(file.file.read())
    try:
        from web import refprep
        info, y = refprep.prepare_upload(raw, text)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    from voice.audio import write16
    write16(os.path.join(d, "ref.wav"), y, 48000)
    store.save_upload_meta(uid, text=text, **{k: info.get(k) for k in
                                              ("duration", "speech_sec",
                                               "warnings", "marked_text",
                                               "f0_med")})
    return dict(info, id=uid)


@app.get("/api/ref/{uid}/audio")
def ref_audio(uid: str):
    p = os.path.join(store.upload_dir(uid), "ref.wav")
    if not os.path.exists(p):
        raise HTTPException(404, "опора не найдена")
    return FileResponse(p, media_type="audio/wav")


# ---------------------------------------------------------------- обучение
class JobReq(BaseModel):
    name: str = ""
    mode: str
    ref_id: str = ""
    marked_text: str = ""
    model_id: str = ""


@app.post("/api/jobs")
def job_create(req: JobReq):
    if req.mode not in ("probe", "full"):
        raise HTTPException(400, "mode: probe | full")
    if req.model_id:                                 # проба -> полное обучение
        meta = store.model_meta(req.model_id)
        if not meta:
            raise HTTPException(404, "модель не найдена")
        if not os.path.exists(os.path.join(store.model_dir(req.model_id),
                                           "ref", "ref.wav")):
            raise HTTPException(400, "у модели нет опорной записи")
        mid = req.model_id
    else:
        if not req.name.strip():
            raise HTTPException(400, "задайте имя модели")
        if not req.ref_id or not req.marked_text.strip():
            raise HTTPException(400, "сначала приготовьте опору")
        up = store.upload_meta(req.ref_id)
        if not up or not os.path.exists(os.path.join(store.upload_dir(req.ref_id),
                                                     "ref.wav")):
            raise HTTPException(404, "загрузка не найдена, приготовьте опору заново")
        try:
            mid = store.create_model(req.name, "probe" if req.mode == "probe" else "full")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        ref_dir = os.path.join(store.model_dir(mid), "ref")
        os.makedirs(ref_dir, exist_ok=True)
        os.replace(os.path.join(store.upload_dir(req.ref_id), "ref.wav"),
                   os.path.join(ref_dir, "ref.wav"))
        with open(os.path.join(ref_dir, "ref_text.txt"), "w",
                  encoding="utf-8") as f:
            f.write(req.marked_text.strip() + "\n")
        if up.get("f0_med"):
            store.save_meta(mid, f0_med=up["f0_med"])
    if req.mode == "probe":
        jobs.write_probe_texts(mid)
    try:
        return jobs.submit(mid, req.mode)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/jobs")
def jobs_list():
    return {"jobs": jobs.list_jobs()}


@app.get("/api/jobs/{jid}")
def job_get(jid: str):
    j = jobs.get_job(jid)
    if not j:
        raise HTTPException(404, "задача не найдена")
    return j


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str):
    if not jobs.cancel(jid):
        raise HTTPException(409, "задача не идёт")
    return {"ok": True}


@app.get("/api/jobs/{jid}/events")
async def job_events(jid: str):
    async def gen():
        last = None
        while True:
            j = jobs.get_job(jid)
            if j is None:
                yield "event: gone\n\n"
                return
            snap = json.dumps({"state": j["state"], "progress": j["progress"],
                               "steps": j["steps"], "log": j["log"][-6:],
                               "metrics": j.get("metrics", {})},
                              ensure_ascii=False)
            if snap != last:
                last = snap
                yield f"data: {snap}\n\n"
            if j["state"] in ("done", "error", "cancelled"):
                return
            await asyncio.sleep(1.0)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- генерация
class GenReq(BaseModel):
    text: str
    voice: str = "stock:xenia"
    params: dict = {}


@app.post("/api/generate")
def generate(req: GenReq):
    if not req.text.strip():
        raise HTTPException(400, "пустой текст")
    if not re.search(r"[а-яёА-ЯЁ]", req.text):
        raise HTTPException(400, "в тексте нет русских букв")
    try:
        r = engine().generate(req.text, req.voice, req.params)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    voice_name = r["speaker"] if req.voice.startswith("stock:") \
        else (store.model_meta(req.voice) or {}).get("name", req.voice)
    wav = engine().wav(r)
    entry = store.add_history(
        {"ts": time.time(), "text": req.text, "marked": r["text"],
         "voice_id": req.voice, "voice_name": voice_name,
         "params": r["params"], "sec": round(r["sec"], 2), "lost": r["lost"]},
        wav)
    h = {"X-Entry-Id": entry["id"], "X-Voice": _latin(str(voice_name)),
         "X-Marked": _latin(r["text"]), "X-Sec": f"{r['sec']:.2f}",
         "X-Synth-Sec": f"{r['synth_sec']:.3f}",
         "X-Lost": _latin(" ".join(r["lost"]))}
    return Response(content=wav, media_type="audio/wav", headers=h)


@app.get("/api/history")
def history_list(limit: int = 100):
    return {"items": store.history_index()[:max(1, min(limit, 500))]}


@app.get("/api/history/{hid}/audio")
def history_audio(hid: str, dl: int = 0):
    for e in store.history_index():
        if e["id"] == hid:
            p = store.history_path(e)
            if not os.path.exists(p):
                raise HTTPException(404, "файл пропал")
            if dl:
                return FileResponse(p, media_type="audio/wav", filename=e["file"])
            return FileResponse(p, media_type="audio/wav")
    raise HTTPException(404, "нет в истории")


@app.delete("/api/history/{hid}")
def history_delete(hid: str):
    store.delete_history(hid)
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"ok": True, "models": len(store.list_models()),
            "history": len(store.history_index()),
            "uptime": round(time.time() - _T0, 1)}


_T0 = time.time()

app.mount("/", StaticFiles(
    directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"),
    html=True), name="static")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=10580)
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
