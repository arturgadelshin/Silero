"""asr.py — распознавание через wyoming-onnxasr (gigaam-v3-rnnt) для приёмки.

Контейнер ASR поднят с --network host, поэтому адрес по умолчанию
127.0.0.1:10407, и говорить с ним можно только с самого сервера: контейнеру
синтеза тоже нужен --network host, иначе connection refused.
"""
import json
import socket
import wave

CHUNK = 16000  # сэмплов за событие (0.5 с при 16 кГц)
HOST = "127.0.0.1"
PORT = 10407


def _send(sock, typ, data=None, payload=None):
    hdr = {"type": typ}
    blob = json.dumps(data or {}, ensure_ascii=False).encode() if data is not None else None
    if blob is not None:
        hdr["data_length"] = len(blob)
    if payload is not None:
        hdr["payload_length"] = len(payload)
    sock.sendall(json.dumps(hdr, ensure_ascii=False).encode() + b"\n")
    if blob is not None:
        sock.sendall(blob)
    if payload is not None:
        sock.sendall(payload)


def _recv_event(rd):
    line = rd.readline()
    if not line:
        return None, None
    hdr = json.loads(line)
    data = {}
    n = hdr.get("data_length") or 0
    if n:
        data = json.loads(rd.read(n))
    m = hdr.get("payload_length") or 0
    if m:
        rd.read(m)
    return hdr.get("type"), data


def transcribe_pcm(pcm, host=HOST, port=PORT, rate=16000, width=2, ch=1, timeout=120.0):
    s = socket.create_connection((host, port), timeout=timeout)
    s.settimeout(timeout)
    try:
        rd = s.makefile("rb")
        _send(s, "transcribe", {"name": None, "language": "ru"})
        _send(s, "audio-start", {"rate": rate, "width": width, "channels": ch, "timestamp": 0})
        step = CHUNK * width * ch
        for k in range(0, len(pcm), step):
            _send(s, "audio-chunk", {"rate": rate, "width": width, "channels": ch,
                                     "timestamp": k // (rate * width * ch)}, pcm[k:k + step])
        _send(s, "audio-stop", {"timestamp": len(pcm) // (rate * width * ch)})
        while True:
            typ, data = _recv_event(rd)
            if typ is None:
                return ""
            if typ == "transcript":
                return (data.get("text") or "").strip()
    finally:
        s.close()


def transcribe(path, host=HOST, port=PORT, timeout=120.0):
    with wave.open(path, "rb") as wf:
        rate, width, ch = wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
        pcm = wf.readframes(wf.getnframes())
    return transcribe_pcm(pcm, host, port, rate, width, ch, timeout)


def transcribe_wav48(x, host=HOST, port=PORT):
    """звук синтеза (48 кГц float) -> текст: ресемпл в 16 кГц и PCM"""
    from voice.audio import pcm_bytes, resample
    from voice.model import SR
    return transcribe_pcm(pcm_bytes(resample(x, SR, 16000)), host, port)
