"use strict";
/* silero web: обучение (запись/файл -> опора -> задача) и генерация
   (текст -> wav, история справа). Без сборки, один файл. */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const safeName = (s) => String(s).replace(/[\\/:*?"<>|]+/g, "_").trim() || "voice";

async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = j.detail || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  return r;
}
async function apiJSON(url, opts) { return (await api(url, opts)).json(); }
const latin1fix = (s) => s ? decodeURIComponent(escape(s)) : "";

/* ---------------------------------------------------------------- вкладки */
function switchTab(which) {
  const train = which === "train";
  $("train").hidden = !train;
  $("gen").hidden = train;
  $("tabTrain").classList.toggle("active", train);
  $("tabGen").classList.toggle("active", !train);
  if (!train) { loadVoices(); loadHistory(); }
}
$("tabTrain").onclick = () => switchTab("train");
$("tabGen").onclick = () => switchTab("gen");

/* ================================================== ОБУЧЕНИЕ ============ */
let srcBlob = null;          // записанный или выбранный файл опоры
let uploadId = null;         // приготовленная опора на сервере
let rec = null;              // состояние записи {ctx, proc, stream, chunks, t0, sr}
let recTimer = null;

function encodeWav(x, sr) {
  const buf = new ArrayBuffer(44 + x.length * 2);
  const v = new DataView(buf);
  const wstr = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  wstr(0, "RIFF"); v.setUint32(4, 36 + x.length * 2, true); wstr(8, "WAVE");
  wstr(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  wstr(36, "data"); v.setUint32(40, x.length * 2, true);
  let o = 44;
  for (let i = 0; i < x.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, x[i]));
    v.setInt16(o, s < 0 ? s * 32768 : s * 32767, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

$("recBtn").onclick = async () => {
  if (rec) { stopRec(); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false,
               autoGainControl: false } });
  } catch (e) { alert("Микрофон недоступен: " + e.message); return; }
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc);
  const sink = ctx.createGain(); sink.gain.value = 0;
  proc.connect(sink); sink.connect(ctx.destination);
  rec = { ctx, proc, stream, chunks, t0: Date.now(), sr: ctx.sampleRate };
  $("recBtn").textContent = "Остановить запись";
  $("recBtn").classList.add("rec-on");
  recTimer = setInterval(() => {
    $("recTime").textContent = ((Date.now() - rec.t0) / 1000).toFixed(1) + " с";
  }, 100);
};

function stopRec() {
  const s = rec;
  rec = null;
  clearInterval(recTimer);
  $("recBtn").textContent = "Записать с микрофона";
  $("recBtn").classList.remove("rec-on");
  s.proc.disconnect();
  s.stream.getTracks().forEach((t) => t.stop());
  s.ctx.close();
  let n = 0; s.chunks.forEach((c) => n += c.length);
  if (n < s.sr) { alert("Слишком короткая запись"); return; }
  const x = new Float32Array(n);
  let o = 0;
  for (const c of s.chunks) { x.set(c, o); o += c.length; }
  srcBlob = new File([encodeWav(x, s.sr)], "rec.wav", { type: "audio/wav" });
  $("srcAudio").src = URL.createObjectURL(srcBlob);
  $("srcAudio").hidden = false;
}

$("fileInput").onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  srcBlob = f;
  $("srcAudio").src = URL.createObjectURL(f);
  $("srcAudio").hidden = false;
};

$("prepareBtn").onclick = async () => {
  if (!srcBlob) { alert("Сначала запишите фразу или выберите файл"); return; }
  const text = $("refText").value.trim();
  if (!text) { alert("Введите текст опорной фразы"); return; }
  const fd = new FormData();
  fd.append("file", srcBlob, srcBlob.name || "rec.wav");
  fd.append("text", text);
  $("prepareBtn").disabled = true;
  $("prepInfo").hidden = true;
  try {
    const info = await apiJSON("/api/ref/prepare", { method: "POST", body: fd });
    uploadId = info.id;
    $("prepInfo").hidden = false;
    $("markedText").value = info.marked_text;
    const asr = info.asr
      ? `сверка с ASR: CER ${(info.asr.cer * 100).toFixed(1)} % — «${info.asr.text}»`
      : "ASR (gigaam) недоступен — просто проверьте текст глазами";
    $("prepStats").textContent =
      `опора ${info.duration} с, речи ${info.speech_sec} с; ${asr}`;
    $("refAudio").src = `/api/ref/${info.id}/audio`;
    $("prepWarn").innerHTML = (info.warnings || [])
      .map((w) => `<div>${w}</div>`).join("");
    $("prepWarn").hidden = !info.warnings || !info.warnings.length;
  } catch (e) {
    alert("Опора не приготовилась: " + e.message);
  } finally {
    $("prepareBtn").disabled = false;
  }
};

async function startJob(mode) {
  const name = $("modelName").value.trim();
  if (!name) { alert("Задайте имя модели"); return; }
  if (!uploadId) { alert("Сначала приготовьте опору"); return; }
  const marked = $("markedText").value.trim();
  if (!marked) { alert("Пустой текст с ударениями"); return; }
  try {
    await apiJSON("/api/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, mode, ref_id: uploadId, marked_text: marked }),
    });
    switchTab("train");
    pollJobs(true);
  } catch (e) { alert("Не запустилось: " + e.message); }
}
$("probeBtn").onclick = () => startJob("probe");
$("fullBtn").onclick = () => startJob("full");

/* задачи: карточки с прогрессом */
const stepDot = { done: "done", running: "run", pending: "wait", error: "err" };
const stateName = { queued: "в очереди", running: "идёт", done: "готово",
                    error: "ошибка", cancelled: "отменено" };

async function pollJobs(force) {
  let jobs = [];
  try { jobs = (await apiJSON("/api/jobs")).jobs; } catch (e) { return; }
  const box = $("jobs");
  if (!jobs.length) { box.textContent = "пока пусто"; box.classList.add("mut"); return; }
  box.classList.remove("mut");
  box.innerHTML = "";
  for (const j of jobs.slice(0, 12)) {
    const card = document.createElement("div");
    card.className = "job " + j.state;
    const steps = j.steps.map((s) => `
      <div class="step ${stepDot[s.state] || ""}">
        <span class="sdot"></span><span class="stitle">${esc(s.title)}</span>
        <span class="sdet">${esc(s.state === "running" ? s.detail : "")}</span>
      </div>`).join("");
    const log = (j.log || []).slice(-8).join("\n");
    const live = j.state === "queued" || j.state === "running";
    const canFull = j.state === "done" && j.mode === "probe";
    let metrics = "";
    if (j.state === "done" && j.metrics && j.metrics.cer != null) {
      metrics = `CER ${(j.metrics.cer * 100).toFixed(2)} %, F0 ${j.metrics.f0} Гц`;
    }
    card.innerHTML = `
      <div class="jhead">
        <b>${esc(j.name)}</b>
        <span class="badge">${j.mode === "probe" ? "проба" : "полное"}</span>
        <span class="badge ${j.state}">${stateName[j.state] || j.state}</span>
        <span class="mut">${Math.round(j.progress * 100)} %</span>
      </div>
      <div class="pbar"><div class="pfill ${j.state === "error" ? "err" : ""}"
        style="width:${Math.round(j.progress * 100)}%"></div></div>
      <div class="steps">${steps}</div>
      ${metrics ? `<div class="mut">${metrics}</div>` : ""}
      ${j.error && j.state !== "cancelled" ? `<div class="warn">${esc(j.error)}</div>` : ""}
      <pre class="jlog">${esc(log)}</pre>
      <div class="row">
        ${live ? `<button class="ghost" data-cancel="${j.id}">отменить</button>` : ""}
        ${canFull ? `<button class="primary" data-full="${j.model_id}">Продолжить до полного</button>` : ""}
        ${canFull ? `<button class="ghost" data-reprobe="${j.model_id}" title="Пересчитать черновой голос (например, после правки опоры)">пере-проба</button>` : ""}
      </div>`;
    box.appendChild(card);
  }
  box.querySelectorAll("[data-cancel]").forEach((b) => {
    b.onclick = async () => {
      try { await apiJSON(`/api/jobs/${b.dataset.cancel}/cancel`, { method: "POST" }); }
      catch (e) { alert(e.message); }
      pollJobs(true);
    };
  });
  box.querySelectorAll("[data-full]").forEach((b) => {
    b.onclick = async () => {
      try {
        await apiJSON("/api/jobs", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_id: b.dataset.full, mode: "full" }),
        });
        pollJobs(true);
      } catch (e) { alert(e.message); }
    };
  });
  box.querySelectorAll("[data-reprobe]").forEach((b) => {
    b.onclick = async () => {
      try {
        await apiJSON("/api/jobs", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_id: b.dataset.reprobe, mode: "probe" }),
        });
        pollJobs(true);
      } catch (e) { alert(e.message); }
    };
  });
}
setInterval(pollJobs, 2000);

/* ================================================== ГЕНЕРАЦИЯ =========== */
let MODELS = [];

async function loadVoices() {
  let d;
  try { d = await apiJSON("/api/models"); } catch (e) { return; }
  MODELS = d.models;
  const sel = $("voiceSel");
  const was = sel.value;
  sel.innerHTML = "";
  const own = MODELS.filter((m) => m.ready);
  if (own.length) {
    const g = document.createElement("optgroup");
    g.label = "Свои модели";
    for (const m of own) {
      const o = document.createElement("option");
      o.value = m.id;
      o.textContent = m.name + (m.kind === "probe" ? " (проба)" : "");
      g.appendChild(o);
    }
    sel.appendChild(g);
  }
  const gs = document.createElement("optgroup");
  gs.label = "Стоковые";
  for (const v of d.stock) {
    const o = document.createElement("option");
    o.value = "stock:" + v;
    o.textContent = v;
    gs.appendChild(o);
  }
  sel.appendChild(gs);
  const full = own.find((m) => m.kind === "full");
  sel.value = was || (full || own[0] || { id: "" }).id || "stock:xenia";
  if (!sel.value) sel.value = "stock:xenia";
}

function selectedModel() {
  const v = $("voiceSel").value;
  return v.startsWith("stock:") ? null : MODELS.find((m) => m.id === v);
}

$("previewBtn").onclick = () => {
  const m = selectedModel();
  if (!m) { alert("Превью есть только у своих моделей"); return; }
  new Audio(`/api/models/${m.id}/preview`).play();
};

$("renameBtn").onclick = async () => {
  const m = selectedModel();
  if (!m) { alert("Переименовать можно свою модель"); return; }
  const name = prompt("Новое имя модели", m.name);
  if (!name || !name.trim()) return;
  try {
    await apiJSON(`/api/models/${m.id}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name.trim() }),
    });
    await loadVoices();
    loadHistory();
  } catch (e) { alert(e.message); }
};

$("delModelBtn").onclick = async () => {
  const m = selectedModel();
  if (!m) { alert("Удалить можно свою модель"); return; }
  if (!confirm(`Удалить модель «${m.name}» вместе с весами?`)) return;
  try {
    await apiJSON(`/api/models/${m.id}`, { method: "DELETE" });
    await loadVoices();
  } catch (e) { alert(e.message); }
};

/* настройки: живой вывод значений */
const bindRange = (id, out, fmt) => {
  $(id).oninput = () => { $(out).textContent = fmt($(id).value); };
};
bindRange("rate", "rateOut", (v) => (+v).toFixed(2));
bindRange("pitch", "pitchOut", (v) => v);
bindRange("pause", "pauseOut", (v) => (+v).toFixed(2));
bindRange("gap", "gapOut", (v) => (+v).toFixed(2));

$("resetBtn").onclick = () => {
  $("rate").value = 1; $("rateOut").textContent = "1.00";
  $("pitch").value = 0; $("pitchOut").textContent = "0";
  $("pause").value = 0.5; $("pauseOut").textContent = "0.50";
  $("gap").value = 0.25; $("gapOut").textContent = "0.25";
  $("numbersChk").checked = true;
  $("manualChk").checked = false;
  $("manualBox").hidden = true;
  $("srSel").value = "48000";
};

$("manualChk").onchange = async () => {
  const on = $("manualChk").checked;
  $("manualBox").hidden = !on;
  if (on && !$("genMarked").value.trim()) await fillMarked();
};

async function fillMarked() {
  const text = $("genText").value.trim();
  if (!text) { alert("Сначала введите текст"); return; }
  try {
    const d = await apiJSON("/api/mark", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    $("genMarked").value = d.marked;
  } catch (e) { alert(e.message); }
}
$("markBtn").onclick = fillMarked;

function genParams() {
  return {
    speech_rate: +$("rate").value,
    pitch_pct: +$("pitch").value,
    pause: +$("pause").value,
    gap: +$("gap").value,
    numbers: $("numbersChk").checked,
    accent: !$("manualChk").checked,
    sr: +$("srSel").value,
  };
}

$("genBtn").onclick = async () => {
  const manual = $("manualChk").checked;
  const text = (manual ? $("genMarked").value : $("genText").value).trim();
  if (!text) { alert("Введите текст"); return; }
  const voice = $("voiceSel").value;
  $("genBtn").disabled = true;
  $("genTime").textContent = "генерирую…";
  $("lostWarn").hidden = true;
  const t0 = performance.now();
  try {
    const r = await api("/api/generate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, voice, params: genParams() }),
    });
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    $("resultAudio").hidden = false;
    $("resultAudio").src = url;
    $("resultAudio").play();
    const voiceName = latin1fix(r.headers.get("X-Voice"));
    const lost = latin1fix(r.headers.get("X-Lost"));
    const sec = r.headers.get("X-Sec");
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    $("genTime").textContent =
      `${sec} с звука за ${dt} с (синтез ${r.headers.get("X-Synth-Sec")} с)`;
    if (lost) {
      $("lostWarn").hidden = false;
      $("lostWarn").textContent = `ВНЕ АЛФАВИТА, выпало из речи: ${lost}` +
        (manual ? " — помните, «+» стоит перед ударной гласной" : "");
    }
    $("markedShow").textContent = latin1fix(r.headers.get("X-Marked"));
    const ts = new Date();
    const p2 = (n) => String(n).padStart(2, "0");
    const fname = `${safeName(voiceName)}_${ts.getFullYear()}${p2(ts.getMonth() + 1)}` +
      `${p2(ts.getDate())}-${p2(ts.getHours())}${p2(ts.getMinutes())}${p2(ts.getSeconds())}.wav`;
    $("dlLink").hidden = false;
    $("dlLink").href = url;
    $("dlLink").setAttribute("download", fname);
    loadHistory();
  } catch (e) {
    $("genTime").textContent = "";
    alert("Не сгенерировалось: " + e.message);
  } finally {
    $("genBtn").disabled = false;
  }
};

/* история */
const histAudio = new Audio();

async function loadHistory() {
  let items = [];
  try { items = (await apiJSON("/api/history?limit=200")).items; } catch (e) { return; }
  const box = $("history");
  box.innerHTML = "";
  if (!items.length) { box.textContent = "пока пусто"; box.classList.add("mut"); return; }
  box.classList.remove("mut");
  for (const e of items) {
    const d = new Date(e.ts * 1000);
    const p2 = (n) => String(n).padStart(2, "0");
    const row = document.createElement("div");
    row.className = "hitem";
    row.innerHTML = `
      <div class="row wrap">
        <button class="play" data-id="${e.id}" title="Прослушать">играть</button>
        <b class="vbadge">${esc(e.voice_name)}</b>
        <span class="mut">${p2(d.getDate())}.${p2(d.getMonth() + 1)} ${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())} · ${e.sec} с</span>
        <span class="grow"></span>
        <a href="/api/history/${e.id}/audio?dl=1" download="${esc(e.file)}" class="ghost">скачать</a>
        <button class="ghost" data-repeat="${e.id}" title="Подставить текст и настройки">повторить</button>
        <button class="ghost" data-del="${e.id}">удалить</button>
      </div>
      <div class="htext">${esc(e.text)}</div>`;
    box.appendChild(row);
  }
  box.querySelectorAll("[data-id]").forEach((b) => {
    b.onclick = () => {
      histAudio.src = `/api/history/${b.dataset.id}/audio`;
      histAudio.play();
      box.querySelectorAll(".play").forEach((x) => x.classList.remove("playing"));
      b.classList.add("playing");
    };
  });
  box.querySelectorAll("[data-del]").forEach((b) => {
    b.onclick = async () => {
      try { await apiJSON(`/api/history/${b.dataset.del}`, { method: "DELETE" }); }
      catch (er) { alert(er.message); }
      loadHistory();
    };
  });
  box.querySelectorAll("[data-repeat]").forEach((b) => {
    b.onclick = () => {
      const e = items.find((x) => x.id === b.dataset.repeat);
      if (!e) return;
      $("genText").value = e.text;
      $("genMarked").value = e.marked || "";
      const p = e.params || {};
      const set = (id, out, v, fmt) => {
        if (v == null) return;
        $(id).value = v;
        $(out).textContent = fmt(v);
      };
      set("rate", "rateOut", p.speech_rate, (v) => (+v).toFixed(2));
      set("pitch", "pitchOut", p.pitch_pct, (v) => v);
      set("pause", "pauseOut", p.pause, (v) => (+v).toFixed(2));
      set("gap", "gapOut", p.gap, (v) => (+v).toFixed(2));
      $("numbersChk").checked = p.numbers !== false;
      const manual = p.accent === false;
      $("manualChk").checked = manual;
      $("manualBox").hidden = !manual;
      if (p.sr) $("srSel").value = String(p.sr);
      const opt = Array.from($("voiceSel").options)
        .find((o) => o.value === e.voice_id);
      if (opt) $("voiceSel").value = e.voice_id;
    };
  });
}

/* старт */
loadVoices();
loadHistory();
pollJobs(true);
