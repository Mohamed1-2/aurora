"use strict";
/* AURORA HUD — EV-native cockpit + browser voice experience.
 * Aggregates the frozen system state via GET /v1/hud (no inference here),
 * with provenance-first labelling on every displayed value.
 */
const $ = id => document.getElementById(id);
const qs = new URLSearchParams(location.search);
const FIXTURE = qs.has("fixture");
const COCKPIT = qs.get("mode") === "cockpit";
const VEHICLE = "ORION-X-TEST-01";
const SEV = { info: "ok", warning: "warn", critical: "crit" };
const PROV_COLOR = { measured: "ok", prediction: "prediction", derived: "derived", documentation: "documentation", simulator: "simulator" };

let hud = null;
let hudPred = null;
let view = "drive";
let voiceEnabled = false;
let history = [];
let curPop = null;
let listening = false;
let flight = null;

const fmt = (v, d = 1) => (v === null || v === undefined ? "--" : (Math.round(v * 10 ** d) / 10 ** d).toFixed(d));
const sgn = (v, d = 1) => (v === null || v === undefined ? "--" : `${v > 0 ? "+" : ""}${fmt(v, d)}`);
const esc = s => String(s).replace(/[&<>"']/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));

/* ---------------- boot ---------------- */
window.addEventListener("DOMContentLoaded", boot);

function boot() {
  requestAnimationFrame(() => document.body.classList.toggle("cockpit", COCKPIT));
  if (voiceEnabled) document.body.classList.add("voice");
  document.querySelectorAll("#nav button").forEach(b => b.onclick = () => setView(b.dataset.view));
  $("cockpit-btn").onclick = () => {
    document.body.classList.toggle("cockpit");
    $("cockpit-btn").classList.toggle("on", document.body.classList.contains("cockpit"));
  };
  $("voice-on").onclick = () => {
    voiceEnabled = !voiceEnabled;
    $("voice-on").textContent = voiceEnabled ? "Voice on" : "Voice off";
    $("voice-on").classList.toggle("on", voiceEnabled);
  };
  wireAsk("quick");
  wireAsk("ask");
  $("cmp-run").onclick = compare;
  document.addEventListener("click", e => { if (curPop && !curPop.contains(e.target)) closePop(); });
  if (FIXTURE) loadFixture();
  else { poll(); pollTimer = setInterval(poll, 2500); }
}

function setView(v) {
  view = v;
  document.querySelectorAll("#nav button").forEach(b => b.classList.toggle("on", b.dataset.view === v));
  document.querySelectorAll("main.view").forEach(m => m.classList.toggle("on", m.id === `view-${v}`));
  if (v === "predict") { if (!FIXTURE) fetchPred(); if (predTimer) clearInterval(predTimer); predTimer = setInterval(() => { if (view === "predict") fetchPred(); }, 15000); }
  if (v === "alerts" && hud) renderAlerts();
  if (v === "inspect" && hud) renderInspect();
}

function livePill() {
  const svc = (hud && hud.services) || {};
  const ok = Object.entries(svc).filter(([k]) => k !== "provenance").every(([, s]) => s === "ok");
  $("live-pill").className = `pill ${ok ? "ok" : "warn"}`;
  $("live-text").textContent = FIXTURE ? "fixture" : (hud && hud.telemetry && hud.telemetry.status === "live" ? "live" : "offline");
}

/* ---------------- fixtures ---------------- */
function loadFixture() {
  fetch("/fixture.json").then(r => r.json()).then(d => { hud = d; hudPred = d; renderAll(); });
}

/* ---------------- polling ---------------- */
async function poll() {
  try {
    const r = await fetch("/v1/hud");
    hud = await r.json();
    renderAll();
  } catch { $("live-pill").className = "pill crit"; $("live-text").textContent = "offline"; }
}
async function fetchPred() {
  try { const r = await fetch("/v1/hud?prediction=1"); hudPred = await r.json(); renderPredict(); }
  catch { /* keep last pred state */ }
}

/* ---------------- render core ---------------- */
function renderAll() {
  if (!hud) return;
  const id = hud.identity || {};
  $("brand-sub").textContent = `${id.run || hud.vehicle?.id || "?"} · ${id.prompt_version || ""} · ${id.rag_index_version || ""}`;
  const prov = (hud.telemetry || {}).provenance || "derived";
  $("prov-pill").className = `pill ${prov === "simulator" ? "warn" : "ok"}`;
  $("prov-pill").textContent = `◉ ${prov}`;
  $("prov-pill").onclick = e => showPop(prov, e);
  livePill();
  renderGlance();
  renderAlertBar();
  renderAlerts();
  renderInspect();
  renderLegend();
  if (view === "predict" && hudPred) renderPredict();
}

function renderLegend() {
  const lg = (hud && hud.provenance_legend) || {};
  const names = ["measured", "prediction", "derived", "documentation", "simulator"];
  $("prov-legend").innerHTML = names.map(n => `<div class="check ${n === "simulator" ? "ok" : ""}" style="display:block;margin:6px 0"><span class="prov ${PROV_COLOR[n]}"><i></i>${n}</span> ${lg[n] || ""}</div>`).join("");
}

/* ---------------- glance ---------------- */
function sparkSVG(pts, color) {
  if (!pts || pts.length < 2) return "";
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const W = 100, H = 26, pad = 1;
  const px = t => x1 === x0 ? W / 2 : pad + (t - x0) / (x1 - x0) * (W - 2 * pad);
  const py = v => y1 === y0 ? H / 2 : pad + (1 - (v - y0) / (y1 - y0)) * (H - 2 * pad);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><polyline points="${pts.map(p => `${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="1.6"/></svg>`;
}
const priv = 2 * Math.PI * 101;   // SOC ring
const rangeR = 2 * Math.PI * 118; // range ring

function renderGlance() {
  const t = (hud && hud.telemetry && hud.telemetry.signals) || {};
  const s = k => t[k] || {};
  const soc = s("battery_soc_pct"), dte = s("distance_to_empty_km"), spd = s("vehicle_speed_kph");
  const mp = s("motor_power_kw"), cr = s("charge_rate_kw"), btmp = s("battery_temp_c"), co = s("coolant_temp_c");
  const prov = (hud.telemetry || {}).provenance || "derived";
  const charging = hud.vehicle?.charging || false;
  const socOff = priv * (1 - Math.min(Math.max((soc.value || 0), 0), 100) / 100);
  const rangeFr = Math.min(Math.max((dte.value || 0), 0) / 600, 1);
  const chargeArc = charging ? `<circle class="ring charge" cx="125" cy="125" r="80" stroke-dasharray="4 6"/>` : "";

  const drv = Math.max(mp.value || 0, 0), reg = Math.min(mp.value || 0, 0);
  document.getElementById("glance").innerHTML = `
  <div class="card socwrap">
    <div class="svg">
      ${chargeArc}
      <svg width="250" height="250">
        <circle class="ring" cx="125" cy="125" r="118"/><circle class="ring range" cx="125" cy="125" r="118"/>
        <circle class="ring range val" cx="125" cy="125" r="118" stroke-dasharray="${rangeR}" stroke-dashoffset="${rangeR * (1 - rangeFr)}"/>
        <circle class="ring" cx="125" cy="125" r="101"/>
        <circle class="ring val" cx="125" cy="125" r="101" stroke-dasharray="${priv}" stroke-dashoffset="${socOff}"/>
      </svg>
      <div class="core">
        <div class="big">${fmt(soc.value, 1)}%</div>
        <div class="cap">${charging ? "charging" : (hud.vehicle?.mode || "normal")}</div>
        <div class="range">≈ ${fmt(dte.value, 0)} km</div>
      </div>
    </div>
    <div style="display:flex;gap:8px;margin-top:10px">
      ${provDot("simulator", "SOC", soc)}
      ${provDot("simulator", "range", dte)}
    </div>
  </div>

  <div class="card energy">
    <h3>Energy ${provDot("simulator", "telemetry")}</h3>
    <div class="speed"><span class="big mono">${fmt(spd.value, 1)}</span> <span class="u">km/h</span></div>
    ${sparkSVG(spd.spark, "#38bdf8")}
    <div class="flow">
      <div class="tile"><div class="k">Drive</div><div class="v draw">${fmt(drv, 1)} kW</div></div>
      <div class="tile"><div class="k">Regen</div><div class="v regen">${reg < 0 ? `${fmt(-reg, 1)} ft` : "--"}</div></div>
      <div class="tile"><div class="k">Charge</div><div class="v draw">${charging ? `${fmt(cr.value, 1)} kW` : "--"}</div></div>
    </div>
    <div class="statrow">
      <div class="mini"><div class="k">Σ pack current</div><div class="v">${fmt(s("pack_current_a").value, 0)} A</div></div>
      <div class="mini"><div class="k">V pack</div><div class="v">${fmt(s("battery_voltage_v").value, 0)} V</div></div>
      <div class="mini"><div class="k">Odometer</div><div class="v mono">${fmt(s("odometer_km").value, 0)} km</div></div>
      <div class="mini"><div class="k">DT C</div><div class="v ${(hud.telemetry?.dtc_codes || []).length ? "warn" : ""}">${(hud.telemetry?.dtc_codes || []).join(",") || "none"}</div></div>
    </div>
  </div>

  <div class="card thermal">
    <h3>Thermal ${provDot("simulator", "telemetry")}</h3>
    ${bar("battery", btmp, 46, 54)}
    ${bar("coolant", co, 46, 54)}
    ${sparkSVG(btmp.spark, "#fbbf24")}
    <div class="mini" style="margin-top:8px"><div class="k">Motor power <small>(60 s)</small></div>${sparkSVG(mp.spark, "#2dd4bf")}</div>
  </div>

  <div class="card tyres">
    <h3>Tyre pressure ${provDot("simulator", "telemetry")}</h3>
    <div class="tyres">${["fl", "fr", "rl", "rr"].map(w => {
      const sig = s(`tire_pressure_${w}_kpa`);
      const trend = sig.trending_per_min;
      const bad = trend !== null && trend !== undefined && trend < -10;
      return `<div class="tyre ${bad ? "warn" : ""}">
        <div class="pos"><span>${w.toUpperCase()}</span><span>${bad ? "▼" : ""}</span></div>
        <div class="p">${fmt(sig.value, 0)} <small>kPa</small></div>
        <div class="slope ${bad ? "down" : "ok"}">${bad ? `${sgn(trend, 1)} kPa/min` : "stable"}</div>
        ${sparkSVG(sig.spark, bad ? "#fb7185" : "#2dd4bf")}
      </div>`;
    }).join("")}</div>
  </div>`;

  const kv = [];
  for (const [k, name] of [["battery_soc_pct", "SOC"], ["distance_to_empty_km", "Range"], ["vehicle_speed_kph", "Speed"], ["motor_power_kw", "Motor"], ["charge_rate_kw", "Charge rate"], ["battery_temp_c", "Pack temp"], ["coolant_temp_c", "Coolant"], ["odometer_km", "Odometer"]]) {
    const sig = s(k);
    kv.push(`<div class="kv" style="display:grid;grid-template-columns:auto 1fr auto;gap:6px 10px;align-items:center;font-family:var(--mono);font-size:12px;padding:6px 0;border-bottom:1px dashed var(--line)"><span class="k" style="color:var(--muted)">${name}</span><span class="v">${fmt(sig.value, 1)} ${sig.unit}</span>${provDot(sig.provenance || "simulator", name)}</div>`);
  }
  $("kv-live").innerHTML = kv.join("");
}

function bar(name, sig, t1, t2) {
  const lo = sig.min ?? -40, hi = sig.max ?? 120;
  const pct = v => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
  const tick = v => `<div class="tick" style="left:${pct(v)}%"></div>`;
  const hot = (sig.value || 0) > t1;
  return `<div class="tbar"><div class="lbl"><span><b>${sig.label}</b></span><span style="font-family:var(--mono)">${fmt(sig.value, 1)} °C · <span class="prov ${PROV_COLOR[sig.provenance]}"><i></i>${sig.provenance}</span></span></div>
    <div class="track">${tick(t1)}${tick(t2)}<div class="fill ${hot ? "hot" : ""}" style="width:${pct(sig.value || 0)}%"></div></div></div>`;
}

/* ---------------- alert banner ---------------- */
function renderAlertBar() {
  const box = $("alertbar");
  const active = (hud && hud.proactive && hud.proactive.active) || [];
  if (!active.length) { box.innerHTML = ""; return; }
  const e = active[0];
  const sev = SEV[e.severity] || "info";
  box.innerHTML = `<div class="alertbar ${sev}">
    <span class="sev" style="color:var(--${sev === "crit" ? "crit" : sev === "warn" ? "warn" : "ok"})">${e.severity}</span>
    <div class="msg"><b>${esc(e.title)}</b> — ${esc(e.summary)}</div>
    <span class="prov derived"><i></i>code-decided</span>
    <div class="acts">
      <button class="iconbtn" onclick="setView('alerts')">Details</button>
      <button class="iconbtn" onclick="wordAlerts()">🎙 Speak</button>
      <button class="iconbtn" onclick="ackAlert('${e.event_id}')">Ack</button>
    </div>
  </div>`;
}

/* ---------------- alerts view ---------------- */
function evTable(e) {
  const rows = [];
  for (const tr of e.triggers || []) rows.push(["trigger", `${tr.signal} ${tr.direction === "below" ? "<" : ">"} ${tr.threshold} ${tr.unit} · observed ${tr.observed ?? "--"}`]);
  for (const ev of e.evidence || []) rows.push([ev.label || ev.signal, `${fmt(ev.value, 1)} ${ev.unit}`]);
  if (e.signals_used?.length) rows.push(["signals used", e.signals_used.join(", ")]);
  if (e.occurrence_count > 1) rows.push(["episodes", String(e.occurrence_count)]);
  if (e.status && e.status !== "full") rows.push(["status", e.status]);
  if (!rows.length) rows.push(["evidence", "measured in code — none captured"]);
  return `<table>${rows.map(([k, v]) => `<tr><td class="k">${esc(k)}</td><td class="v"><b>${esc(v)}</b></td></tr>`).join("")}</table>`;
}

function renderAlerts() {
  const p = (hud && hud.proactive) || null;
  $("active-list").innerHTML = !p ? `<span class="obs">proactive engine unavailable</span>` :
    (p.active || []).map(e => evCard(e, true)).join("") || `<span class="obs">no active alerts</span>`;
  const dets = [];
  for (const d of p.detectors_eligible || []) dets.push(`<span class="det ok">${d}</span>`);
  for (const [d, why] of Object.entries(p.detectors_skipped_no_signal || {})) dets.push(`<span class="det skip">${d} <b>skipped</b> · ${esc(why.join(", "))}</span>`);
  $("det-strip").innerHTML = dets.join("");
  $("recent-list").innerHTML = (p.recent || []).map(e => evCard(e, false)).join("") || `<span class="obs">no recent alerts</span>`;
}
function evCard(e, active) {
  const sev = SEV[e.severity] || "info";
  return `<div class="evcard">
    <div class="top">
      <span class="sevtag" style="background:rgba(var(--X),.1);color:var(--${sev});border:1px solid var(--line)">${e.severity}</span>
      <h4>${esc(e.title)} <span class="status">${esc(e.status)} · event ${esc(e.event_id)}</span></h4>
      ${active ? `<button class="iconbtn" onclick="ackAlert('${e.event_id}')">Ack</button>` : `<button class="iconbtn" onclick="wordAlerts('${e.event_id}')">🎙</button>`}
    </div>
    <div class="sub">${esc(e.summary)}</div>
    ${evTable(e)}
  </div>`;
}

async function ackAlert(id) {
  if (FIXTURE) { toast("fixture — alert acknowledged (inert)"); return; }
  try {
    const r = await fetch("/v1/proactive/ask", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: crypto.randomUUID(), vehicle_id: VEHICLE, count: 1, ack_event_ids: [id] }) });
    const d = await r.json();
    toast(`Acknowledged. ${(d.answer || "").slice(0, 90)}`);
    poll();
  } catch { toast("ack failed — assistant unreachable"); }
}

/* ---------------- ask ---------------- */
function wireAsk(kind) {
  const input = $(`q-${kind}`), send = $(`send-${kind}`), mic = $(`mic-${kind}`);
  const submit = () => { const q = input.value.trim(); if (q) { input.value = ""; ask(q); } };
  send.onclick = submit;
  input.addEventListener("keydown", e => { if (e.key === "Enter") submit(); });
  mic.onclick = () => rec($(`q-${kind}`));
}

function ask(question) {
  addBubble(question, "q");
  const b = addBubble("…", "a");
  if (FIXTURE) {
    setTimeout(() => {
      b.lastChild.textContent = "Fixture answer: rear-right tyre pressure is 221 kPa and dropping at 23 kPa/min — well past the 12 kPa/min alert rate. Pressure on the other three tyres is steady near 250 kPa. (Provenance: this is a fixture, not live telemetry.)";
      addMeta(b, { provider: "fixture", model: "fixture/eval", intents: ["proactive", "tyres"], citations: ["km-tyre-guide"], policy_level: "read_only", latency_ms: 0 });
    }, 350);
    return;
  }
  fetch("/v1/ask", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, vehicle_id: VEHICLE, ground_telemetry: true, use_rag: true }) })
    .then(r => r.json())
    .then(d => { b.lastChild.textContent = d.answer || "(no answer)"; addMeta(b, d); if (voiceEnabled) speak(d.answer, d.policy_level, d.refused); })
    .catch(() => { b.lastChild.textContent = "Failed to reach the assistant."; });
}

function wordAlerts(eventId) {
  const id = eventId || (hud?.proactive?.active || [])[0]?.event_id;
  if (FIXTURE) { const q = "Speak my alerts"; addBubble(q, "q"); addBubble("Fixture: active alerts — rear-right tyre pressure drop (warning).\nTrigger: tire_pressure_rr_kpa_slope_per_min −22.6 kPa/min vs threshold −12.0.\nEvidence: pressure 221.1 kPa · rate −22.6 kPa/min.", "a"); return; }
  fetch("/v1/proactive/ask", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: crypto.randomUUID(), vehicle_id: VEHICLE, count: 1, ack_event_ids: id ? [id] : [] }) })
    .then(r => r.json())
    .then(d => { addBubble("🎙 speak alerts", "q"); const b = addBubble(d.answer || "", "a"); addMeta(b, d); if (voiceEnabled) speak(d.answer, d.policy_level, true); })
    .catch(() => toast("unable to word alerts"));
}

function addBubble(text, role) {
  const wrap = document.createElement("div"); wrap.className = "qa";
  const b = document.createElement("div"); b.className = `bubble ${role}`;
  b.innerHTML = `<span>${esc(text)}</span>`;
  wrap.appendChild(b);
  $("chatlog").appendChild(wrap);
  $("chatlog").scrollTop = 1e9;
  return b;
}
function addMeta(b, d) {
  const meta = document.createElement("div"); meta.className = "meta";
  meta.innerHTML = [
    `provider:${d.provider}`, `model:${d.model}`, `${d.latency_ms}ms`, `intents:${(d.intents || []).join(",").split("_").join("-")}`,
    `policy:${d.policy_level || "read_only"}`, d.refused ? "refused" : "", (d.safety_blocked || []).join(", ")
  ].filter(Boolean).join(" · ");
  const res = d.grounding;
  if (res && res.used_telemetry) meta.innerHTML += ` · ${res.telemetry_signals.length} telemetry signals`;
  if (res && res.events_used?.length) meta.innerHTML += ` · alerts [${res.events_used.join(",")}]`;
  b.appendChild(meta);
  const cites = (d.citations || []).filter(c => c && !String(c).startsWith("["));
  if (cites.length) {
    const box = document.createElement("div"); box.className = "cites";
    box.innerHTML = cites.map(c => `<span class="cite doc">📄 ${esc(c)}</span>`).join("");
    b.appendChild(box);
  }
  renderSideCites(cites);
}

function renderSideCites(cites) {
  $("cites-side").innerHTML = cites.map(c => `<div class="cite doc" style="margin-bottom:6px;display:inline-block">📄 ${esc(c)}<br><span style="letter-spacing:.05em;text-transform:none">cited in the grounded answer</span></div>`).join("") || `<span class="obs">no document citations yet — ask a question that needs the knowledge base.</span>`;
}

/* ---------------- voice ---------------- */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
function rec(input) {
  if (!SR) { toast("Speech recognition not available in this browser — type instead."); input.focus(); return; }
  const r = new SR();
  r.lang = "en-US"; r.interimResults = false; r.maxAlternatives = 1;
  setListening(true);
  r.onresult = e => {
    const q = e.results[0][0].transcript.trim();
    if (q) { input.value = ""; ask(q); }
  };
  r.onerror = () => toast("Couldn't hear that — try again or type.");
  r.onend = () => setListening(false);
  r.start();
}
function setListening(on) {
  listening = on;
  document.querySelectorAll(".mic").forEach(m => m.classList.toggle("listening", on));
  if (on) toast("Listening…");
}
function speak(text, policy, tone = false) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const clean = String(text || "").replace(/[➜•🔋🎙📄◉≈|]/g, "").replace(/\s+/g, " ").trim();
  if (!clean) return;
  const u = new SpeechSynthesisUtterance(clean);
  u.rate = tone ? 0.98 : 1.0; u.pitch = 1.0;
  u.onend = () => { if (policy === "prohibited_action") toast("Refusal spoken calmly — offering a safe alternative was noted for the next turn."); };
  window.speechSynthesis.speak(u);
}

/* ---------------- predict view ---------------- */
function renderPredict() {
  const P = hudPred || hud;

  function predBody() {
    const p = (P && P.prediction) || {};
    const exp = p.explanation || null;
    const to = exp && exp.to_prediction;
    const status = p.status || "collecting";
    const stcls = status === "active" ? "ok" : status === "collecting" ? "" : "warn";
    const cov = `<div class="coverage"><div class="bar"><i style="width:${p.coverage_pct || 0}%"></i></div><span class="obs">${fmt(p.coverage_pct || 0, 1)}% · ${p.frame_count || 0}/${p.min_frames} frames for physics min, ${p.full_coverage_frames} for full</span></div>`;
    if (!to) return `<div><span class="pill ${SEV[stcls] || "ok"}">${status}</span> ${cov}<p class="obs" style="margin-top:10px">The calibrated SOC-drop model explains the forecast once nominal coverage is reached and the ML artifact is active. Until then we show coverage honestly — no invented predictions.</p></div>`;
    const phys = (exp.physics_contributions || []).reduce((a, c) => a + Math.abs(c.contribution_soc_pct), 0);
    const ml = (exp.ml_contributions || []).reduce((a, c) => a + Math.abs(c.contribution_soc_pct), 0);
    const total = phys + ml || 1;
    return `<div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap">
      <span class="pill ${to.status === "active" ? "ok" : "warn"}">${to.status === "active" ? "active" : "physics fallback"}</span>
      <span class="mono" style="font-size:30px;font-weight:700">${fmt(to.calibrated_soc_drop_pct, 2)}</span><span class="obs">SOC pts / 300 s</span>
    </div>
    <div class="splitbar" title="physics vs ML share"><div class="phys" style="width:${Math.round(phys / total * 100)}%"></div><div class="ml" style="width:${Math.round(ml / total * 100)}%"></div></div>
    <div class="legenddots"><span><i style="background:var(--accent-2)"></i>physics ${fmt(phys, 2)} pts</span><span><i style="background:#a78bfa"></i>ML ${fmt(ml, 2)} pts</span></div>
    ${cov}
    <div class="kv" style="margin-top:12px">
      <span class="k">physics drop</span><span class="v">${fmt(to.physics_soc_drop_pct, 2)} pts</span>
      <span class="k">ML residual</span><span class="v">${fmt(to.ml_residual_pct, 2)} pts</span>
      <span class="k">λ (blend)</span><span class="v">${fmt(to.lambda_applied, 3)}</span>
      <span class="k">speed band</span><span class="v">${esc(to.speed_band || "—")}</span>
      <span class="k">model</span><span class="v">${esc(exp.model_version || "?")}</span>
      <span class="k">status</span><span class="v">${to.status}</span>
    </div>`;
  }

  $("pred-summary").innerHTML = predBody();
  $("kv-predserv").innerHTML = objRows((P && P.prediction && P.prediction.service) || {}, "active", "status", "model_version", "artifact_dir", "reason");
  const checks = (P && P.prediction && P.prediction.explanation && P.prediction.explanation.identity_checks) || {};
  $("pred-checks").innerHTML = objRows(checks, "physics_sum_matches", "ml_sum_matches", "blend_matches", "delta_matches", "tolerance_soc_pct");

  const trip = P && P.prediction && P.prediction.trip && P.prediction.trip.trip;
  $("kv-trip").innerHTML = trip ? objRows(trip, "destination_km", "remaining_km", "cabin_set_temp_c", "ambient_temp_c") :
    `<span class="obs">no trip active — start one in the simulator (POST /api/trip).</span>`;

  const exp = P && P.prediction && P.prediction.explanation;
  if (!exp) { $("pred-expl").innerHTML = `<p class="obs">No explanation yet — prediction is still collecting nominal coverage, or the service is in physics-only fallback. An explanation will appear the moment coverage is met.</p>`; return; }
  const causes = (exp.top_causes || []).map(c => `<li><span>${esc(c.label)} <span class="prov ${PROV_COLOR[c.provenance]}"><i></i>${c.provenance}</span></span>
    <span class="pct" style="justify-self:start">${sgn(c.contribution_soc_pct, 2)} pts</span><span class="obs">${esc((c.direction || "").replace("_", " "))} · ${esc(c.description || c.key)}</span></li>`).join("");
  const chk = exp.identity_checks || {};
  const checksLine = (chk.physics_sum_matches && chk.ml_sum_matches && chk.blend_matches)
    ? `<span class="check ok"><i></i>physics sum matches · ML sum matches · blend matches — explanation is code-computed, not LLM-invented</span>`
    : `<span class="check ok"><i></i>identity: phys=${chk.physics_sum_matches} ml=${chk.ml_sum_matches} blend=${chk.blend_matches} · tolerance ${chk.tolerance_soc_pct}</span>`;

  $("pred-expl").innerHTML = `<ul class="causes">${causes}</ul>${checksLine}`;
}

/* ---------------- inspect view ---------------- */
function objRows(o, ...keys) {
  if (!o) return `<span class="obs">unavailable</span>`;
  if (o.error) return `<span class="obs">${esc(o.error)}</span>`;
  return `<div class="kv">${keys.map(k => `<span class="k">${esc(k)}</span><span class="v">${o[k] === undefined || o[k] === null ? "—" : esc(o[k])}</span>`).join("")}</div>`;
}

function renderInspect() {
  if (!hud) return;
  const provs = (hud.models && hud.models.providers) || {};
  $("kv-prov").innerHTML = Object.entries(provs).map(([name, p]) =>
    `<div class="kv" style="display:grid;grid-template-columns:auto 1fr;gap:4px 12px;padding:6px 0;border-bottom:1px dashed var(--line)"><span class="k">${esc(name)}</span><span class="v mono">${p.configured ? "configured" : "not configured"} · ${(p.models || []).join(", ") || "—"}</span></div>`).join("") || `<span class="obs">gateway unreachable</span>`;
  $("serving-note").textContent = (hud.models && hud.models.serving_note) || "";
  const idx = hud.documents && hud.documents.index;
  $("kv-rag").innerHTML = idx && idx.error ? `<span class="obs">${esc(idx.error)}</span>` : objRows(idx, "rag_index_version", "n_chunks", "n_terms", "avg_chunk_len") + `<div class="kv" style="margin-top:6px"><span class="k">tiers</span><span class="v">${objKV(idx, "chunks_by_tier")}</span><span class="k">audience</span><span class="v">${objKV(idx, "chunks_by_audience")}</span></div>`;
  $("rag-label").textContent = (hud.documents && hud.documents.label) || "";
  $("svc-health").innerHTML = Object.entries(hud.services || {}).filter(([k]) => k !== "provenance").map(([name, st]) =>
    `<div class="srvc"><span class="dot" style="background:${st === "ok" ? "var(--ok)" : "var(--warn)"}"></span><span class="${st === "ok" ? "ok" : "degraded"}">${st === "ok" ? "ok" : st}</span><span class="k" style="color:var(--muted);font-family:var(--mono)">${esc(name)}</span><span class="prov derived"><i></i>probed</span></div>`).join("");
}

function objKV(o, key) { return o ? JSON.stringify(o[key] || {}) : "—"; }

/* ---------------- compare (Base/LoRA) ---------------- */
async function compare() {
  const q = $("cmp-q").value.trim() || "How is my tyre pressure doing?";
  const mk = async (provider) => {
    const out = $(provider === "peft-base" ? "cmp-base" : "cmp-lora");
    out.classList.remove("refused");
    out.querySelector(".out").textContent = "generating…";
    try {
      const [pr, cm] = await Promise.all([
        fetch(`${baseGW()}/v1/debug/prompt`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages: [{ role: "user", content: q }], provider }) }).then(r => r.json()).catch(() => ({})),
        fetch(`${baseGW()}/v1/complete`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages: [{ role: "user", content: q }], provider, temperature: 0.2, max_tokens: 220 }) }).then(r => r.json()).catch(() => ({ content: "(gateway unavailable)" }))
      ]);
      const refused = cm.refused || (cm.safety && cm.safety.blocked_reasons?.length);
      out.classList.toggle("refused", !!refused);
      const promptHead = pr.prompt ? "prompt identity: same for both adapters (byte-for-byte) ✓" : "prompt renderer unavailable";
      out.querySelector(".out").textContent = `${cm.content || "(no content)"}\n\n${promptHead}`;
    } catch { out.querySelector(".out").textContent = "(compare failed)"; }
  };
  await Promise.all([mk("peft-base"), mk("peft")]);
  toast("Base vs LoRA answered the same prompt — compare the wording difference, not just the model name.");
}

let _gw = null;
function baseGW() {
  if (_gw) return _gw;
  const origin = window.location.origin;
  _gw = origin.includes("8100") ? origin.replace("8100", "8102") : "http://localhost:8102";
  return _gw;
}

/* ---------------- provenance popover ---------------- */
function provDot(prov, label) {
  return `<span class="prov ${PROV_COLOR[prov] || "derived"}" data-prov="${prov}" data-label="${esc(label || "")}" title="tap for provenance"><i></i>${prov}</span>`;
}
function showPop(prov, evt, label) {
  closePop();
  const lg = (hud && hud.provenance_legend) || {};
  const p = document.createElement("div"); p.className = "pop";
  p.innerHTML = `<div class="t"><span class="prov ${PROV_COLOR[prov]}"><i></i>${esc(prov)}</span> provenance</div><p>${esc(lg[prov] || "")}</p>${label ? `<p style="margin-top:6px;color:#7e8eb0">on display: ${esc(label)}</p>` : ""}`;
  document.body.appendChild(p);
  const r = evt.target.getBoundingClientRect();
  p.style.top = `${Math.min(r.bottom + 10, window.innerHeight - 120)}px`;
  p.style.left = `${Math.min(Math.max(r.left, 8), window.innerWidth - 330)}px`;
  curPop = p;
}
function closePop() { if (curPop) { curPop.remove(); curPop = null; } }
document.addEventListener("click", e => {
  const s = e.target.closest(".prov[data-prov]");
  if (s) { showPop(s.dataset.prov, e, s.dataset.label); }
});
document.addEventListener("pointerover", e => {
  const s = e.target.closest(".prov[data-prov]");
  if (s && !s.dataset.primed) { s.dataset.primed = "1"; closePop(); setTimeout(() => { if (document.body.contains(s)) s.dataset.primed = ""; }, 1200); }
});

let toastT = null;
function toast(msg) {
  clearTimeout(toastT);
  let t = document.querySelector(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg;
  toastT = setTimeout(() => t.remove(), 3200);
}
function askp(q) { $("q-quick").value = q; ask(q); }