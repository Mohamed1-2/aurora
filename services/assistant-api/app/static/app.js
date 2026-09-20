"use strict";
/* AURORA cockpit — immersive, telemetry-driven driving experience.
 *
 * Every displayed value comes from a real backend surface:
 *   GET  /v1/latest                raw live telemetry frame (the road + cluster)
 *   GET  /v1/hud[?prediction=1]    provenance-labelled state facade
 *   GET  /v1/control               commanded cruise/climate state
 *   POST /v1/control/{speed,hvac,trip,scenario}   driver commands (relayed)
 *   POST /v1/ask  ·  /v1/proactive/ask            grounded AURORA answers
 *
 * Nothing here is simulated or hardcoded: missing signals render honestly
 * ("--"), unreachable controls surface explicit errors, and AURORA only speaks
 * grounded answers or code-decided alerts.
 */
const $ = id => document.getElementById(id);
const qs = new URLSearchParams(location.search);
const VEHICLE = "ORION-X-TEST-01";
const PROV_COLOR = { measured: "ok", prediction: "prediction", derived: "derived", documentation: "documentation", simulator: "simulator" };
const SEV = { info: "ok", warning: "warn", critical: "crit" };

const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "--" : (Math.round(v * 10 ** d) / 10 ** d).toFixed(d));
const esc = s => String(s).replace(/[&<>"']/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));

function createSessionId() {
  if (typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }

  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;

  const hex = Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20)
  ].join("-");
}

const state = {
  latest: null, hud: null, control: null,
  eng: qs.get("eng") === "1", engKeyCount: 0,
  history: [], sessionId: createSessionId(), lastAnswer: "", requestId: 0, voiceOn: true, session: false, engine: "Eco",
  flow: "idle", timer: null, lastActivity: Date.now(),
  speaking: false, listening: false, meter: false,
};

/* ------------------------------ fixture -------------------------------- */
async function loadFixture() {
  try {
    const f = await fetch("fixture.json").then(r => r.json());
    state.latest = f.latest; state.hud = f.hud; state.control = f.control;
    livePill("fixture");
    renderAll(); renderControls(); renderCluster(); renderPred();
    setFlow("idle");
    toast("Offline fixture view — values shown are a captured snapshot, not live.");
  } catch { toast("no fixture.json found"); }
}

/* ------------------------------ boot ---------------------------------- */
window.addEventListener("DOMContentLoaded", boot);
function boot() {
  AuroraWorld.start();
  wireHeader();
  wireConsole();
  wireControls();
  wireEng();
  wirePop();
  wireDirector();
  AuroraWorld.resume();
  if (state.eng) enterEng();
  clock();
  setInterval(clock, 1000);
  if (qs.get("fixture") === "1") { drawMeterLoop(); loadFixture(); return; }
  pollLatest(); pollHud(); pollControl(); pollPred();
  setInterval(pollLatest, 1000);
  setInterval(pollHud, 3000);
  setInterval(pollControl, 4000);
  setInterval(() => { if (!document.hidden && state.hud) pollPred(); }, 20000);
  drawMeterLoop();
}
function clock() {
  const d = new Date();
  $("clock").textContent = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/* ------------------------------ polling -------------------------------- */
async function pollLatest() {
  try {
    const r = await fetch("/v1/latest");
    const d = await r.json();
    if (!r.ok) throw new Error("telemetry unavailable");
    if (d && d.signals) state.latest = d;
    state.lastReceived = Date.now();
    document.body.classList.toggle("stale", !AuroraDriving.fresh(d));
    const s = (d && d.signals) || {};
    AuroraWorld.sync(s.vehicle_speed_kph, s.odometer_km, d.charging);
    renderCluster();
    renderRibbon();
  } catch { document.body.classList.add("stale"); livePill("offline"); }
}
async function pollHud() {
  try {
    const r = await fetch("/v1/hud");
    state.hud = await r.json();
    renderAll();
  } catch { livePill("offline"); }
}
async function pollControl() {
  try {
    const r = await fetch("/v1/control");
    state.control = await r.json();
    renderControls();
  } catch { state.control = null; renderControls(); }
}
let predBusy = false;
async function pollPred() {
  if (predBusy) return; predBusy = true;
  try {
    const r = await fetch("/v1/hud?prediction=1");
    const d = await r.json();
    if (state.hud) state.hud.prediction = d.prediction;
    renderPred();
  } catch { /* keep last */ } finally { predBusy = false; }
}

/* ------------------------------ renderers ------------------------------ */
function livePill(kind) {
  const pill = $("livePill");
  pill.className = `pill ${kind === "offline" ? "crit" : (kind === "fixture" ? "warn" : "ok")}`;
  $("liveTxt").textContent = kind || "live";
}

function renderAll() {
  const t = (state.hud && state.hud.telemetry) || {};
  document.body.classList.toggle("stale", !AuroraDriving.fresh(state.latest) || Date.now() - (state.lastReceived || 0) > 5000);
  const prov = t.provenance || "derived";
  $("provPill").textContent = `◉ ${prov === "simulator" ? "sim telemetry" : prov}`;
  $("provPill").className = `pill ${prov === "simulator" ? "warn" : "ok"}`;
  $("provPill").dataset.prov = prov;
  livePill(AuroraDriving.fresh(state.latest) && Date.now() - (state.lastReceived || 0) < 5000 ? "live" : "stale");
  const idt = state.hud.identity || {};
  $("brandSub").textContent = "Driving companion";
  renderNav();
  renderAlerts();
  renderRibbon();
  renderPred();
  renderEngStatic();
}

const sig = k => (state.hud && state.hud.telemetry && state.hud.telemetry.signals && state.hud.telemetry.signals[k]) || {};

function renderCluster() {
  const s = (state.latest && state.latest.signals) || {};
  const spd = s.vehicle_speed_kph;
  $("speedVal").textContent = fmt(spd, 0);
  speedGlow(spd);
  const mode = (state.latest && state.latest.driving_mode) || "eco";
  $("modeLabel").textContent = { eco: "ECO", normal: "NORMAL", sport: "SPORT" }[mode] || "ECO";
  $("gear").textContent = (state.control && state.control.charging) ? "P" : (spd > 1 ? "D" : "D");
  const soc = sig("battery_soc_pct").value;
  $("socVal").textContent = fmt(soc, 1);
  const range = sig("distance_to_empty_km").value;
  $("rangeVal").textContent = fmt(range, range > 0 && range < 1 ? 1 : 0);
  const frac = Math.min(Math.max((soc || 0) / 100, 0), 1);
  const C = 2 * Math.PI * 42;
  $("socArc").style.strokeDashoffset = C * (1 - frac);
  const mp = sig("motor_power_kw").value;
  const drv = Math.max(mp || 0, 0), reg = Math.min(mp || 0, 0);
  $("powerVal").textContent = `${drv > 0.5 ? fmt(drv, 0) + " kW" : reg < -0.5 ? fmt(-reg, 0) + " kW regen" : "cruise"}`;
  $("powerBar").style.width = `${Math.min(100, (Math.abs(mp || 0) / 180) * 100)}%`;
  $("powerBar").classList.toggle("regen", (mp || 0) < 0);
  const cr = (state.control && state.control.target_speed_kph);
  $("cruiseTag").textContent = cr ? `cruise ${fmt(cr, 0)}` : "cruise --";
  $("cruiseTag").classList.toggle("ovr", !!(state.control && state.control.override_speed_kph !== null));
}

function speedGlow(spd) {
  const el = $("speedWrap");
  el.classList.toggle("hot", (spd || 0) >= 130);
  el.classList.toggle("fast", (spd || 0) >= 100);
}

function renderNav() {
  const trip = state.hud && state.hud.prediction && state.hud.prediction.trip && state.hud.prediction.trip.trip;
  const rem = sig("destination_remaining_km").value;
  const eta = sig("arrival_eta_min").value;
  const box = $("navCard");
  if ((!trip || trip.completed) && (rem === null || rem === undefined || rem < 0)) {
    box.classList.add("empty");
    $("navDir").textContent = "No destination set";
    $("navDest").textContent = "Ready for your next trip";
    $("navEta").textContent = "";
    return;
  }
  box.classList.remove("empty");
  const full = trip ? trip.destination_km : 1;
  const pct = rem !== null && rem !== undefined ? Math.max(0, Math.min(100, (1 - rem / Math.max(1, full)) * 100)) : 0;
  $("navDir").textContent = trip && trip.completed ? "You have arrived" : "To destination";
  $("navDest").textContent = rem !== null && rem !== undefined ? `${fmt(rem, 1)} km` : "--";
  $("navEta").textContent = eta !== null && eta !== undefined && eta >= 0 ? `ETA ~${fmt(eta, 0)} min` : "";
  $("navBar").style.width = `${pct}%`;
}
function evTable(e) {
  const rows = [];
  for (const tr of e.triggers || []) rows.push([tr.signal, `${tr.threshold} ${tr.unit}`]);
  for (const ev of (e.evidence || []).slice(0, 3)) rows.push([ev.label || ev.signal, `${fmt(ev.value, 1)} ${ev.unit}`]);
  return rows;
}
const alertDelivery = AuroraDriving.createAlertDelivery();
function renderAlerts() {
  const active = (state.hud && state.hud.proactive && state.hud.proactive.active) || [];
  const pending = alertDelivery.next(state.latest, active);
  if (pending && state.voiceOn) {
    alertDelivery.mark(pending);
    // A new critical event interrupts a pending request and voice turn.
    state.controller?.abort(); ++state.requestId;
    stopListen(); stopSpeech();
    $("answer").classList.remove("loading");
    showAnswer(pending.speech, pending.summary);
    $("answerProvider").textContent = "Critical warning";
    speakNeutral(pending.speech, {alerts:true});
  }
  const box = $("alertSlot");
  if (!active.length) { box.innerHTML = ""; document.body.classList.remove("alerting"); return; }
  document.body.classList.add("alerting");
  const worst = active.reduce((a, b) => (SEV[b.severity] === "crit" ? -1 : SEV[b.severity] === "warn" ? 0 : 1) < (SEV[a.severity] === "crit" ? -1 : SEV[a.severity] === "warn" ? 0 : 1) ? b : a, active[0]);
  const sev = SEV[worst.severity] || "info";
  box.innerHTML = `
    <div class="alerts ${sev}">
      <div class="alerts-body">
        <span class="sevdot"></span>
        <div class="alerts-msg"><b>${esc(worst.title)}</b>
          <span>${esc(worst.summary)}</span>
          ${active.length > 1 ? `<em>+${active.length - 1} more</em>` : ""}
          ${evTable(worst).map(([k, v]) => `<i>${esc(k)} ${esc(v)}</i>`).join("")}
        </div>
      </div>
      <div class="alerts-acts">
        <button class="act" onclick="window.AURORA.speakAlerts()">Speak</button>
        <button class="act why" id="alertWhy">Why?</button>
        <button class="act" onclick="window.AURORA.ackAlert('${worst.event_id}')">Acknowledge</button>
      </div>
    </div>`;
  const whyBtn = box.querySelector("#alertWhy");
  if (whyBtn) whyBtn.onclick = () => { showAnswer(worst.summary, worst.summary + "\n" + evTable(worst).map(([k,v]) => `${k}: ${v}`).join("\n")); };
}
function provTag(p, label) {
  return `<span class="prov ${PROV_COLOR[p] || "derived"}" data-prov="${p}" data-label="${esc(label || "")}"><i></i>${p}</span>`;
}
function renderRibbon() {
  if (!state.latest) return;
  const s = state.latest.signals || {};
  const tyreWarn = ((state.hud?.proactive?.active || []).some(e => /tyre|tire/i.test(e.title))) || ["fl", "fr", "rl", "rr"].some(w => { const tr = state.hud && state.hud.telemetry && state.hud.telemetry.signals && state.hud.telemetry.signals[`tire_pressure_${w}_kpa`]; return tr && tr.trending_per_min !== null && tr.trending_per_min !== undefined && tr.trending_per_min < -10; });
  $("tyreTag").classList.toggle("warn", tyreWarn);
  $("tyreTag").innerHTML = tyreWarn ? "CHECK TYRES" : "TYRE MONITOR";
  $("battTemp").textContent = fmt(s.battery_temp_c, 0) + " °C";
  $("battTempTrack").style.width = `${Math.min(100, Math.max(0, ((s.battery_temp_c || 0) - 20) / 35 * 100))}%`;
  $("battTempTrack").classList.toggle("hot", (s.battery_temp_c || 0) > 46);
  $("hvacPwr").textContent = fmt(s.hvac_power_kw, 2) + " kW";
  $("outTemp").textContent = fmt(s.outside_temp_c, 0) + " °C";
  const btmp = sig("battery_temp_c").trending_per_min;
  $("thermalSlope").textContent = btmp !== null && btmp !== undefined ? `${btmp > 0 ? "+" : ""}${fmt(btmp, 1)} °C/min` : "";
}
function renderPred() {
  const p = (state.hud && state.hud.prediction) || {};
  const box = $("predChip");
  const st = p.status || "collecting";
  const exp = p.explanation || null;
  const to = exp && exp.to_prediction;
  let body;
  if (to) {
    body = `<b>Δ SOC ${fmt(to.calibrated_soc_drop_pct, 2)} pts / 5 min</b>
      <span>${st} · ${esc((exp.model_version || "model").slice(0, 14))} ${PROV_COLOR.prediction ? "" : ""}</span>`;
  } else {
    body = `<b>${st === "active" ? "prediction active" : st === "collecting" ? "building prediction…" : "physics fallback"}</b>
      <span>${fmt(p.coverage_pct || 0, 0)}% telemetry coverage</span>`;
  }
  box.innerHTML = `${body}
    <button class="act why" id="predWhy" title="Why is the prediction what it is?">Why?</button>
    <button class="act" id="predWhatIf" title="Perturb speed / cabin and recompute arrival SOC">What-if…</button>
    <i class="prov prediction" data-prov="prediction" data-label="prediction · calibrated SOC-drop model"><u></u>prediction</i>`;
  const why = box.querySelector("#predWhy");
  if (why) why.onclick = () => showWhy();
  const wi = box.querySelector("#predWhatIf");
  if (wi) wi.onclick = () => showWhatIf();
}

/* --------------------------- why / what-if popover -------------------- */
function popSet(mode) {
  const pop = $("predPop");
  pop.classList.toggle("why-mode", mode === "why");
  pop.querySelector(".pop-top b").textContent = mode === "why" ? "Energy explanation" : "What-if";
  pop.hidden = false;
}
function popClose() { $("predPop").hidden = true; }
async function showWhy() {
  if (!state.hud) { toast("no live state yet"); return; }
  popSet("why");
  $("wiOut").className = "pop-out";
  $("wiOut").textContent = "computing from the live prediction window…";
  try {
    const r = await fetch("/v1/explain/why", { method: "POST", headers: { "Accept": "application/json" } });
    const d = await r.json();
    if (!r.ok) { $("wiOut").className = "pop-out err"; $("wiOut").textContent = d.detail || "explain unavailable"; return; }
    const exp = d.explanation || {};
    const top = (exp.top_causes || []).slice(0, 4).map(c => `· ${c.label || c.key}: ${(c.contribution_soc_pct >= 0 ? "+" : "")}${fmt(c.contribution_soc_pct, 2)} pts`).join("\n");
    $("wiOut").textContent = d.narrative + (top ? "\n\n" + top : "");
    if (state.voiceOn) speakNeutral(AuroraDriving.brief(d.narrative));
  } catch {
    $("wiOut").className = "pop-out err";
    $("wiOut").textContent = "could not reach the explainer — is the vehicle online?";
  }
}
function liveWhatIfDefaults() {
  const speed = (state.latest && state.latest.signals && state.latest.signals.vehicle_speed_kph) || 60;
  const cabin = (state.control && state.control.cabin_set_temp_c !== null && state.control.cabin_set_temp_c !== undefined)
    ? state.control.cabin_set_temp_c : 21;
  $("wiSpeed").value = Math.max(0, Math.min(200, Math.round(speed)));
  $("wiSpeedV").textContent = `${$("wiSpeed").value} km/h`;
  $("wiCabin").value = Math.max(12, Math.min(30, Math.round(cabin)));
  $("wiCabinV").textContent = `${$("wiCabin").value}°`;
}
function showWhatIf() {
  if (!(state.latest && state.hud)) { toast("no live state yet"); return; }
  liveWhatIfDefaults();
  $("wiOut").className = "pop-out";
  $("wiOut").textContent = "";
  popSet("whatif");
}
async function runWhatIf() {
  const body = { speed_kph: Number($("wiSpeed").value), cabin_temp_c: Number($("wiCabin").value) };
  $("wiOut").textContent = "recomputing arrival SOC (physics)…";
  try {
    const r = await fetch("/v1/explain/whatif", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok) { $("wiOut").className = "pop-out err"; $("wiOut").textContent = d.detail || "what-if unavailable"; return; }
    const res = d.result || {};
    const extra = `baseline ${fmt(res.baseline_arrival_soc_pct, 1)}% → ${fmt(res.perturbed_arrival_soc_pct, 1)}% (${(res.delta_arrival_soc_pct >= 0 ? "+" : "")}${fmt(res.delta_arrival_soc_pct, 1)} pts)`;
    $("wiOut").textContent = d.narrative + "\n" + extra;
    if (state.voiceOn) speakNeutral(AuroraDriving.brief(d.narrative));
  } catch {
    $("wiOut").className = "pop-out err";
    $("wiOut").textContent = "could not reach the simulator — is the vehicle online?";
  }
}
function wirePop() {
  $("popClose").onclick = popClose;
  $("wiRun").onclick = runWhatIf;
  $("wiReset").onclick = liveWhatIfDefaults;
  $("wiSpeed").oninput = e => $("wiSpeedV").textContent = `${e.target.value} km/h`;
  $("wiCabin").oninput = e => $("wiCabinV").textContent = `${e.target.value}°`;
  document.addEventListener("click", e => {
    if ($("predPop").hidden) return;
    if (!e.target.closest("#predPop") && !e.target.closest("#predChip")) popClose();
  });
}

/* --------------------------- scenario director ------------------------- */
function sceneNote(title, sub) {
  const el = $("sceneNote");
  el.className = "scene-note";
  el.innerHTML = `<b>${esc(title)}</b><span>${esc(sub)}</span>`;
  void el.offsetWidth;
  el.classList.add("show");
}
function dirLog(line, kind = "") {
  const log = $("direcLog");
  const entry = document.createElement("div");
  entry.className = kind;
  entry.textContent = line;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}
function dirCheck(step) {
  for (let i = 1; i <= step; i++) {
    const li = document.querySelector(`#direcCheck [data-step="${i}"]`);
    if (li) li.classList.add("done");
  }
}
async function renderDirector() {
  try {
    const r = await fetch("/v1/scenarios", { headers: { "Accept": "application/json" } });
    const list = await r.json();
    if (!r.ok) { $("direcCards").innerHTML = `<div class="dir-log bad">${esc(list.detail || "catalog unavailable")}</div>`; return; }
    $("direcCards").innerHTML = list.map(s =>
      `<button class="dircard" data-k="${esc(s.key)}"><b>${esc(s.title)}</b><span>${esc(s.tagline)}</span></button>`
    ).join("");
    $("direcCards").querySelectorAll(".dircard").forEach(card => {
      card.onclick = () => launchScenario(card.dataset.k, card);
    });
  } catch {
    $("direcCards").innerHTML = `<div class="dir-log bad">director unreachable — is the assistant-api online?</div>`;
  }
}
async function launchScenario(key, card) {
  if (card) card.classList.add("launching");
  dirLog(`▶ launching: ${key}`, "stp");
  dirCheck(0);
  try {
    const r = await fetch(`/v1/scenarios/${encodeURIComponent(key)}`, { method: "POST", headers: { "Accept": "application/json" } });
    const d = await r.json();
    if (!r.ok) { dirLog(`failed: ${d.detail || r.status}`, "bad"); if (card) card.classList.add("failed"); return; }
    for (const step of d.steps || []) {
      dirLog(`→ ${step.method} ${step.path}`, "stp");
      const ok = step.result && step.result.ack !== false;
      dirLog(`  ${JSON.stringify(step.body)}`, ok ? "out" : "bad");
    }
    const ctl = d.control || {};
    if (d.complete) {
      dirCheck(1);
      dirLog(`✔ scenario active → control ${JSON.stringify({ scenario: ctl.scenario, trip: ctl.trip_active, cabin: ctl.cabin_set_temp_c, ambient: ctl.ambient_temp_c })}`, "out");
    } else {
      dirLog(`✖ incomplete at step ${(d.steps || []).length} — ${d.error || "unknown"}`, "bad");
      if (card) card.classList.add("failed");
    }
    if (card && d.complete) card.classList.add("done");
    const title = card ? card.querySelector("b").textContent : key;
    sceneNote(title || key, "scenario active · watch prediction + alerts react");
    dirLog(`→ "Why?" on the prediction / alerts explains what the model sees now.`, "");
    pollControl(); pollLatest(); pollHud();
  } catch {
    dirLog("director unreachable — attempt failed", "bad");
    if (card) card.classList.add("failed");
  } finally {
    if (card) card.classList.remove("launching");
  }
}
function toggleDirector() {
  const open = $("director").classList.toggle("open");
  $("director").setAttribute("aria-hidden", String(!open));
  if (open) renderDirector();
}
function wireDirector() {
  $("labBtn").onclick = toggleDirector;
  $("direcClose").onclick = toggleDirector;
}

/* ------------------------------ controls -------------------------------- */
function clean(text) { return String(text).replace(/[➜•🔋🎙📄◉≈|—–]/g, "").replace(/\s+/g, " ").trim(); }
function renderControls() {
  const c = state.control || {};
  $("cabinVal").textContent = c.cabin_set_temp_c !== null && c.cabin_set_temp_c !== undefined ? `${fmt(c.cabin_set_temp_c, 0)}°` : "--";
  $("cruiseVal").textContent = c.target_speed_kph !== null && c.target_speed_kph !== undefined ? `${fmt(c.target_speed_kph, 0)}` : "--";
  $("chargeBtn").classList.toggle("on", !!c.charging);
  $("chargeBtn").textContent = c.charging ? "Charging…" : "Charge";
  document.body.classList.toggle("charging", !!c.charging);
  if (c.charging) {
    const soc = sig("battery_soc_pct").value || 0;
    const rate = sig("charge_rate_kw").value || 0;
    $("chgSoc").textContent = `${fmt(soc, 1)}%`;
    $("chgPct").textContent = `${soc >= 100 ? "target reached" : "+" + fmt(rate / 82 * 60, 1) + " %/h"}`;
    document.querySelector(".charger-card .chg-bar").style.width = `${Math.min(100, soc)}%`;
  }
  if (c.cabin_set_temp_c === null || c.cabin_set_temp_c === undefined) $("cabinVal").title = "climate needs a destination";
}
async function postControl(path, body, ok) {
  try {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) { let d; try { d = await r.json(); } catch (e) { } toast((d && d.detail) || "vehicle rejected the command"); return; }
    toast(ok || "command sent");
    pollControl();
  } catch { toast("vehicle unreachable — command not sent"); }
}
async function setCruise(delta) {
  const base = (state.control && state.control.target_speed_kph) || 60;
  const next = Math.min(130, Math.max(0, base + delta));
  await postControl("/v1/control/speed", { speed_kph: next }, `cruise → ${fmt(next, 0)} km/h`);
}
async function setCabin(delta) {
  const base = (state.control && state.control.cabin_set_temp_c) || 21;
  if (base === null || base === undefined) { toast("climate needs an active destination — start a trip"); return; }
  const next = Math.min(30, Math.max(12, base + delta));
  await postControl("/v1/control/hvac", { cabin_temp_c: next }, `cabin set point → ${fmt(next, 0)}°`);
}
async function startTrip() {
  const km = parseFloat($("destKm").value);
  if (!km || km <= 0) { toast("enter a destination distance in km"); return; }
  const cabin = (state.control && state.control.cabin_set_temp_c) || 21;
  await postControl("/v1/control/trip", {
    destination_km: Math.min(km, 1500), cabin_set_temp_c: cabin, ambient_temp_c: 14,
    speed_profile_kph: [60, 90, 110], segment_s: 96,
  }, `trip → ${fmt(km, 1)} km`);
  setTimeout(pollLatest, 500);
}
async function toggleCharge() {
  const charging = !!(state.control && state.control.charging);
  if (charging) { await postControl("/v1/control/scenario", { type: "none", duration_s: 1, intensity: 0 }, "unparked"); }
  else { await postControl("/v1/control/scenario", { type: "charge", duration_s: 150, intensity: 1 }, "charging started · 11 kW"); }
  setTimeout(pollControl, 400);
}
async function runScenario() {
  const type = $("scenSel").value;
  await postControl("/v1/control/scenario", { type, duration_s: type === "none" ? 1 : 240, intensity: 1 },
    type === "none" ? "scenario cleared" : `scenario → ${type}`);
  setTimeout(pollHud, 1200);
  if (type === "leak") setTimeout(pollPred, 3000);
}

/* ------------------------------ AURORA asks ------------------------------ */
function setFlow(flow, status) {
  state.flow = flow;
  $("orbStatus").textContent = status || { idle: "TAP ORB TO LISTEN", listening: "LISTENING", speaking: "SPEAKING", thinking: "THINKING" }[flow] || "";
  document.body.classList.toggle("listening", flow === "listening");
  document.body.classList.toggle("speaking", flow === "speaking");
}
function showAnswer(short, full = short, preserveRefusal = false) {
  const event = AuroraDriving.priorityAlert(state.latest, state.hud?.proactive?.active || []);
  if (event?.thermal && !preserveRefusal) short = event.speech;
  state.lastAnswer = short;
  $("answerProvider").textContent = "Assistant";
  $("answer").innerHTML = `<b>AURORA</b><span>${esc(short)}</span>`;
  $("answerDetails").textContent = full;
  $("answerDetails").hidden = true;
  $("detailsToggle").setAttribute("aria-expanded", "false");
}
async function ask(query, opts = {}) {
  const q = String(query || "").trim();
  if (!q) return;
  if (/^(stop|cancel|be quiet)[.!]?$/i.test(q)) { stopSpeech(); return; }
  if (/^(repeat|repeat that|say that again)[.!]?$/i.test(q)) { if (state.lastAnswer) speakNeutral(state.lastAnswer); return; }
  stopSpeech(); stopListen();
  state.controller?.abort();
  const controller = new AbortController(); state.controller = controller;
  const id = ++state.requestId;
  const timeout = setTimeout(() => controller.abort(), 45000);
  setFlow("thinking", "Thinking…");
  $("transcript").textContent = q;
  $("answer").classList.add("loading");
  $("answer").textContent = "Thinking…";
  $("answerProvider").textContent = "Working";
  try {

    if (/find.*charg|charg.*(nearby|near me|station|stop)/i.test(q)) {
      showAnswer("Charger search isn't connected yet. Use your navigation system to find a station.");
      if (state.voiceOn) speakNeutral(state.lastAnswer);
      return;
    }
    if (/explain.*(battery forecast|prediction)|why.*(prediction|forecast)/i.test(q)) {
      const r = await fetch("/v1/explain/why", {method: "POST", signal: controller.signal});
      const d = await r.json();
      if (!r.ok || !d.narrative) throw new Error("explanation unavailable");
      if (id !== state.requestId) return;
      showAnswer(AuroraDriving.energy(d), d.narrative);
      if (state.voiceOn) speakNeutral(state.lastAnswer);
      return;
    }
    const r = await fetch("/v1/ask", { method: "POST", signal: controller.signal, headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, response_style: "concise", conversation: state.history.slice(-6), session_id: state.sessionId, vehicle_id: VEHICLE, ground_telemetry: true, use_rag: true }) });
    const d = await r.json();
    if (!r.ok || !d.answer) throw new Error("assistant unavailable");
    if (id !== state.requestId) return;
    const short = AuroraDriving.answer(q, d, state.latest, state.hud?.proactive?.active || []);
    state.history.push({role:"user", content:q.slice(0,2000)}, {role:"assistant",content:short.slice(0,2000)});
    state.history = state.history.slice(-6);

    showAnswer(short, AuroraDriving.clean(d.answer) + "\n\n" + (d.citations || []).join("\n") + `\n\n${d.provider} · ${d.latency_ms} ms`, d.refused || d.policy_level === "restricted_action");
    $("answerProvider").textContent = d.provider === "local" ? "Local AI" : d.provider === "guide" ? "App guide" : d.provider === "telemetry" ? "Live reading" : d.provider === "mock" ? "Offline fallback" : d.provider === "policy" ? "Safety policy" : "Assistant";
    if (state.voiceOn) speakNeutral(state.lastAnswer);
    else if (state.session) resumeListening();
    return d;
  } catch (err) {
    if (id === state.requestId) showAnswer(controller.signal.aborted ? "That took too long. Please try again." : "I couldn't connect. Please try again.");
  } finally {
    clearTimeout(timeout);
    if (id === state.requestId) { $("answer").classList.remove("loading"); if (!state.speaking) setFlow("idle"); }
  }
}
function stopSpeech() {
  AuroraVoice.cancel(); state.speaking = false; setFlow("idle");
}
async function speakAlerts() {
  try {
    const r = await fetch("/v1/proactive/ask", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, vehicle_id: VEHICLE, count: 1, ack_event_ids: [] }) });
    const d = await r.json();
    if (!r.ok) throw new Error("alerts unavailable");
    showAnswer(AuroraDriving.brief(d.answer), AuroraDriving.clean(d.answer));
    if (state.voiceOn) speakNeutral(state.lastAnswer, { alerts: true });
  } catch { toast("unable to word active alerts"); }
}
async function ackAlert(id) {
  try {
    const r = await fetch("/v1/proactive/ask", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, vehicle_id: VEHICLE, count: 1, ack_event_ids: [id] }) });
    const d = await r.json();
    if (!r.ok) throw new Error("ack failed");
    const event = (state.hud?.proactive?.active || []).find(e => e.event_id === id);
    if (event) alertDelivery.mark(event);
    toast("Alert acknowledged — AURORA won't repeat it until it recurs.");
    pollHud();
  } catch { toast("ack failed — assistant unreachable"); }
}

/* ------------------------------ voice ------------------------------- */
function studioWpm() {
  return (window.__studioRate && window.__studioRate()) || 175;
}
function speakNeutral(text, { tone = false, alerts = false } = {}) {
  if (!state.voiceOn) return;
  setFlow("speaking", "SPEAKING");
  stopListen();
  state.speaking = true;
  const studio = AuroraVoice.studioState().engine === "studio";
  AuroraVoice.speak(AuroraDriving.clean(text), {
    rate: studio ? studioWpm() : (tone ? 0.94 : (alerts ? 0.96 : 0.99)),
    pitch: 1.0,
    voice: state.selectedVoice,
    onEnd: ok => { if (!ok) toast("Audio unavailable — the answer is on screen."); state.speaking = false; setFlow("idle"); if (state.session) resumeListening(); },
  });
}
let listenHandle = null;
let listenTimeout = null;
function stopListen() {
  if (listenTimeout) { clearTimeout(listenTimeout); listenTimeout = null; }
  if (listenHandle) { listenHandle.stop(); listenHandle = null; }
  state.listening = false;
  AuroraVoice.stopMeter();
}
function startListen(oneShot = false) {
  if (!AuroraVoice.canSTT) { toast("Speech recognition unavailable — type your request below."); return; }
  if (state.listening) return;
  state.listening = true;
  setFlow("listening", "LISTENING");
  AuroraVoice.initMeter();
  listenHandle = AuroraVoice.recognize({
    onInterim: t => { $("transcript").innerHTML = `<b>you</b><span>“${esc(t)}”</span>`; },
    onFinal: q => { stopListen(); if (q.trim()) ask(q); else if (state.session) resumeListening(); },
    onError: e => {
      stopListen();
      const msg = e === "not-allowed" ? "Microphone access denied — enable it or type." : e === "no-speech" ? "I didn't hear anything — try again or type." : "Couldn't hear that — try again or type.";
      toast(msg);
      if (e === "not-allowed" || e === "service-not-allowed") endSession();
      else if (state.session) resumeListening(1200);
    },
    onEnd: () => { state.listening = false; },
  });
  if (!oneShot) {
    listenTimeout = setTimeout(() => {
      if (state.listening) { stopListen(); if (state.session) { toast("Conversation quiet for 45s — ended."); endSession(); } }
    }, 45000);
  }
}
function resumeListening(delay = 500) {
  if (!state.session) return;
  clearTimeout(state.resumeTimer);
  state.resumeTimer = setTimeout(() => { if (state.session && !state.speaking && state.flow !== "thinking") startListen(); }, delay);
}
function startSession() { stopSpeech(); stopListen(); state.session = true; $("orb").classList.add("session"); toast("Conversation on — tap the orb to end."); startListen(); }
function endSession() { clearTimeout(state.resumeTimer); stopSpeech(); stopListen(); state.session = false; $("orb").classList.remove("session"); setFlow("idle"); }

/* ------------------------------ meter loop ------------------------------ */
function drawMeterLoop() {
  const ring = $("orbRing");
  const n = 24;
  ring.innerHTML = Array.from({ length: n }, () => `<i></i>`).join("");
  (function loop() {
    requestAnimationFrame(loop);
    const level = state.listening ? AuroraVoice.meterLevel() : 0;
    const bars = ring.children;
    for (let i = 0; i < n; i++) {
      const v = Math.min(1, Math.max(level + (Math.sin(Date.now() / 120 + i * 1.3)) * 0.05, 0));
      bars[i].style.height = `${6 + v * 14}px`;
      bars[i].style.opacity = state.listening ? (0.35 + v * 0.65) : 0.08;
    }
  })();
}

/* ------------------------------ header & console ------------------------ */
function wireHeader() {
  $("sound").onclick = () => {
    state.voiceOn = !state.voiceOn;
    $("sound").textContent = state.voiceOn ? "Voice on" : "Voice off";
    $("sound").classList.toggle("muted", !state.voiceOn);
    if (!state.voiceOn) { AuroraVoice.cancel(); state.speaking = false; setFlow("idle"); }
  };
  $("fullscreen").onclick = async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); } catch (e) { }
  };
  $("brandMark").onclick = () => {
    state.engKeyCount++;
    clearTimeout(state.engKeyTimer);
    state.engKeyTimer = setTimeout(() => { state.engKeyCount = 0; }, 2200);
    if (state.engKeyCount >= 5) { if (!state.eng) enterEng(); else exitEng(); }
  };
  $("linkHud").onclick = () => { location.href = "./hud/"; };
}
function wireConsole() {
  $("orb").onclick = () => { if (state.session) endSession(); else startSession(); };
  $("form").onsubmit = e => { e.preventDefault(); submitInput(); };
  $("repeatAnswer").onclick = () => { if (state.lastAnswer) speakNeutral(state.lastAnswer); };
  $("stopSpeech").onclick = () => { endSession(); state.controller?.abort(); ++state.requestId; $("answer").classList.remove("loading"); };
  $("detailsToggle").onclick = () => { const open = $("answerDetails").hidden; $("answerDetails").hidden = !open; $("detailsToggle").setAttribute("aria-expanded", String(open)); };
  $("mic").onmousedown = e => { if (e.button === 0) { e.preventDefault(); holdStart(); } };
  $("mic").ontouchstart = e => { e.preventDefault(); holdStart(); };
  ["mouseup", "mouseleave", "touchend"].forEach(k => document.addEventListener(k, () => {
    if (hold) {
      clearTimeout(holdTimer); hold = false;
      if (holdMic) { holdMic.stop(); holdMic = null; }
      const t = document.querySelector(".toast"); if (t) t.remove();
    }
  }));
  const chips = [["How is the car right now?", "Car status"], ["What is my current range?", "Range"], ["Explain my battery forecast", "Energy"]];
  $("chips").replaceChildren(...chips.map(([question, label]) => {
    const button = document.createElement("button"); button.type = "button"; button.textContent = label;
    button.onclick = () => ask(question); return button;
  }));
}
let hold = false, holdTimer = null, holdMic = null;
function holdStart() {
  stopSpeech();
  hold = true;
  holdTimer = setTimeout(() => {
    if (!hold || state.listening) return;
    if (!AuroraVoice.canSTT) { toast("Speech recognition unavailable — type instead."); return; }
    toast("hold to talk…");
    AuroraVoice.initMeter();
    holdMic = AuroraVoice.recognize({
      onInterim: t => { $("transcript").innerHTML = `<b>you</b><span>“${esc(t)}”</span>`; },
      onFinal: q => { holdMic = null; if (q.trim()) ask(q); },
      onError: () => { holdMic = null; toast("Couldn't hear that — try again."); },
      onEnd: () => { holdMic = null; },
    });
  }, 300);
}
function Toast(msg) { toast(msg); }
function submitInput() {
  const q = $("q").value;
  if (q.trim()) { $("q").value = ""; ask(q); }
}
function wireControls() {
  $("cabinMinus").onclick = () => setCabin(-1);
  $("cabinPlus").onclick = () => setCabin(1);
  $("cruiseMinus").onclick = () => setCruise(-10);
  $("cruisePlus").onclick = () => setCruise(10);
  $("setDest").onclick = startTrip;
  $("destKm").addEventListener("keydown", e => { if (e.key === "Enter") startTrip(); });
  $("chargeBtn").onclick = toggleCharge;
  $("scenGo").onclick = runScenario;
}

/* ------------------------------ engineering ----------------------------- */
function enterEng() { state.eng = true; document.body.classList.add("eng"); $("engPill").classList.add("on"); toast("Engineering mode unlocked."); renderEngStatic(); }
function exitEng() { state.eng = false; document.body.classList.remove("eng"); $("engPill").classList.remove("on"); $("engPanel").classList.remove("open"); }
function unlockEngDemo() {
  if (state.eng && $("engPanel").classList.contains("open")) { exitEng(); return; }
  if (!state.eng) enterEng();
  $("engPanel").classList.add("open");
  renderEngDynamic();
}
function wireEng() {
  $("engPill").onclick = unlockEngDemo;
  $("engFtr").onclick = unlockEngDemo;
  document.addEventListener("keydown", e => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = e.target && e.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (e.key.toLowerCase() === "e") { e.preventDefault(); unlockEngDemo(); }
  });
  $("engClose").onclick = exitEng;
  $("cmpRun").onclick = compare;
  AuroraVoice.list(); // warm the browser voice list

  // --- browser voice picker ------------------------------------------------
  const picker = $("voicePicker");
  const fill = () => {
    const vs = AuroraVoice.list();
    picker.innerHTML = `<option value="">auto (best installed voice)</option>` + vs.map(v => `<option value="${esc(v.name)}" ${state.selectedVoice && state.selectedVoice.name === v.name ? "selected" : ""}>${esc(v.name)} · ${v.lang}</option>`).join("");
  };
  fill();
  if ("speechSynthesis" in window) speechSynthesis.addEventListener("voiceschanged", fill);
  picker.onchange = () => { state.selectedVoice = AuroraVoice.pick(picker.value); toast(`Browser voice → ${state.selectedVoice ? state.selectedVoice.name : "auto"}`); };

  // --- studio engine (macOS `say`, served by the backend) ------------------
  const studioPicker = $("studioPicker");
  const fillStudio = () => {
    const s = AuroraVoice.studioState();
    const vs = s.studioVoices.filter(v => /^en/i.test(v.lang || "")).concat(s.studioVoices.filter(v => !/^en/i.test(v.lang || "")));
    studioPicker.innerHTML = vs.map(v => `<option value="${esc(v.name)}" ${s.studioVoice === v.name ? "selected" : ""}>${esc(v.name)} · ${esc(v.lang)}</option>`).join("");
  };
  async function initStudioEngine() {
    const avail = await AuroraVoice.initStudio();
    const vg = document.querySelector('input[name="voiceEngine"]:checked');
    const s = AuroraVoice.studioState();
    const hint = $("voiceHint");
    if (avail) {
      studioPicker.hidden = false;
      fillStudio();
      hint.textContent = `studio ready · ${s.studioVoice || "auto"}`;
      hint.className = "voice-hint ok";
      const studioRadio = $("vgStudio");
      studioRadio.disabled = false;
      if (!vg || vg.value === "studio") { studioRadio.checked = true; AuroraVoice.setEngine("studio"); }
    } else {
      $("vgStudio").disabled = true;
      hint.textContent = "studio unavailable on this host — using the browser voice.";
      hint.className = "voice-hint warn";
      if (vg && vg.value === "studio") { $("vgBrowser").checked = true; AuroraVoice.setEngine("browser"); }
    }
  }
  document.querySelectorAll('input[name="voiceEngine"]').forEach(r => {
    r.addEventListener("change", () => {
      const res = AuroraVoice.setEngine(r.value);
      const s = AuroraVoice.studioState();
      const hint = $("voiceHint");
      studioPicker.hidden = r.value !== "studio";
      toast(`Voice engine → ${r.value === "studio" ? "Studio (macOS say)" : "Browser synthesis"}`);
      hint.textContent = r.value === "studio"
        ? (res.available ? `studio ready · ${s.studioVoice || "auto"}` : "studio unavailable — fallback: browser")
        : "browser-native synthesis · curated pick";
      hint.className = res.available ? "voice-hint ok" : "voice-hint";
    });
  });
  studioPicker.onchange = () => {
    const name = studioPicker.value;
    AuroraVoice.pickStudio(name);
    if (name) toast(`Studio voice → ${name}`);
  };

  $("voiceTest").onclick = () => {
    const s = AuroraVoice.studioState();
    if (s.engine === "studio" && s.studioAvailable) {
      AuroraVoice.speak("This is AURORA's studio voice, synthesized on this Mac. How do I sound?", { rate: 175, pitch: 1.0 });
    } else {
      AuroraVoice.speak("This is AURORA's voice. How do I sound?", { voice: state.selectedVoice, rate: 0.98, pitch: 1.0 });
    }
  };
  $("rateSlider").oninput = e => { $("rateVal").textContent = e.target.value; studioRate(); };
  $("pitchSlider").oninput = e => { $("pitchVal").textContent = e.target.value; studioRate(); };
  function studioRate() { /* slider 0.7..1.35 → say 120..220 words/min when in studio */ return 120 + Math.round((Number($("rateSlider").value) - 0.7) / 0.65 * 100); }
  window.__studioRate = studioRate;
  initStudioEngine();
}
function renderEngStatic() {
  if (!state.hud) return;
  $("ragLabel").textContent = (state.hud.documents && state.hud.documents.label) || "";
  const svc = Object.entries(state.hud.services || {}).filter(([k]) => k !== "provenance");
  $("svcHealth").innerHTML = svc.map(([n, st]) => `<div class="srvc ${st === "ok" ? "ok" : "bad"}"><span class="dot"></span>${st === "ok" ? "ok" : st} <i>${esc(n)}</i></div>`).join("");
  const t = state.hud.telemetry || {};
  const rows = [["source", t.source || "—"], ["provenance", t.provenance || "—"], ["frames", String(t.frame_count ?? "—")], ["window", t.window_seconds ? `${t.window_seconds}s` : "—"], ["status", t.status || "—"]];
  $("kvSrc").innerHTML = rows.map(([k, v]) => `<div class="kv"><b>${k}</b><span>${esc(String(v))}</span></div>`).join("");
  const pol = state.hud.policy || {};
  $("kvSafety").innerHTML = [
    ["policy level", pol.level || "—"],
    ["predictive actions", String(pol.predictive_actions_enabled ?? "—")],
    ["proactive poll s", String(pol.proactive_poll_interval_s ?? "—")],
    ["prediction min frames", String(pol.prediction_min_frames ?? "—")],
    ["provenance", pol.provenance || "—"],
  ].map(([k, v]) => `<div class="kv"><b>${k}</b><span>${esc(String(v))}</span></div>`).join("");
  const diag = state.hud.diagnostics || {};
  const lat = Object.entries(diag.latency_ms || {}).map(([k, v]) => [k.includes("/providers") ? "gateway providers" : k.includes("/index-info") ? "rag index" : "prediction status", v]);
  const probe = diag.probe || {};
  $("kvDiag").innerHTML = lat
    .concat([["probes", Object.entries(probe).map(([k2, v2]) => `${k2}=${v2}`).join(" · ") || "—"]])
    .map(([k, v]) => `<div class="kv"><b>${k}</b><span>${v === null || v === undefined ? "—" : esc(`${v}${typeof v === "number" ? " ms" : ""}`)}</span></div>`).join("");
}
async function renderEngDynamic() {
  const provs = (state.hud && state.hud.models && state.hud.models.providers) || {};
  $("kvProv").innerHTML = Object.entries(provs).map(([n, p]) => `<div class="kv"><b>${esc(n)}</b><span>${p.configured ? "configured" : "not configured"} · ${(p.models || []).join(", ") || "—"}</span></div>`).join("") || `<span>gateway unreachable</span>`;
  const pred = state.hud.prediction || {};
  const svc = pred.service || {};
  const peft = provs.peft || {};
  const peftBase = provs["peft-base"] || {};
  const comparisonReady = !!(peft.configured && peftBase.configured);
  $("cmpRun").disabled = !comparisonReady;
  $("cmpRun").title = comparisonReady ? "Compare configured models" : "Configure both base and LoRA providers first";
  if (!comparisonReady) { $("cmpBase").querySelector(".out").textContent = "Base model not configured for comparison."; $("cmpLora").querySelector(".out").textContent = "LoRA comparison is unavailable on this stack."; }
  $("kvPred").innerHTML = [
    ["status", pred.status || "—"],
    ["coverage", pred.coverage_pct != null ? `${pred.coverage_pct}%` : "—"],
    ["model", svc.model_version || pred.explanation?.model_version || "—"],
    ["lambda", svc.lambda_applied != null ? fmt(svc.lambda_applied, 3) : "—"],
    ["fallback", svc.fallback_reason || pred.status === "physics_only_fallback" ? "physics only" : "none"],
    ["adapter2 (LoRA)", peft.adapter2 || peft.adapter_name || (peft.use_lora ? "enabled" : "—")],
    ["base model", peftBase.base_model || peft.base_model || "—"],
  ].map(([k, v]) => `<div class="kv"><b>${k}</b><span>${esc(String(v))}</span></div>`).join("");
  const idx = state.hud && state.hud.documents && state.hud.documents.index;
  $("kvRag").innerHTML = !idx ? "Index unavailable" : idx.error ? `<span>${esc(idx.error)}</span>` : `<div class="kv"><b>index</b><span>v${idx.rag_index_version || "?"} · ${idx.n_chunks || 0} chunks · ${idx.n_terms || 0} terms</span></div>`;
  const lg = (state.hud && state.hud.provenance_legend) || {};
  $("provLegend").innerHTML = Object.entries(lg).map(([k, v]) => `<div class="kv"><b>${k}</b><span>${esc(v)}</span></div>`).join("");
}
async function compare() {
  const providers = state.hud?.models?.providers || {};
  if (!providers.peft?.configured || !providers["peft-base"]?.configured) { toast("Configure both base and LoRA providers first."); return; }
  const q = $("cmpQ").value.trim() || "How is my tyre pressure doing?";
  const gw = location.origin.includes("8100") ? location.origin.replace("8100", "8102") : "http://localhost:8102";
  const mk = async (provider, el) => {
    const out = $(el);
    out.classList.remove("refused");
    out.querySelector(".out").textContent = "generating…";
    try {
      const cm = await fetch(`${gw}/v1/complete`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [{ role: "user", content: q }], provider, temperature: 0.2, max_tokens: 220 }) }).then(r => r.json()).catch(() => ({ content: "(gateway unavailable)" }));
      out.classList.toggle("refused", !!(cm.refused || (cm.safety && cm.safety.blocked_reasons && cm.safety.blocked_reasons.length)));
      out.querySelector(".out").textContent = cm.content || "(no content)";
      out.querySelector(".who").textContent = provider === "peft-base" ? "peft-base · Mistral-7B base" : "peft · Mistral-7B + LoRA adapter2";
    } catch { out.querySelector(".out").textContent = "(compare failed)"; }
  };
  await Promise.all([mk("peft-base", "cmpBase"), mk("peft", "cmpLora")]);
  toast("Base vs LoRA answered the same prompt — compare the wording, not the name.");
}

/* ------------------------------ plumbing -------------------------------- */
let toastT = null;
function toast(msg) {
  clearTimeout(toastT);
  let t = document.querySelector(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg;
  toastT = setTimeout(() => t.remove(), 3400);
}
document.addEventListener("click", e => {
  const s = e.target.closest(".prov[data-prov]");
  if (!s) return;
  const lg = (state.hud && state.hud.provenance_legend) || {};
  toast(`${s.dataset.prov}: ${lg[s.dataset.prov] || ""}${s.dataset.label ? ` — ${s.dataset.label}` : ""}`);
});
document.addEventListener("visibilitychange", () => { if (document.hidden) { AuroraWorld.pause(); } else { AuroraWorld.resume(); } });

window.AURORA = { refreshEngineering: () => state.hud && renderEngDynamic(), refreshDirector: renderDirector, ask, speakAlerts, ackAlert, setCruise, setCabin, startTrip, toggleCharge, runScenario, pollHud };
window.toast = toast;

// Reviewer-led journey: each step is explicit, with no hidden timed mutations.
window.addEventListener('DOMContentLoaded', () => {
  const start = $('journeyReset'), next = $('journeyNext'), hint = $('journeyHint');
  let step = 0;
  const steps = [
    ['Ask "How is the car right now?" and compare the answer with the cluster.', 'How is the car right now?'],
    ['Check whether the current range covers the configured destination.', 'Can I reach my destination?'],
    ['Try a conversational follow-up about that answer.', 'What should I do?'],
    ['Select Energy to see the five-minute battery forecast; Details shows evidence.', 'Explain my battery forecast'],
    ['Open Trip & simulator controls and run Leak drill. Ask a tyre question while the warning appears.', 'Which tyre is lowest?'],
    ['Demo complete. Use Tour to explore more, or reset for another reviewer.', 'What can you do?']
  ];
  start.addEventListener('click', async () => {
    start.disabled = true; next.disabled = true;
    AuroraVoice.cancel(); state.controller?.abort(); ++state.requestId;
    $("answer").classList.remove("loading"); setFlow("idle");
    try {
      const r = await fetch('/v1/control/demo/reset', {method:'POST'});
      if (!r.ok) throw new Error('Reset unavailable. Check the simulator connection.');
      alertDelivery.reset();
      state.history = []; state.sessionId = createSessionId();
      step = 0; hint.textContent = 'Fresh demo started: 8 km trip, 65% battery. Select Next step to begin.';
      next.disabled = false;
    } catch(e) { hint.textContent = e.message; }
    finally { start.disabled = false; }
  });
  next.addEventListener('click', () => {
    const [instruction, question] = steps[step++];
    hint.textContent = `${step} / ${steps.length}: ${instruction}`;
    $('q').value = question; $('q').focus();
    next.disabled = step >= steps.length;
  });
});
