function auroraSpeechText(text) {
  return String(text || "")
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/<=|≤/g, " less than or equal to ")
    .replace(/>=|≥/g, " greater than or equal to ")
    .replace(/</g, " less than ").replace(/>/g, " greater than ")
    .replace(/(-?\d+(?:\.\d+)?)\s*%/g, "$1 percent")
    .replace(/\bkm\/h\b/g, "kilometres per hour")
    .replace(/\bkPa\b/g, "kilopascals")
    .replace(/°\s*C\b/g, " degrees Celsius")
    .replace(/°\s*F\b/g, " degrees Fahrenheit")
    .replace(/[()]/g, ", ").replace(/[—–]/g, ", ")
    .replace(/\s+/g, " ").replace(/(?:,\s*)+/g, ", ").trim();
}

"use strict";
/* AURORA voice — STT + TTS for the cockpit.
 *
 * Honest-by-design: AURORA only ever *voices grounded answers and
 * code-decided alerts*; this module never fabricates speech content.
 *
 * TTS strategy (no API key, no remote model on this stack):
 *  - sentence-chunked narration with controlled cadence (robotic-sounding
 *    long utterances are avoided);
 *  - curated voice pick (Daniel/Google/Aria/neural voices first) plus a
 *    picker so the operator can choose the most natural installed voice;
 *  - punctuation-aware pauses and a calm, low-cadence default.
 *
 * STT uses the browser Web Speech API (typed input is always the fallback).
 */
window.AuroraVoice = (() => {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const PREF = [
    /\b(google [a-z ]*english)\b/i, /\b(natural|neural)\b/i, /\bsamantha\b/i,
    /\baria\b/i, /\b(daniel|alex)\b/i, /\bsiri\b/i, /\b(premium|enhanced)\b/i,
    /^en[-_]/i
  ];
  let voices = [];
  let selectedVoice = null;
  let meterCtx = null, meterAnalyser = null, meterStream = null;

  // --- studio engine: macOS `say` neural voices served by the backend ------
  let engine = "browser";             // "browser" | "studio"
  let studioVoices = [];
  let studioDefault = null;
  let studioAvailable = false;
  let studioVoice = null;             // operator-chosen studio voice name

  function _load() {
    if (!('speechSynthesis' in window)) return;
    voices = speechSynthesis.getVoices() || [];
    if (selectedVoice && !voices.some(v => v.name === selectedVoice.name)) selectedVoice = null;
    if (voices.length && !selectedVoice) selectedVoice = _best();
  }
  if ('speechSynthesis' in window) {
    _load();
    speechSynthesis.onvoiceschanged = _load;
  }

  async function initStudio() {
    try {
      const resp = await fetch('/v1/voice', { headers: { 'Accept': 'application/json' } });
      if (!resp.ok) return false;
      const info = await resp.json();
      studioAvailable = !!info.available;
      studioVoices = Array.isArray(info.voices) ? info.voices : [];
      studioDefault = info.default_voice || null;
      if (studioAvailable && engine === "studio" && (!studioVoice || !studioVoices.some(v => v.name === studioVoice))) {
        studioVoice = studioDefault;
      }
      return studioAvailable;
    } catch (e) {
      studioAvailable = false; studioVoices = []; studioDefault = null;
      return false;
    }
  }

  function setEngine(name) {
    const next = name === "studio" ? "studio" : "browser";
    const changed = next !== engine;
    engine = next;
    if (changed) cancel();
    if (next === "studio") initStudio();
    return { engine, available: next === "studio" ? studioAvailable : ('speechSynthesis' in window) };
  }
  function pickStudio(name) {
    if (studioVoices.some(v => v.name === name)) studioVoice = name;
    return studioVoice;
  }
  function studioState() {
    return { engine, studioAvailable, studioVoice: studioVoice || studioDefault, studioVoices: studioVoices.slice() };
  }

  function _rank(v) {
    for (let i = 0; i < PREF.length; i++) if (PREF[i].test(v.name)) return i;
    return PREF.length;
  }
  function _best() {
    const en = voices.filter(v => /^en/i.test(v.lang || '')) || [];
    if (!en.length) return voices[0] || null;
    return en.slice().sort((a, b) => _rank(a) - _rank(b) || (b.localService ? 1 : -1))[0];
  }

  function list() { return voices.slice(); }
  function pick(name) { selectedVoice = voices.find(v => v.name === name) || null; return selectedVoice; }
  function current() { return selectedVoice || _best(); }

  // --- TTS: sentence-chunked narration -------------------------------------
  let gen = 0;
  const activeAudio = new Set();

  function cancel() {
    gen++;
    if ('speechSynthesis' in window) speechSynthesis.cancel();
    activeAudio.forEach(a => { try { a.pause(); } catch (e) { } });
    activeAudio.clear();
  }

  function speak(text, opts = {}) {
    text = auroraSpeechText(text);
    if (engine === "studio" && studioAvailable) return speakStudio(text, opts);
    return speakBrowser(text, opts);
  }

  function _chunks(text) {
    const clean = String(text).replace(/\s+/g, ' ').replace(/\s+([,.;:!?])/g, '$1').trim();
    const raw = clean.match(/.+?(?:[.!?](?=\s|$)|$)/g) || [clean];
    const out = [];
    for (let part of raw) {
      part = part.trim();
      while (part.length > 230) {
        const cut = part.lastIndexOf(' ', 220);
        const at = cut > 90 ? cut : 220;
        out.push(part.slice(0, at).trim());
        part = part.slice(at).trim();
      }
      if (part) out.push(part);
    }
    return out.filter(Boolean).slice(0, 8);
  }

  function speakBrowser(text, opts = {}) {
    const { rate = 0.98, pitch = 1.02, onStart, onEnd } = opts;
    if (!('speechSynthesis' in window) || !text) {
      if (onEnd) onEnd(false);
      return Promise.resolve(false);
    }
    cancel();
    const mine = gen;
    const queue = [String(text).trim()];
    if (!queue.length) { if (onEnd) onEnd(false); return Promise.resolve(false); }
    let started = false;
    return new Promise(resolve => {
      const next = () => {
        if (mine !== gen) return resolve(false);
        const chunk = queue.shift();
        if (!chunk) { onEnd && onEnd(started); return resolve(started); }
        const u = new SpeechSynthesisUtterance(chunk);
        const voice = opts.voice || current();
        if (voice) u.voice = voice;
        u.lang = voice?.lang || 'en-US'; u.rate = Math.min(1.35, Math.max(0.7, rate > 10 ? rate / 175 : rate)); u.pitch = pitch;
        u.voiceURI = voice ? voice.voiceURI : '';
        u.onstart = () => {
          if (mine !== gen) { speechSynthesis.cancel(); return; }
          started = true;
          onStart && onStart();
        };
        u.onend = () => setTimeout(next, opts.gap ?? 70);
        u.onerror = e => { if (mine !== gen) return resolve(false); onEnd && onEnd(false); resolve(false); };
        speechSynthesis.speak(u);
      };
      next();
    });
  }

  // Studio: each sentence is synthesized by the backend's macOS `say` (real
  // neural audio) and played in sequence. A failed fetch stops, honestly —
  // AURORA never pretends it spoke.
  function speakStudio(text, opts = {}) {
    const { rate = 170, pitch = 1.0, onStart, onEnd } = opts;
    if (!text) return Promise.resolve(false);
    cancel();
    const mine = gen;
    const queue = _chunks(text);
    if (!queue.length) return Promise.resolve(false);
    let started = false;
    const voiceName = studioVoice || studioDefault || '';
    const base = `/v1/voice/say?voice=${encodeURIComponent(voiceName)}&rate=${encodeURIComponent(rate)}&pitch=${encodeURIComponent(pitch)}&text=`;
    const play = (url) => new Promise(resolve => {
      if (mine !== gen) return resolve(false);
      const el = new Audio();
      activeAudio.add(el);
      el.preload = 'auto';
      el.src = url;
      let playing = false;
      const timer = setTimeout(() => { el.pause(); activeAudio.delete(el); resolve(false); }, 15000);
      el.oncanplaythrough = () => {
        if (playing) return; playing = true; clearTimeout(timer);
        if (mine !== gen) { el.pause(); return; }
        el.play().then(() => { started = true; onStart && onStart(); }).catch(() => resolve(false));
      };
      el.onended = () => { clearTimeout(timer); activeAudio.delete(el); resolve(true); };
      el.onerror = () => { clearTimeout(timer); activeAudio.delete(el); resolve(false); };
    });
    const next = async () => {
      if (mine !== gen) return false;
      const chunk = queue.shift();
      if (!chunk) { onEnd && onEnd(started); return started; }
      const ok = await play(base + encodeURIComponent(chunk.slice(0, 400)));
      if (!ok) { if (mine !== gen) return false; if (!started) return speakBrowser(text, {...opts, rate: 1.0}); onEnd && onEnd(false); return false; }
      await new Promise(r => setTimeout(r, 120));   // natural inter-sentence pause
      return next();
    };
    return next();
  }

  // --- STT ------------------------------------------------------------------
  let recognizer = null;
  function stopRecognition() {
    const r = recognizer; recognizer = null;
    if (r) { r.onend = null; r.onresult = null; r.onerror = null; try { r.abort(); } catch (e) { } }
  }
  function recognize(opts = {}) {
    const { lang = 'en-US', interim = true, onStart, onInterim, onFinal, onError, onEnd } = opts;
    if (!SR) { onError && onError('unsupported'); return { stop: () => { } }; }
    if (recognizer) { onError && onError('busy'); return { stop: stopRecognition }; }
    cancel(); // Starting the microphone interrupts both speech engines.
    const r = new SR();
    recognizer = r;
    r.lang = lang; r.interimResults = interim; r.continuous = false; r.maxAlternatives = 1;
    r.onstart = () => onStart && onStart();
    r.onresult = e => {
      if (recognizer !== r) return;
      let interimT = '', finalT = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) finalT += t; else interimT += t;
      }
      if (interimT && onInterim) onInterim(interimT);
      if (finalT && onFinal) onFinal(finalT);
    };
    r.onerror = e => { if (recognizer === r) onError && onError(e.error || 'error'); };
    r.onend = () => { if (recognizer === r) { recognizer = null; onEnd && onEnd(); } };
    try { r.start(); } catch (e) { recognizer = null; onError && onError('not-allowed'); }
    return { stop: stopRecognition };
  }

  // --- audio level meter (live microphone energy) --------------------------
  async function initMeter() {
    if (meterAnalyser) return meterAnalyser;
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return null;
      meterStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      meterCtx = new (window.AudioContext || window.webkitAudioContext)();
      meterAnalyser = meterCtx.createAnalyser();
      meterAnalyser.fftSize = 256;
      meterCtx.createMediaStreamSource(meterStream).connect(meterAnalyser);
      return meterAnalyser;
    } catch (e) { return null; }
  }
  function meterLevel() {
    if (!meterAnalyser) return 0;
    const a = new Uint8Array(meterAnalyser.frequencyBinCount);
    meterAnalyser.getByteFrequencyData(a);
    let s = 0; for (const v of a) s += v;
    return Math.min(1, s / a.length / 170);
  }
  async function stopMeter() {
    if (meterCtx) { try { meterCtx.close(); } catch (e) { } meterCtx = null; meterAnalyser = null; }
    if (meterStream) { meterStream.getTracks().forEach(t => t.stop()); meterStream = null; }
  }

  return {
    canSTT: !!SR,
    list, pick, current, speak, cancel,
    initStudio, setEngine, pickStudio, studioState,
    recognize, stopRecognition,
    initMeter, meterLevel, stopMeter
  };
})();