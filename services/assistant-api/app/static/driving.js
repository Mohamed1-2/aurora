"use strict";
/* Reply policy: fresh warnings follow the answer; fresh critical events lead.
 * A critical battery thermal event replaces routine replies with a stop message.
 * Refusals remain intact. Repeated automatic speech is controlled by event identity.
 */
const AuroraDriving = (() => {
  const clean = text => String(text || "").replace(/【[^】]*】/g, "").replace(/\*\*|__/g, "").replace(/\s+/g, " ").trim();
  const brief = text => {
    const sentences = clean(text).match(/.+?(?:[.!?](?=\s|$)|$)/g) || [];
    const result = sentences.slice(0, 2).join("").trim();
    if (result.split(/\s+/).length <= 45) return result;
    const first = (sentences[0] || "").trim();
    return first || clean(text);
  };
  const fresh = frame => Number.isFinite(frame?.timestamp) && Date.now() / 1000 - frame.timestamp < 5 && Date.now() / 1000 - frame.timestamp > -5;

  function priorityAlert(frame, alerts) {
    if (!fresh(frame)) return null;
    const events = (alerts || []).filter(e => !e.dismissed && ['warning','critical'].includes(e.severity));
    const e = events.find(e => e.severity === 'critical' && /battery_overtemp/.test(e.type || e.detector || e.kind || ''))
      || events.find(e => e.severity === 'critical') || events[0];
    if (!e) return null;
    const thermal = e.severity === 'critical' && /battery_overtemp/.test(e.type || e.detector || e.kind || '');
    return {...e, thermal, speech: thermal
      ? 'Critical battery temperature warning. Stop when safe and check the vehicle warning display.'
      : clean(`${e.title || 'Vehicle warning'}. ${e.summary || ''}`)};
  }
  function answer(question, response, frame, alerts) {
    const text = response.refused || response.policy_level === 'restricted_action'
      || /tell me more|more detail|explain in detail/i.test(question) ? clean(response.answer) : brief(response.answer);
    const event = priorityAlert(frame, alerts);
    if (!event) return text;
    if (event.thermal && !response.refused && response.policy_level !== 'restricted_action') return event.speech;
    if (text.includes(event.title)) return text;
    return event.severity === 'critical' ? `${event.speech} ${text}` : `${text} ${event.speech}`;
  }
  function createAlertDelivery() {
    const delivered = new Set();
    return {
      next(frame, alerts) {
        if (!fresh(frame)) return null;
        const active = new Set((alerts || []).map(e => `${e.event_id || e.type || e.detector || e.kind}:${e.severity}`));
        for (const key of delivered) if (!active.has(key)) delivered.delete(key);
        const event = priorityAlert(frame, alerts);
        if (!event || event.severity !== 'critical') return null;
        const key = `${event.event_id || event.type || event.detector || event.kind}:${event.severity}`;
        if (delivered.has(key)) return null;
        return {...event, deliveryKey:key};
      },
      mark(event) { delivered.add(event.deliveryKey || `${event.event_id || event.type || event.detector || event.kind}:${event.severity}`); },
      reset() { delivered.clear(); }
    };
  }

  function energy(data) {
    const exp = data.explanation || {};
    const drop = exp.to_prediction?.calibrated_soc_drop_pct;
    if (!Number.isFinite(drop)) return brief(data.narrative);
    const causes = (exp.top_causes || []).filter(c => Number.isFinite(c.contribution_soc_pct)).sort((a,b) => Math.abs(b.contribution_soc_pct) - Math.abs(a.contribution_soc_pct));
    const lead = causes[0];
    const label = {"Aerodynamic drag":"Air resistance", "Rolling resistance":"Tyre rolling resistance", "Inertia / acceleration":"Acceleration", "Auxiliary load":"Accessories"}[lead?.label] || lead?.label;
    const estimate = drop >= 0 ? `At this pace, estimated battery use is ${drop.toFixed(1)} percentage points over five minutes.` : `At this pace, the battery is estimated to gain ${Math.abs(drop).toFixed(1)} percentage points over five minutes.`;
    return estimate + (label ? ` ${label} is the biggest contributor to this estimate.` : "");
  }
  return { clean, brief, fresh, answer, energy, priorityAlert, createAlertDelivery };
})();
if (typeof module !== "undefined") module.exports = AuroraDriving;
