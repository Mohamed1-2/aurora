"""WS4 proactive intelligence engine — deterministic alert detection.

Code decides whether a vehicle event exists; the LLM is only ever a wording
layer over an already-decided event. Nothing in this module is probabilistic,
calls an LLM, or reaches for signals that are not present in the live frame.

Lifecycle per (detector, identity) key, e.g. (tire_pressure_leak, rr):
    probation: candidate evidence must persist ``confirm_polls`` consecutive
       polls before the event is emitted (fake-one-shot jitter cannot alert);
    active:    the event is emitted. While active, identical evidence is
       suppressed until ``cooldown_s`` elapses; severity is monotonic — an
       escalation (e.g. battery over-temperature -> critical) bypasses the
       cooldown and re-emits with the higher severity;
    resolved:  a clear signal (no evidence for ``resolve_after_polls`` polls)
       moves the event to ``recent``; a later recurrence re-arms it as a fresh
       episode, so the driver is re-alerted after genuine recovery — never for
       the same unresolved burst.

Strict signal-gating: every detector declares its required signals; if any is
absent at a poll the detector is skipped (recorded in the snapshot) and never
infers. The discharge-risk detector additionally requires full nominal 300 s
coverage (3000 frames @ 10 Hz) before it can use the calibrated model, and it
only ever reports *discharge rate* — never an arrival-SOC / "make it" claim.
ML numbers are surfaced only when the explanation status is ``active``;
``physics_only_fallback`` downgrades the event and suppresses ML wording.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from vehicle_ai.logging_conf import get_logger
from vehicle_ai.schemas import (
    PredictionExplanation,
    ProactiveEvent,
    ProactiveEvidence,
    ProactiveSnapshot,
    ProactiveTrigger,
    Severity,
)

from app.core import proactive_policy as policy

logger = get_logger("aurora.proactive")

_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}
_RECENT_KEEP = 20


@dataclass
class DetectorCandidate:
    detector: str
    key: str
    severity: Severity
    status: str
    title: str
    summary: str
    triggers: list[ProactiveTrigger]
    signals_used: list[str]
    explanation: PredictionExplanation | None = None
    escalate_severity: Severity | None = None
    evidence: list[ProactiveEvidence] = field(default_factory=list)


@dataclass
class _EventState:
    detector: str
    key: str
    event_id: str
    title: str
    active: bool = False
    has_emitted: bool = False
    confirm_count: int = 0
    occurrence_count: int = 0
    first_emitted_ts: float | None = None
    last_emitted_ts: float | None = None
    last_seen_ts: float = 0.0
    missing_polls: int = 0
    resolved: bool = False
    resolved_ts: float | None = None
    dismissed: bool = False
    severity: Severity = Severity.INFO
    status: str = "full"
    summary: str = ""
    signals_used: list[str] = field(default_factory=list)
    triggers: list[ProactiveTrigger] = field(default_factory=list)
    evidence: list[ProactiveEvidence] = field(default_factory=list)
    explanation: PredictionExplanation | None = None


def _slope_per_min(points: list[tuple[float, float]]) -> float:
    """Least-squares slope in signal-units per minute over (ts, value) points."""
    if len(points) < 3:
        return 0.0
    n = len(points)
    xs = [p[0] / 60.0 for p in points]
    ys = [p[1] for p in points]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if abs(den) < 1e-9:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Pure detectors: (ctx) -> list[DetectorCandidate]. ctx is a read-only dict:
#   latest  raw latest frame (dict) | signals  latest signals dict
#   charging bool                   | series {signal: [(ts, value)]}
#   previous_dtc  {code: severity}  | explanation PredictionExplanation|None
#   frame_count  int                | now float
# ---------------------------------------------------------------------------


def d_low_range(ctx: dict[str, Any]) -> list[DetectorCandidate]:
    cfg = policy.detector_config("low_range")
    v = ctx["signals"].get("distance_to_empty_km")
    if v is None:
        return []
    if ctx.get("charging") and cfg["suppress_while_charging"]:
        return []
    if v < cfg["threshold"]:
        return [
            DetectorCandidate(
                detector="low_range",
                key="range",
                severity=cfg["severity"],
                status="full",
                title=cfg["title"],
                summary=f"Remaining range {v:.0f} km is below the {cfg['threshold']:.0f} km alert level.",
                triggers=[
                    ProactiveTrigger(
                        signal="distance_to_empty_km",
                        observed=round(v, 1),
                        threshold=cfg["threshold"],
                        direction="below",
                        unit="km",
                    )
                ],
                signals_used=["distance_to_empty_km"],
            )
        ]
    return []


def d_tire_pressure_leak(ctx: dict[str, Any]) -> list[DetectorCandidate]:
    cfg = policy.detector_config("tire_pressure_leak")
    out: list[DetectorCandidate] = []
    for wheel in policy.TIRE_WHEELS:
        pts = ctx["series"].get(wheel, [])
        if len(pts) < 3:
            continue
        slope = _slope_per_min(pts)
        drop = pts[0][1] - pts[-1][1]
        if slope < cfg["leak_slope_kpa_per_min"] and drop > cfg["min_drop_kpa"]:
            rate_signal = f"{wheel}_slope_per_min"  # e.g. tire_pressure_rr_kpa_slope_per_min (kPa/min)
            out.append(
                DetectorCandidate(
                    detector="tire_pressure_leak",
                    key=wheel,
                    severity=cfg["severity"],
                    status="full",
                    title=cfg["title"],
                    summary=(
                        f"Pressure on {wheel.removeprefix('tire_pressure_').removesuffix('_kpa')} "
                        f"is falling at {abs(slope):.0f} kPa/min ({drop:.1f} kPa over {cfg['window_s']:.0f} s)."
                    ),
                    triggers=[
                        ProactiveTrigger(
                            signal=rate_signal,
                            observed=round(slope, 2),
                            threshold=cfg["leak_slope_kpa_per_min"],
                            direction="below",
                            unit="kPa/min",
                            window_seconds=cfg["window_s"],
                            samples=len(pts),
                        )
                    ],
                    signals_used=[wheel],
                    evidence=[
                        # Raw pressure (kPa) and its change rate (kPa/min) are
                        # kept as separate evidence fields — never conflated.
                        ProactiveEvidence(
                            signal=wheel,
                            value=round(pts[-1][1], 1),
                            unit="kPa",
                            label="current pressure",
                        ),
                        ProactiveEvidence(
                            signal=rate_signal,
                            value=round(slope, 2),
                            unit="kPa/min",
                            label="pressure change rate",
                        ),
                    ],
                )
            )
    return out


def _series_mean(ctx: dict[str, Any], signal: str) -> tuple[float | None, int]:
    pts = ctx["series"].get(signal, [])
    if not pts:
        return None, 0
    return statistics.fmean(p[1] for p in pts), len(pts)


def d_battery_overtemperature(ctx: dict[str, Any]) -> list[DetectorCandidate]:
    cfg = policy.detector_config("battery_overtemperature")
    mean, n = _series_mean(ctx, "battery_temp_c")
    if mean is None:
        return []
    if mean > cfg["max_temp_c"]:
        escalate = (
            Severity.CRITICAL
            if mean > cfg["escalate_temp_c"]
            else None
        )
        return [
            DetectorCandidate(
                detector="battery_overtemperature",
                key="battery",
                severity=cfg["severity"],
                escalate_severity=escalate,
                status="full",
                title=(
                    "Battery overheating — thermal risk"
                    if escalate == Severity.CRITICAL
                    else cfg["title"]
                ),
                summary=(
                    f"Battery temperature averaged {mean:.1f} °C over the last "
                    f"{cfg['window_s']:.0f} s (limit {cfg['max_temp_c']:.0f} °C)."
                ),
                triggers=[
                    ProactiveTrigger(
                        signal="battery_temp_c",
                        observed=round(mean, 1),
                        threshold=cfg["max_temp_c"],
                        direction="above",
                        unit="degC",
                        window_seconds=cfg["window_s"],
                        samples=n,
                    )
                ],
                signals_used=["battery_temp_c"],
            )
        ]
    return []


def d_charging_stall(ctx: dict[str, Any]) -> list[DetectorCandidate]:
    cfg = policy.detector_config("charging_stall")
    if not ctx.get("charging"):
        return []
    rate, n = _series_mean(ctx, "charge_rate_kw")
    if rate is None:
        return []
    if rate < cfg["max_rate_kw"]:
        return [
            DetectorCandidate(
                detector="charging_stall",
                key="charge",
                severity=cfg["severity"],
                status="full",
                title=cfg["title"],
                summary=(
                    f"Vehicle is plugged in but charging power averaged "
                    f"{rate:.2f} kW over {cfg['window_s']:.0f} s (below "
                    f"{cfg['max_rate_kw']:.1f} kW)."
                ),
                triggers=[
                    ProactiveTrigger(
                        signal="charge_rate_kw",
                        observed=round(rate, 2),
                        threshold=cfg["max_rate_kw"],
                        direction="below",
                        unit="kW",
                        window_seconds=cfg["window_s"],
                        samples=n,
                    )
                ],
                signals_used=["charge_rate_kw", "charging"],
            )
        ]
    return []


def d_dtc_escalation(ctx: dict[str, Any]) -> list[DetectorCandidate]:
    cfg = policy.detector_config("dtc_escalation")
    latest = ctx.get("latest") or {}
    if "active_dtcs" not in latest:
        return []  # field absent: strictly no DTC inference
    dtcs = latest["active_dtcs"] or []
    prev = ctx.get("previous_dtc", {})
    out: list[DetectorCandidate] = []
    for item in dtcs:
        code = item.get("code")
        if not code:
            continue
        sev = item.get("severity", "warning")
        sev = Severity.CRITICAL if sev == "critical" else Severity.WARNING
        desc = item.get("description", "")
        # Persistence candidate: fires while the code is active (cooldown +
        # suppression in the engine decide how often it is surfaced).
        out.append(
            DetectorCandidate(
                detector="dtc_escalation",
                key=code,
                severity=sev,
                status="full",
                title=cfg["title"],
                summary=f"Diagnostic trouble code {code} active ({desc}).",
                triggers=[
                    ProactiveTrigger(
                        signal="active_dtcs",
                        observed=None,
                        threshold=1.0,
                        direction="above",
                        unit="",
                    )
                ],
                signals_used=["active_dtcs"],
            )
        )
        # Severity escalation (e.g. warning -> critical) re-emits immediately.
        old = prev.get(code)
        if old is not None and (_SEVERITY_RANK[sev.value] > _SEVERITY_RANK[old]) and sev == Severity.CRITICAL:
            out.append(
                DetectorCandidate(
                    detector="dtc_escalation",
                    key=code,
                    severity=Severity.CRITICAL,
                    escalate_severity=Severity.CRITICAL,
                    status="full",
                    title=cfg["title"],
                    summary=f"Trouble code {code} escalated to critical ({desc}).",
                    triggers=[
                        ProactiveTrigger(
                            signal="active_dtcs",
                            observed=None,
                            threshold=1.0,
                            direction="above",
                            unit="",
                        )
                    ],
                    signals_used=["active_dtcs"],
                )
            )
    return out


def d_prediction_discharge_risk(ctx: dict[str, Any]) -> list[DetectorCandidate]:
    cfg = policy.detector_config("prediction_discharge_risk")
    exp = ctx.get("explanation")
    if exp is None:
        return []
    if ctx.get("frame_count", 0) < cfg["min_frames"]:
        return []  # insufficient nominal coverage: no inference
    drop = exp.to_prediction.calibrated_soc_drop_pct
    if drop < cfg["min_soc_drop_pct"]:
        return []
    status = "full" if exp.status == "active" else "fallback"
    severity = cfg["severity"] if status == "full" else cfg["fallback_severity"]
    summary = (
        f"Predicted battery discharge {drop:.2f} SOC pts over the 300 s window "
        f"(threshold {cfg['min_soc_drop_pct']:.0f})."
    )
    if status == "fallback":
        summary += " Physics estimate only — the ML model is unavailable."
    return [
        DetectorCandidate(
            detector="prediction_discharge_risk",
            key="discharge",
            severity=severity,
            status=status,
            title=cfg["title"],
            summary=summary,
            triggers=[
                ProactiveTrigger(
                    signal="calibrated_soc_drop_pct",
                    observed=round(drop, 2),
                    threshold=cfg["min_soc_drop_pct"],
                    direction="above",
                    unit="SOC pts / 300 s",
                    window_seconds=cfg["window_s"],
                    samples=ctx.get("frame_count"),
                )
            ],
            signals_used=policy.PREDICTION_INPUT_SIGNALS,
            explanation=exp,
        )
    ]


DETECTOR_FUNCS: dict[str, Callable[[dict[str, Any]], list[DetectorCandidate]]] = {
    "low_range": d_low_range,
    "tire_pressure_leak": d_tire_pressure_leak,
    "battery_overtemperature": d_battery_overtemperature,
    "charging_stall": d_charging_stall,
    "dtc_escalation": d_dtc_escalation,
    "prediction_discharge_risk": d_prediction_discharge_risk,
}


class ProactiveEvaluator:
    """Stateful evaluator over the assistant's telemetry buffer (no LLM)."""

    def __init__(
        self,
        telemetry,
        vehicle_id: str = "ORION-X-TEST-01",
        poll_interval_s: float = 5.0,
        get_explanation: Callable[[], Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.telemetry = telemetry
        self.vehicle_id = vehicle_id
        self.poll_interval_s = poll_interval_s
        self.get_explanation = get_explanation
        self._clock = clock or time.time
        self._states: dict[tuple[str, str], _EventState] = {}
        self._previous_dtc: dict[str, str] = {}
        self._last_eligible: list[str] = []
        self._last_skipped: dict[str, list[str]] = {}
        self.stats = {"polls": 0, "emitted": 0, "suppressed": 0, "resolved": 0, "escalated": 0}

    # -- id / hash ----------------------------------------------------------

    def _event_id(self, detector: str, key: str) -> str:
        raw = f"{detector}:{key}:{self.vehicle_id}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    # -- context -------------------------------------------------------------

    def _gate(self, name: str, cfg: dict) -> list[str]:
        """Return the list of missing required signals ([] == gate passes)."""
        missing = [s for s in cfg.get("required_signals", []) if not self.telemetry.has_signal(s)]
        return missing

    def _series_for(self, cfg: dict) -> dict[str, list[tuple[float, float]]]:
        out: dict[str, list[tuple[float, float]]] = {}
        window = cfg.get("window_s")
        for sig in cfg.get("required_signals", []):
            out[sig] = self.telemetry.signal_series(sig, max_age_s=window or 45.0)
        return out

    def _build_ctx(self, explanation: PredictionExplanation | None) -> dict[str, Any]:
        grounding = self.telemetry.grounding()
        latest = list(self.telemetry.frames)[-1] if self.telemetry.frames else {}
        return {
            "grounding": grounding,
            "latest": latest,
            "signals": grounding.get("signals", {}) or {},
            "charging": bool(grounding.get("charging", False)),
            "series": {},
            "previous_dtc": dict(self._previous_dtc),
            "explanation": explanation,
            "frame_count": len(self.telemetry.frames),
            "now": time.time(),
        }

    # -- poll ----------------------------------------------------------------

    async def poll(self) -> ProactiveSnapshot:
        self.stats["polls"] += 1
        now = self._clock()
        explanation: PredictionExplanation | None = None
        discharge_cfg = policy.detector_config("prediction_discharge_risk")

        eligible: list[str] = []
        skipped: dict[str, list[str]] = {}
        candidates: list[DetectorCandidate] = []
        pre_gate: dict[str, list[str]] = {}

        for name, cfg in policy.DETECTORS.items():
            missing = self._gate(name, cfg)
            pre_gate[name] = missing
            if missing:
                skipped[name] = missing
            else:
                eligible.append(name)

        # Prediction detector fetches the deterministic explanation only when
        # its signal gate passes AND nominal coverage is met.
        if "prediction_discharge_risk" in eligible and self.get_explanation is not None:
            if len(self.telemetry.frames) >= discharge_cfg["min_frames"]:
                try:
                    fetched = self.get_explanation()
                    if inspect.isawaitable(fetched):
                        fetched = await asyncio.wait_for(fetched, timeout=10.0)
                    explanation = fetched
                except Exception as exc:  # noqa: BLE001 - best effort
                    logger.warning("proactive prediction fetch failed", extra={"payload": {"error": str(exc)}})
            else:
                skipped["prediction_discharge_risk"] = ["<300 s coverage: %d frames>" % len(self.telemetry.frames)]

        base_ctx = self._build_ctx(explanation)
        for name in eligible:
            cfg = policy.detector_config(name)
            ctx = dict(base_ctx)
            if cfg.get("required_signals"):
                ctx["series"] = self._series_for(cfg)
            candidates.extend(DETECTOR_FUNCS[name](ctx))

        self._update(candidates, now)
        self._last_eligible, self._last_skipped = eligible, skipped
        return self.snapshot(now, eligible=eligible, skipped=skipped)

    # -- state machine ---------------------------------------------------------

    def _update(self, candidates: list[DetectorCandidate], now: float) -> None:
        seen: set[tuple[str, str]] = set()

        for cand in candidates:
            sid = (cand.detector, cand.key)
            seen.add(sid)
            cfg = policy.detector_config(cand.detector)
            st = self._states.get(sid)

            if st is None:
                st = _EventState(
                    detector=cand.detector,
                    key=cand.key,
                    event_id=self._event_id(cand.detector, cand.key),
                    title=cand.title,
                    last_seen_ts=now,
                )
                self._states[sid] = st

            st.last_seen_ts = now
            st.status = cand.status
            st.title = cand.title
            st.summary = cand.summary
            st.signals_used = cand.signals_used
            st.triggers = list(cand.triggers)
            st.evidence = list(cand.evidence)
            st.explanation = cand.explanation

            if st.active:
                # Monotonic severity: an escalation re-emits immediately.
                if cand.escalate_severity and _SEVERITY_RANK[cand.escalate_severity.value] > _SEVERITY_RANK[st.severity.value]:
                    st.severity = cand.escalate_severity
                    st.occurrence_count += 1
                    st.last_emitted_ts = now
                    st.has_emitted = True
                    self.stats["escalated"] += 1
                    self.stats["emitted"] += 1
                    continue
                # Duplicate suppression while within the cooldown.
                since = now - (st.last_emitted_ts or 0.0)
                if since < cfg["cooldown_s"]:
                    self.stats["suppressed"] += 1
                    continue
                # Cooldown elapsed and evidence persists -> repeat emission.
                st.occurrence_count += 1
                st.last_emitted_ts = now
                st.has_emitted = True
                self.stats["emitted"] += 1
                continue

            # Probation (fresh episode, or re-armed after recovery).
            if st.resolved:
                st.resolved = False
                st.resolved_ts = None
                st.dismissed = False
                st.confirm_count = 0
            if st.confirm_count >= cfg["confirm_polls"] - 1:
                st.confirm_count += 1
                st.active = True
                st.has_emitted = True
                st.occurrence_count += 1
                st.first_emitted_ts = st.first_emitted_ts or now
                st.last_emitted_ts = now
                st.severity = cand.escalate_severity or cand.severity
                self.stats["emitted"] += 1
            else:
                st.confirm_count += 1
                st.severity = cand.escalate_severity or cand.severity

        # Events that did NOT fire this poll: resolve active ones, decay probation.
        for sid, st in list(self._states.items()):
            if st.resolved and not st.active:
                continue
            if sid in seen:
                st.missing_polls = 0
                continue
            if st.active:
                st.missing_polls += 1
                cfg = policy.detector_config(st.detector)
                if st.missing_polls >= cfg["resolve_after_polls"]:
                    st.active = False
                    st.resolved = True
                    st.resolved_ts = now
                    self.stats["resolved"] += 1
            elif st.confirm_count > 0 and not st.has_emitted:
                st.confirm_count = 0  # evidence vanished before confirmation

        self._previous_dtc = {
            (item.get("code") or ""): (item.get("severity") or "warning")
            for item in (list(self.telemetry.frames)[-1].get("active_dtcs", []) if self.telemetry.frames else [])
        }

    # -- output ------------------------------------------------------------

    def ack(self, event_ids: list[str]) -> None:
        for sid, st in self._states.items():
            if st.event_id in event_ids:
                st.dismissed = True

    def top_active(self, count: int = 1) -> list[ProactiveEvent]:
        active = self._events([st for st in self._states.values() if st.active and not st.dismissed])
        active.sort(key=lambda e: (_SEVERITY_RANK[e.severity.value], e.last_seen_ts), reverse=True)
        return active[: max(0, count)]

    def snapshot(self, now: float | None = None, eligible: list[str] | None = None, skipped: dict[str, list[str]] | None = None) -> ProactiveSnapshot:
        now = now or self._clock()
        eligible = self._last_eligible if eligible is None else eligible
        skipped = self._last_skipped if skipped is None else skipped
        active = self._events([st for st in self._states.values() if st.active and not st.dismissed])
        active.sort(key=lambda e: (_SEVERITY_RANK[e.severity.value], e.last_seen_ts), reverse=True)
        recent = self._events(
            sorted(
                (st for st in self._states.values() if st.has_emitted),
                key=lambda st: st.last_emitted_ts or 0.0,
                reverse=True,
            )
        )[: _RECENT_KEEP]
        return ProactiveSnapshot(
            generated_at=now,
            vehicle_id=self.vehicle_id,
            active=active,
            recent=recent,
            detectors_eligible=eligible,
            detectors_skipped_no_signal=skipped,
            poll_interval_s=self.poll_interval_s,
        )

    @staticmethod
    def _events(states: list[_EventState]) -> list[ProactiveEvent]:
        return [
            ProactiveEvent(
                event_id=s.event_id,
                type=s.detector,
                severity=s.severity,
                status=s.status,
                title=s.title,
                summary=s.summary,
                triggers=s.triggers,
                evidence=s.evidence,
                explanation=s.explanation,
                signals_used=s.signals_used,
                first_seen_ts=s.first_emitted_ts or s.last_seen_ts,
                last_seen_ts=s.last_seen_ts,
                occurrence_count=s.occurrence_count,
                dismissed=s.dismissed,
            )
            for s in states
        ]