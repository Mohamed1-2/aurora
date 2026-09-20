"""Assistant core pipeline: ground -> prompt -> gateway -> respond."""

from __future__ import annotations

import time
from typing import Any

import httpx

from vehicle_ai.logging_conf import get_logger
from vehicle_ai.schemas import (
    AssistantRequest,
    AssistantResponse,
    ExplainFromTelemetryRequest,
    Grounding,
    GroundingSummary,
    GatewayRequest,
    PolicyLevel,
    PredictionExplanation,
    ProactiveEvent,
    RAGChunk,
    Severity,
    Message,
)
from vehicle_ai.version import PROMPT_VERSION, RAG_INDEX_VERSION

from app.config import settings
from app.core.observability import traced, span, current
from app.core import intents as intent_lib
from app.core.driving_time import estimate_driving_time, render_driving_refusal, render_driving_time
from app.core.policy import classify as policy_classify
from app.core.prompt import build_messages
from app.telemetry.ingest import TelemetryIngest

logger = get_logger("aurora.assistant")

_GATEWAY_TIMEOUT = 40.0
_RAG_TIMEOUT = 10.0
_WHY_PREDICTION_INTENT = "why_prediction"
_DRIVING_TIME_INTENT = "driving_time"
_PREDICTION_EXPLAIN_PATH = "/api/prediction/explain-from-telemetry"


class AssistantPipeline:
    def __init__(self, telemetry: TelemetryIngest) -> None:
        self.telemetry = telemetry
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(_GATEWAY_TIMEOUT))
        from vehicle_ai.ai_metrics import ai_quality_metrics

        self._ai = ai_quality_metrics("assistant")

    def _record(self, **kw: object) -> None:
        try:
            self._ai.record(**kw)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass

    # -- retrieval -----------------------------------------------------------

    async def _retrieve(self, question: str, intents: list[str]) -> list[RAGChunk]:
        if not settings.rag_url:
            return []
        doc_weights: dict[str, float] = {}
        for intent in intents:
            doc_weights.update(intent_lib.INTENT_DOC_WEIGHTS.get(intent, {}))
        try:
            resp = await self._client.post(
                f"{settings.rag_url}/retrieve",
                json={"query": question, "doc_weights": doc_weights or None},
            )
            resp.raise_for_status()
            data = resp.json()
            return [RAGChunk(**d) for d in data.get("top_k", [])]
        except httpx.HTTPError as exc:
            logger.warning("rag retrieve failed", extra={"payload": {"error": str(exc)}})
            return []

    # -- prediction explanation (WS3 auto-fetch) ------------------------------

    def _prediction_windows(self) -> tuple[list[dict], list[dict] | None]:
        """Split the cached frame buffer into to/from prediction windows.

        ``to`` is the most recent prediction window; ``from`` the immediately
        preceding one (only when the buffer is long enough). The simulator does
        the binning/features; the assistant only forwards raw frames.
        """
        frames = list(self.telemetry.frames)
        if len(frames) < settings.prediction_min_frames:
            return [], None
        window_frames = max(1, int(settings.prediction_window_seconds * 10))
        to = frames[-window_frames:]
        fro = frames[-2 * window_frames : -window_frames] if len(frames) >= 2 * window_frames else None
        return to, fro

    async def _fetch_prediction_explanation(self) -> PredictionExplanation | None:
        """Auto-fetch the deterministic explanation for the current prediction.

        Any failure (unreachable sim, too few frames, bad payload) yields None —
        the block is then simply omitted and the LLM is barred from inventing
        causes by the prompt rules.
        """
        to_frames, from_frames = self._prediction_windows()
        if not to_frames:
            return None
        req = ExplainFromTelemetryRequest(to_frames=to_frames, from_frames=from_frames)
        try:
            resp = await self._client.post(
                f"{settings.telemetry_http}{_PREDICTION_EXPLAIN_PATH}",
                json=req.model_dump(),
            )
            resp.raise_for_status()
            return PredictionExplanation.model_validate(resp.json())
        except Exception as exc:  # noqa: BLE001 - explanation is best-effort
            logger.warning(
                "prediction explanation fetch failed",
                extra={"payload": {"error": str(exc), "frames": len(to_frames)}},
            )
            return None

    # -- deterministic driving-time answer -----------------------------------

    async def _answer_driving_time(self, req: AssistantRequest, intents: list[str]) -> AssistantResponse:
        """Answer "how long can I drive?" without an LLM.

        Every number comes from ``driving_time.estimate_driving_time``: grounded
        range + representative speed, or the calibrated discharge forecast. If
        the signals are genuinely missing the estimator refuses honestly.
        """
        start = time.perf_counter()
        telemetry_grounding = self.telemetry.grounding()
        if req.response_style == "concise":
            documents = [d.model_copy(update={"content": d.content[:1200]}) for d in documents[:2]]

        explanation = req.prediction_explanation
        if explanation is None:
            explanation = await self._fetch_prediction_explanation()
        est = estimate_driving_time(telemetry_grounding, explanation)

        used_signals = sorted(
            k
            for k in ("vehicle_speed_kph", "distance_to_empty_km", "battery_soc_pct", "destination_remaining_km", "arrival_eta_min")
            if k in telemetry_grounding.get("signals", {})
        )
        if est.estimable:
            answer, model, refused = render_driving_time(est), "driving-time-estimator", False
        else:
            answer, model, refused = render_driving_refusal(est), "driving-time-refusal", True

        latency_ms = (time.perf_counter() - start) * 1000
        self._record(
            outcome="refused" if refused else "estimated",
            refused=refused,
            refusal_kind="grounding" if refused else "",
            grounded_sources=int(bool(telemetry_grounding)),
            safety_level="read_only",
            latency_ms=latency_ms,
        )
        return AssistantResponse(
            question=req.question,
            answer=answer,
            intents=intents,
            grounding=GroundingSummary(
                used_telemetry=bool(telemetry_grounding),
                telemetry_signals=used_signals,
            ),
            provider="deterministic",
            model=model,
            trace_id="",
            latency_ms=round(latency_ms, 2),
            citations=[],
            risk_level=Severity.INFO,
            session_id=req.session_id,
            prompt_version=PROMPT_VERSION,
            rag_index_version=RAG_INDEX_VERSION,
            refused=refused,
            safety_blocked=[],
            policy_level=PolicyLevel.READ_ONLY,
        )

    # -- main entry -----------------------------------------------------------

    async def ask(self, req: AssistantRequest) -> AssistantResponse:
        return await traced(self, req)

    async def _ask(self, req: AssistantRequest) -> AssistantResponse:
        start = time.perf_counter()
        intents = intent_lib.detect(req.question)

        # --- safety policy gate (runs BEFORE any LLM call) ---
        with span("Policy and intent", intents=intents) as stage:
            policy = policy_classify(req.question, intents)
            stage["decision"] = policy.level.value
        if policy.level.value == "prohibited_action":
            latency_ms = (time.perf_counter() - start) * 1000
            self._record(outcome="prohibited", refused=True, refusal_kind="policy", safety_level="prohibited_action", latency_ms=latency_ms)
            return AssistantResponse(
                question=req.question,
                answer=policy.deterministic_answer,
                intents=intents,
                grounding=GroundingSummary(),
                provider="policy",
                model="deterministic-refusal",
                trace_id="",
                latency_ms=round(latency_ms, 2),
                citations=[],
                risk_level=Severity.CRITICAL,
                session_id=req.session_id,
                prompt_version=PROMPT_VERSION,
                rag_index_version=RAG_INDEX_VERSION,
                refused=True,
                safety_blocked=[policy.reason],
                policy_level=PolicyLevel.PROHIBITED_ACTION,
            )
        if policy.level.value == "restricted_action":
            latency_ms = (time.perf_counter() - start) * 1000
            self._record(outcome="restricted", refused=False, safety_level="restricted_action", latency_ms=latency_ms)
            return AssistantResponse(
                question=req.question,
                answer=policy.deterministic_answer,
                intents=intents,
                grounding=GroundingSummary(),
                provider="policy",
                model="restricted-confirmation",
                trace_id="",
                latency_ms=round(latency_ms, 2),
                citations=[],
                risk_level=Severity.WARNING,
                session_id=req.session_id,
                prompt_version=PROMPT_VERSION,
                rag_index_version=RAG_INDEX_VERSION,
                refused=False,
                safety_blocked=[policy.reason],
                policy_level=PolicyLevel.RESTRICTED_ACTION,
            )

        # Direct factual answers are computed, not paraphrased by a language model.
        if req.response_style == "concise":
            from app.core.conversation import direct_answer, app_guide
            with span("Vehicle snapshot") as stage:
                telemetry = self.telemetry.grounding() if req.ground_telemetry else {}
                stage["timestamp"] = telemetry.get("timestamp")
                stage["signal_count"] = len(telemetry.get("signals", {}))
                stage["signals"] = telemetry.get("signals", {})
            guide = app_guide(req.question)
            direct = (guide, []) if guide else direct_answer(req.question, telemetry, req.conversation)
            if direct is not None:
                text, signals = direct
                return AssistantResponse(question=req.question, answer=text, intents=intents,
                    grounding=GroundingSummary(used_telemetry=bool(signals),telemetry_signals=signals),
                    provider="guide" if guide else "telemetry", model="app-guide" if guide else "direct-facts", latency_ms=round((time.perf_counter()-start)*1000,2),
                    session_id=req.session_id, policy_level=policy.level, risk_level=self._risk_level(telemetry),
                    prompt_version=PROMPT_VERSION,rag_index_version=RAG_INDEX_VERSION)

        # --- deterministic "how long can I drive" path (no LLM) ---------------
        if _DRIVING_TIME_INTENT in intents and req.ground_telemetry:
            return await self._answer_driving_time(req, intents)

        telemetry_grounding: dict[str, Any] = {}
        if req.ground_telemetry:
            with span("Vehicle context") as stage:
                telemetry_grounding = self.telemetry.grounding()
                stage["timestamp"] = telemetry_grounding.get("timestamp")
                stage["signal_count"] = len(telemetry_grounding.get("signals", {}))
                stage["signals"] = telemetry_grounding.get("signals", {})

        if req.response_style == "concise" and telemetry_grounding:
            ts = telemetry_grounding.get("timestamp")
            if not isinstance(ts, (int, float)) or not -5 <= time.time() - ts <= 5:
                telemetry_grounding = {}

        documents: list[RAGChunk] = []
        if req.use_rag and not (req.response_style == "concise" and set(intents) <= {"general", "greeting"}):
            with span("Retrieve knowledge") as stage:
                documents = await self._retrieve(req.question, intents)
                stage["sources"] = [d.source for d in documents]
                stage["document_count"] = len(documents)

        explanation = req.prediction_explanation
        if _WHY_PREDICTION_INTENT in intents and explanation is None:
            explanation = await self._fetch_prediction_explanation()

        events = telemetry_grounding.get("warnings", []) or []
        grounding = Grounding(
            telemetry=telemetry_grounding,
            documents=documents[: settings.max_context_documents],
            events=events,
        )

        messages = build_messages(req.question, grounding, intents, explanation=explanation)
        if req.response_style == "concise":
            # Bound context; the latest grounded user message remains last for gateway policy checks.
            from app.core.conversation import concise_messages
            messages = concise_messages(req.question, telemetry_grounding, documents, req.conversation, intents, explanation)

        g_req = GatewayRequest(
            messages=messages,
            grounding=grounding,
            temperature=0.2,
            max_tokens=110 if req.response_style == "concise" else 600,
            metadata={
                "intents": intents,
                "session_id": req.session_id,
                "vehicle_id": req.vehicle_id,
                "ground_telemetry": req.ground_telemetry,
            },
        )

        release = getattr(self, "release", None)
        if release:
            g_req.provider = release["provider"]
            g_req.model = release["model"]
            g_req.temperature = release["temperature"]
            if req.response_style == "concise":
                g_req.max_tokens = release["max_tokens"]
        trace = current.get()
        if trace:
            g_req.trace_id = trace["id"]
            trace["release"] = release
        with span("LLM gateway", requested_provider=g_req.provider or "configured priority") as stage:
            g_resp = await self._client.post(f"{settings.gateway_url}/v1/complete", json=g_req.model_dump())
            g_resp.raise_for_status()
            data = g_resp.json()
            stage.update(provider=data.get("provider"), model=data.get("model"), cache_hit=data.get("cache_hit"),
                         fallback_chain=data.get("fallback_chain", []), usage=data.get("usage", {}),
                         gateway_latency_ms=data.get("latency_ms"))

        latency_ms = (time.perf_counter() - start) * 1000
        used_telemetry = bool(telemetry_grounding)
        used_docs = bool(documents)

        response = AssistantResponse(
            question=req.question,
            answer=data.get("content", ""),
            intents=intents,
            grounding=GroundingSummary(
                used_telemetry=used_telemetry,
                used_documents=used_docs,
                telemetry_signals=sorted(telemetry_grounding.get("signals", {}).keys()) if used_telemetry else [],
                document_sources=[d.source for d in documents],
                events_used=[w["kind"] for w in (telemetry_grounding.get("warnings", []) or [])],
            ),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            trace_id=data.get("trace_id", ""),
            latency_ms=round(latency_ms, 2),
            citations=[d.source for d in documents],
            risk_level=self._risk_level(telemetry_grounding),
            session_id=req.session_id,
            prompt_version=PROMPT_VERSION,
            rag_index_version=RAG_INDEX_VERSION,
            refused=bool(data.get("refused", False)),
            safety_blocked=list(data.get("safety", {}).get("blocked_reasons", [])),
            policy_level=policy.level,
        )

        self._record(
            outcome="refused" if response.refused else "answered",
            refused=response.refused,
            refusal_kind="safety" if response.safety_blocked else "grounding",
            grounded_sources=int(used_telemetry) + (len(documents) if used_docs else 0),
            safety_level=policy.level.value,
            latency_ms=latency_ms,
            cost_usd=float(data.get("cost_usd") or 0.0),
            cache_hit=bool(data.get("cache_hit")),
            tokens_in=int(data.get("usage", {}).get("prompt_tokens") or 0),
            tokens_out=int(data.get("usage", {}).get("completion_tokens") or 0),
        )
        return response

    async def word_proactive(
        self,
        events: list[ProactiveEvent],
        session_id: str = "",
        vehicle_id: str = "",
    ) -> AssistantResponse:
        """Word already-decided proactive alerts (WS4, never detect here).

        The proactive engine determines the events and their evidence; this
        method only routes them through the LLM as a wording layer. Empty event
        list short-circuits to a deterministic response (no LLM call).
        """
        if not events:
            return AssistantResponse(
                question="(proactive)",
                answer="No active vehicle alerts right now.",
                intents=["proactive"],
                grounding=GroundingSummary(),
                provider="deterministic",
                model="proactive-empty",
                trace_id="",
                latency_ms=0.0,
                risk_level=Severity.INFO,
                session_id=session_id,
                prompt_version=PROMPT_VERSION,
                rag_index_version=RAG_INDEX_VERSION,
                policy_level=PolicyLevel.READ_ONLY,
            )

        start = time.perf_counter()
        telemetry_grounding = self.telemetry.grounding()
        grounding = Grounding(
            telemetry=telemetry_grounding,
            events=[e.model_dump() for e in events[: settings.proactive_max_alerts]],
        )
        from app.core.prompt import build_proactive_messages

        messages = build_proactive_messages(events[: settings.proactive_max_alerts], grounding)
        g_req = GatewayRequest(
            messages=messages,
            grounding=grounding,
            temperature=0.2,
            max_tokens=600,
            metadata={
                "intents": ["proactive"],
                "session_id": session_id,
                "vehicle_id": vehicle_id,
                "proactive": True,
            },
        )
        g_resp = await self._client.post(f"{settings.gateway_url}/v1/complete", json=g_req.model_dump())
        g_resp.raise_for_status()
        data = g_resp.json()
        latency_ms = (time.perf_counter() - start) * 1000

        risk = Severity.INFO
        for e in events:
            if e.severity == Severity.CRITICAL:
                risk = Severity.CRITICAL
                break
            if e.severity == Severity.WARNING:
                risk = Severity.WARNING

        response = AssistantResponse(
            question=f"(proactive) {events[0].type}",
            answer=data.get("content", ""),
            intents=["proactive"],
            grounding=GroundingSummary(
                used_telemetry=bool(telemetry_grounding),
                telemetry_signals=sorted(telemetry_grounding.get("signals", {}).keys()) if telemetry_grounding else [],
                events_used=[e.event_id for e in events],
            ),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            trace_id=data.get("trace_id", ""),
            latency_ms=round(latency_ms, 2),
            citations=[e.title for e in events],
            risk_level=risk,
            session_id=session_id,
            prompt_version=PROMPT_VERSION,
            rag_index_version=RAG_INDEX_VERSION,
            refused=bool(data.get("refused", False)),
            safety_blocked=list(data.get("safety", {}).get("blocked_reasons", [])),
            policy_level=PolicyLevel.READ_ONLY,
        )
        self._record(
            outcome="proactive",
            grounded_sources=len(events),
            safety_level="read_only",
            latency_ms=latency_ms,
            cost_usd=float(data.get("cost_usd") or 0.0),
            cache_hit=bool(data.get("cache_hit")),
        )
        return response

    def live_summary(self) -> dict[str, Any]:
        return self.telemetry.live_summary()

    @staticmethod
    def _risk_level(telemetry: dict) -> Severity:
        for dtc in telemetry.get("dtcs", []):
            if dtc.get("severity") == "critical":
                return Severity.CRITICAL
        for w in telemetry.get("warnings", []):
            sev = w.get("severity")
            if sev == "critical":
                return Severity.CRITICAL
            if sev == "warning":
                return Severity.WARNING
        return Severity.INFO

    async def aclose(self) -> None:
        await self._client.aclose()