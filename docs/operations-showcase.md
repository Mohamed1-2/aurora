# AURORA Operations showcase

Open http://localhost:8100/operations/ or choose **Operations** in the cockpit header.

## Three-minute interview journey

1. **Request journey.** Ask “What is the battery level?” Show the direct vehicle-data path. Then choose the cold-weather example to reveal retrieval and a real gateway call. Expand evidence to inspect the exact telemetry snapshot, sources, cache status and token counts. A refusal demonstrates the policy path. The displayed total is pipeline wall time, not mouth-to-ear latency.
2. **Release decisions.** Create a Baseline profile and a Concise profile. Evaluate each against the same 21 synthetic frozen replay turns. Compare the deterministic and model timings separately. Only a current, passing candidate can be activated locally. Demonstrate rollback. The generation settings change; the weights are the same. Baseline and candidate run sequentially, so cold-start differences can affect timings; this is not a statistically controlled performance benchmark.
3. **Failure & recovery.** Run the controlled demo. An isolated synthetic primary times out after 80 ms. The production LlmRouter selects its backup, skips the primary during cooldown, and restores primary routing after cooldown. All four checks and measured times appear on screen.

## Implementation and evidence

- `services/assistant-api/app/core/observability.py`: ContextVar traces isolate concurrent requests. Keeps 100 completed previews in memory; no persistent question/audio log. Stages are actual executed blocks, not invented timing animations. RAG retrieval returning no results is visible; this is not an end-to-end distributed OpenTelemetry deployment.
- `services/assistant-api/app/api/operations.py`: local-only API, atomic local state writes, evaluation jobs, source/fixture fingerprint, evidence hash, actual-provider checks, one-hour activation window and rollback. Single assistant process; not a distributed control plane. Model tags are checked, but immutable model-weight digests are not pinned.
- `scripts/evaluate_replay.py`: production pipeline against controlled telemetry and frozen retrieval excerpts, real pinned gateway calls, unique evaluation cache keys. Records raw exchanges and exact command. Does not benchmark live retrieval ranking.
- `services/llm-gateway/app/operations_demo.py`: isolated providers and router state. This is not a real provider outage. The existing production circuit implementation is exercised as-is; fleet-level concurrency/recovery is not established.
- Gateway cache separates generation parameters and returns a fresh trace ID for a cache hit.
- Release state and immutable run directories live under `artifacts/operations`. Keep evidence artifacts together with their source snapshot. There is no Git revision claim, cloud deployment, GitHub push or trained-weight promotion here.

## What the gate means

All 18 deterministic and 3 model turns pass the existing objective replay checks. All model replies must come from the selected provider/model, be uncached, and finish within 35 seconds each. Source must be unchanged during evaluation. Activation rechecks the source fingerprint, evidence hash, configured model and evidence age. These thresholds only authorize a **local demo configuration**. Human semantic review, noise/voice assessment and automotive readiness remain unvalidated.

A blocked candidate is a useful result: inspect its raw answers and failed checks. Do not weaken the gate just to display a green badge. If the service restarts during evaluation, the UI shows the interrupted state and allows rerunning. If candidate code changes, previous evidence becomes stale.

## Validation commands

Run from the project root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
node --test tests/test_driving_ui.cjs tests/test_voice_contract.cjs
```

The operations cases cover rejection of mismatched/cached/slow model evidence, origin restrictions, concurrent trace isolation, blocked activation, activation/restore/rollback, changed evidence, timeout fallback, circuit skip, recovery and cache identity.
