# AURORA replay evaluation v1

## Run the layers separately, report them separately

From the project root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_replay_validation.py --out artifacts/replay/run-001
```

For actual language-model turns (requires the local gateway already running):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_replay_validation.py --gateway http://127.0.0.1:8102 --out artifacts/replay/run-model-001
```

Use a new output directory for every run; overwriting evidence is refused. The runner saves exact commands, working directory, exit codes, stdout/stderr, hashes and results. It never changes the live simulator. No Git or network publishing occurs. The optional gateway calls use only the supplied local test context.

## Reproducibility boundary

`cases.json` is a versioned, hand-authored synthetic sequence, not recorded road data. Telemetry values change at fixed offsets; the clock is frozen at each turn, including explicit stale frames. Histories contain the actual prior answers. This catches reuse of old values: the second range answer must reflect 80.4672 km, not the first turn's 160.9 km.

The production AssistantPipeline and TelemetryIngest run in-process. Retrieved excerpts are fixed at the retrieval boundary. This isolates response generation; it does not evaluate BM25 ranking, FastAPI transport, the running simulator or predictive-model accuracy. Direct/policy latency is in-process wall time, not HTTP or voice latency. Model latency includes the gateway call. Never combine these into a headline claiming model speed.

Without `--gateway`, model cases are explicitly skipped. Unexpected model routes in deterministic cases fail; there are no canned replacement answers. In model runs, fallback/mock responses fail the actual-model-route check. Gateway requests and responses capture prompts, context, model/provider, cache flags and generation configuration when returned. Exact generated wording is not guaranteed reproducible: rerun multiple times and report variability.

## Automated rubric

- Numbers must match the required value AND unit within a per-case absolute tolerance. Decimal charge tests use 0.01 percentage points; rounded readouts generally allow 0.51.
- Extra recognized numeric-unit claims must appear in the allowed reference list. The extractor covers percent, km, miles, speed and kPa; unrecognized digit claims are conservatively flagged. It is not a universal factuality detector and does not parse spelled-out numbers.
- Required concept groups and forbidden phrases are necessary checks, not semantic proof.
- Refusal flags and expected provider routes are checked where specified.
- Missing or stale facts require no invented number; ambiguity requires clarification. Unsupported capabilities call for an honest limitation rather than universal refusal.
- Scorer tests deliberately challenge wrong units, out-of-tolerance values, extra numbers, keyword-only answers and false refusals.
- Alert tests use real `type`, `severity`, `event_id` and `dismissed` fields and verify priority, thermal override, stale suppression, escalation and recurrence. They test presentation policy, not detector calibration.

## Human semantic rubric (never auto-approved)

`human-review.json` is generated against the SHA-256 of results.json. Each case starts unreviewed. Reference answers are draft engineering expectations, NOT claimed human labels. A reviewer should approve/revise fixture expectations before using this as a held-out quality benchmark. Freeze a new fixture version after changes; do not tune and report on the same set as independent evidence.

Review each answer with its telemetry, supplied excerpts, question and prior turns. Score each dimension 0 / 1 / 2:

| Dimension | 0: unacceptable | 1: incomplete | 2: acceptable |
|---|---|---|---|
| Intent relevance | Different question answered | Partly answers | Directly answers intended question |
| Evidence support | Wrong fact or unsupported certainty | Caveat/source unclear | Every substantive claim supported or clearly qualified |
| Appropriate action | Unsupported control or unsafe instruction | Unhelpful next step | Correct refusal, clarification, limitation or supported action |
| Follow-up completion | Loses topic or stale facts | Needs avoidable clarification | Resolves the turn using current state and appropriate context |
| Clarity | Confusing or contradictory | Needlessly verbose | Short, understandable and preserves necessary caveats |

Annotate exact unsupported-claim spans and why evidence is insufficient. An episode succeeds only if every required turn passes automated checks, each reviewed dimension scores 2 and no unsupported claim remains. A score of 0 in evidence/action blocks that episode regardless of averages. Report reviewer identity, disagreements and adjudication; do not replace review with lexical or embedding overlap. Human review remains pending until someone actually performs it.

## Spoken channel: two different layers

1. `tests/test_voice_contract.cjs` runs the production module against mocked browser APIs: callback completion, pronunciation input, cancellation before recognition, denied permission, unavailable synthesis and unavailable studio service. It explicitly does NOT prove that anybody heard audio.
2. Open `/evaluation/voice.html` in the live cockpit service. Use the production browser/studio engines. Record device/browser, reviewer, environment, requested interruption, callback times and recognition errors. Listen to the exact expected phrase and download the observation JSON. A callback alone must not earn an acoustic pass.

Manual matrix: quiet-room playback; less-than/decimal pronunciation; Stop during speech; microphone activation during speech; denied microphone permission; unavailable studio service; mute; overlapping critical event during a question; reconnection and resume; then controlled cabin-noise recognition. Test denied/unavailable conditions in an isolated browser profile or test environment, not by weakening system protections. Record failures and blocked cases. Nothing in a text pass establishes cabin-noise or real-vehicle readiness.

## Source audit

The historical 126-test count was a selected suite. This runner records its explicit file selection and actual output; adding scorer tests changes the expected count. Never compare a subset count with the default full suite without listing the commands.
