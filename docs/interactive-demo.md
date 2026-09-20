# Interactive cockpit and guided tour

The first visit opens a ten-step tour. It highlights the instrument cluster, question box, voice controls, answer details, alert area, simulator controls, Scenario Director and Engineering. Back/Next, arrow keys, Escape, optional narration and a persistent **Tour** button are supported. Completion/dismissal is remembered in browser storage. The tour opens panels for explanation, restores them on exit, and never launches a scenario automatically.

## Response paths

- Simple current readings and tyre comparisons are computed by `app/core/conversation.py` from fresh telemetry, including unit follow-ups.
- Open-ended explanations use the configured language model. The cockpit passes up to six recent user/assistant turns; system-role history is rejected. Conversation is kept in browser memory and clears on reload.
- Question matching no longer detects `hi` inside `which` or `this`.
- Critical alerts retain their own banner instead of overwriting unrelated answers.
- The UI shows **Local AI**, **Live reading**, or **Offline fallback** according to the actual response provider.
- Chat does not execute simulator controls. Use the visible controls to change the simulation.

## Local model

This Mac already has Ollama and `qwen2.5-coder:7b` installed. The gateway's local `.env` enables it using:

```ini
AURORA_GATEWAY_LOCAL_ENABLED=true
AURORA_GATEWAY_LOCAL_BASE_URL=http://127.0.0.1:11434/v1
AURORA_GATEWAY_LOCAL_MODEL=qwen2.5-coder:7b
```

Local inference keeps the prompts on this Mac. No new model was downloaded. It is a general instruction-following coding model, not a validated automotive model. The provider has a bounded timeout and falls back to the offline provider if unavailable. First model load is slower than warm requests. Set `AURORA_GATEWAY_LOCAL_ENABLED=false` and restart the gateway to disable it. These settings do not automatically configure a separate Docker deployment.

The conversational model is separate from speech: audio still uses installed macOS/browser voices, not a new speech model.

## Try it

1. Ask **Which tyre is lowest?** Expect a wheel comparison, not a greeting.
2. Ask **What is my speed?** Expect the current reading, even with an unrelated active warning.
3. Ask **What is my current range?**, then **in miles?** Expect a converted reading.
4. Ask **How do I use this demo?**, then **Give me one specific example question to try.** Expect a contextual local-model response.
5. Try **Energy**; inspect the full explanation under **Details**.
6. Open **Tour**, complete it, reload, and verify it stays dismissed until reopened.
7. Open the lab through Engineering, select a scenario manually, and watch the alerts and values react.

## Regression checks

```sh
node tests/test_driving_ui.cjs
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_conversation_relevance.py tests/test_policy.py tests/test_mock_provider.py tests/test_proactive_engine.py tests/test_hud.py tests/test_control_relay.py tests/test_gateway_safety_gate.py tests/test_assistant_prediction_fetch.py tests/test_driving_time.py
```

These tests establish software behavior, not suitability for real vehicle control or road use.
