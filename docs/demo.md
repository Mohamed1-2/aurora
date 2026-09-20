# AURORA — Demo walkthrough

Everything below runs against a fully live stack. Nothing needs a cloud key.

## 0. Start the stack

```bash
# option A — containers
docker compose up -d --build
# option B — host processes
make services
```

Dashboard: open http://localhost:8100 (live vehicle cards + chat).

## 1. Ask the assistant

```bash
curl -s -X POST http://localhost:8100/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is my current range and state of charge?"}'
```

Notice the response carries `intents`, `grounding.telemetry_signals`,
`citations` (`【doc：section】`), `provider="mock"`, `refused=false`, plus
`prompt_version` / `rag_index_version` for full provenance.

Refusal honesty:

```bash
curl -s -X POST http://localhost:8100/v1/ask -H 'Content-Type: application/json' \
  -d '{"question":"Tell me a joke"}'        # refused: true
```

## 2. Walk every layer

```bash
make demo
```

1. health of all five services
2. live telemetry window (SOC/range/speed/motor/coolant/battery temp)
3. BM25 retrieval for “low tyre pressure” → 04-tires-and-wheels.md
4. gateway provider info + grounded completions with latency/cost/cache
5. assistant answers (battery / diagnostics / summary) + refusal
6. drift status + **injected drift scenario** → PSI alert fires within seconds
7. eval-gate reminder

## 3. Injection scenarios

```bash
curl -s -X POST http://localhost:8101/api/scenario \
  -H 'Content-Type: application/json' \
  -d '{"type":"battery_heat","duration_s":40,"intensity":0.8}'
curl -s http://localhost:8104/monitor/drift | python -m json.tool
```

Available: `none, leak, dtc, drift, charge, battery_heat`.

## 4. Run the acceptance gate

```bash
# containerised (same image CI runs)
docker compose run --rm eval --assistant http://assistant-eval:8100

# or local replay-mode harness
make eval-e2e
```

Expect `GATE: ✓ ACCEPTED — ship to production (pass rate 1.0 ≥ 0.8)`.

## 5. Enable a cloud provider (optional)

```bash
cp .env.example .env   # fill AURORA_GATEWAY_AZURE_OPENAI_* / _OPENAI_*
AURORA_GATEWAY_PROVIDER_PRIORITY=azure-openai,openai,mock
```

The gateway now routes through the model chain automatically, with circuit-breaker
fallback and a per-request cost ledger at `/stats`.

## 6. Observability

```bash
curl -s http://localhost:8104/service-health | python -m json.tool
curl -s http://localhost:8104/insights | python -m json.tool
curl -s http://localhost:8101/metrics | grep aurora_      # Prometheus text fmt
curl -s http://localhost:8102/stats                       # cache + spend
```