.PHONY: install services stop logs eval eval-e2e baseline test lint build up down docker-eval demo data data-v2 finetune finetune-real compare compare-v2 clean

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

## install: create venv and editable-install shared package
install:
	$(PY) -m pip install --upgrade pip
	$(PIP) install -e shared
	$(PIP) install -r requirements-dev.txt 2>/dev/null || true

## services: run the full stack locally (background uvicorn, each on 8100-8104)
services: stop
	$(MAKE) --no-print-directory _start-svc

_start-svc:
	cd services/telemetry-simulator && (../../$(PY) -m uvicorn app.main:app --port 8101 >/tmp/aurora-telemetry.log 2>&1 &)
	cd services/llm-gateway && (../../$(PY) -m uvicorn app.main:app --port 8102 >/tmp/aurora-gateway.log 2>&1 &)
	cd services/rag-service && (../../$(PY) -m uvicorn app.main:app --port 8103 >/tmp/aurora-rag.log 2>&1 &)
	cd services/assistant-api && (../../$(PY) -m uvicorn app.main:app --port 8100 >/tmp/aurora-assistant.log 2>&1 &)
	cd services/monitoring && (../../$(PY) -m uvicorn app.main:app --port 8104 >/tmp/aurora-monitoring.log 2>&1 &)
	@echo "stack starting... check /tmp/aurora-*.log"; sleep 3; curl -fs http://localhost:8100/healthz && echo " assistant-api ready"

## stop: kill any locally running aurora services
stop:
	-pkill -f "uvicorn app.main" 2>/dev/null || true
	@echo "stopped"

## logs: tail local service logs
logs:
	tail -f /tmp/aurora-telemetry.log /tmp/aurora-gateway.log /tmp/aurora-rag.log /tmp/aurora-assistant.log /tmp/aurora-monitoring.log

## eval: gate the assistant against the golden set (replay-mode assistant on 8105)
eval:
	cd services/evaluation && ../../$(PY) run.py --assistant http://127.0.0.1:8105 --environment ci

eval-e2e: services
	cd services/assistant-api && (AURORA_ASSISTANT_TELEMETRY_WS=ws://127.0.0.1:59999/ws AURORA_ASSISTANT_TELEMETRY_REPLAY=../evaluation/datasets/telemetry_snapshot.json ../../$(PY) -m uvicorn app.main:app --port 8105 >/tmp/aurora-assistant-replay.log 2>&1 &)
	@sleep 6; cd services/evaluation && ../../$(PY) run.py --assistant http://127.0.0.1:8105 --environment ci

## baseline: regenerate drift baseline from the simulator (seed 42 deterministic)
baseline:
	$(PY) scripts/gen_baseline.py

## test: run the unit/integration test suite
test:
	$(PY) -m pytest -q

## lint: syntax + import smoke check across packages
lint:
	$(PY) -m compileall -q shared services scripts
	$(PY) -m pytest tests/test_imports.py -q

## demo: end-to-end interactive demo (asks a scripted battery of questions)
demo:
	$(PY) scripts/demo.py

## data: build the realistic production-shaped dataset (curated + validated + versioned)
## Optionally ground TelemetryContext in real UDS field data:
##   make data TELEMETRY_DIR=/path/to/electric-vehicle-uds-dataset-main
data:
	$(PY) services/evaluation/data/generate.py $(if $(TELEMETRY_DIR),--telemetry-dir $(TELEMETRY_DIR),)

## data-v2: build the AURORA v2 corpus from real public telemetry (TUM driving + EPFL charging)
##   make data-v2 TUM_DIR=/path/to/electric-vehicle-uds-dataset-main EPFL_DIR=/path/to/Level-3-EV-charging-dataset-main
##   make data-v2                   (uses default Downloads paths)
data-v2:
	$(PY) services/evaluation/data/v2/build.py \
	  --tum-uds $(if $(TUM_DIR),$(TUM_DIR),/Users/mohamedaldofani/Downloads/electric-vehicle-uds-dataset-main) \
	  --epfl   $(if $(EPFL_DIR),$(EPFL_DIR),/Users/mohamedaldofani/Downloads/Level-3-EV-charging-dataset-main) \
	  --out artifacts/dataset/v2 \
	  --max-windows $(if $(V2_WINDOWS),$(V2_WINDOWS),30000) \
	  --max-sessions $(if $(V2_SESSIONS),$(V2_SESSIONS),1129)

## finetune: build data + prepare the PEFT/LoRA dataset + adapter (CI-safe simulation by default)
finetune: data
	$(PY) services/evaluation/finetune.py

finetune-real:
	AURORA_FINETUNE_REAL=1 $(PY) services/evaluation/finetune.py real

## compare: head-to-head baseline vs fine-tuned assistant on the identical golden set
## requires the baseline URI (default live :8100) and the adapter-serving URI.
compare:
	cd services/evaluation && ../../$(PY) compare.py --baseline http://127.0.0.1:8100 --finetuned http://127.0.0.1:8105 --method lora-r8-alpha16

## compare-v2: system/gateway-level v2 evaluation — peft (LoRA) vs peft-base
## on the identical v2 held-out rows via the LLM gateway (experiment 2; the
## model-level experiment 1 lives in colab/). Smoke mode first, then full.
compare-v2:
	cd services/evaluation && ../../$(PY) compare_v2.py --gateway http://localhost:8102 --mode smoke --out ../../artifacts/eval/v2-gateway
	cd services/evaluation && ../../$(PY) compare_v2.py --gateway http://localhost:8102 --mode full --out ../../artifacts/eval/v2-gateway

# ---- containerised runtime ----
build:
	docker compose build

up:
	docker compose up -d --build

down:
	docker compose down

docker-eval:
	docker compose run --rm eval --assistant http://assistant-api:8100

clean:
	rm -rf .pytest_cache services/**/__pycache__ shared/vehicle_ai/__pycache__