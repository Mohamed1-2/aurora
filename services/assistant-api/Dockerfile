FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "uvicorn[standard]" "fastapi" "pydantic>=2" "pydantic-settings" \
        "numpy" "prometheus-client" "httpx" "websockets" "pyyaml"

WORKDIR /app

COPY shared /app/shared
RUN pip install --no-cache-dir -e /app/shared

ENV SERVICE_PORT=8100
EXPOSE ${SERVICE_PORT}

COPY services/assistant-api/app /app/app

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD curl -fs http://localhost:${SERVICE_PORT}/healthz >/dev/null || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]