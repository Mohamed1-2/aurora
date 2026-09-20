FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "fastapi" "pydantic>=2" "pydantic-settings" \
        "numpy" "prometheus-client" "httpx" "websockets" "pyyaml"

WORKDIR /app

COPY shared /app/shared
RUN pip install --no-cache-dir -e /app/shared

COPY services/evaluation /app/evaluation
WORKDIR /app/evaluation

ENV PYTHONPATH=/app
ENTRYPOINT ["python", "run.py"]