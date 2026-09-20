"""Create demo Dockerfiles and Kubernetes YAML from the repository's existing files."""
import os
from pathlib import Path
import yaml

SERVICES = ['telemetry-simulator', 'llm-gateway', 'rag-service', 'assistant-api']
OUT = Path('work/azure')
OUT.mkdir(parents=True, exist_ok=True)
for service in SERVICES:
    text = Path(f'services/{service}/Dockerfile').read_text()
    # Preserve repository depth required by Path(__file__).parents[4].
    text = text.replace('WORKDIR /app', f'WORKDIR /app/services/{service}')
    text = text.replace(f'COPY services/{service}/app /app/app',
                        f'COPY services/{service}/app /app/services/{service}/app')
    if service == 'telemetry-simulator':
        text += '\nCOPY services/calibration /app/services/calibration\n'
    (OUT / f'Dockerfile.{service}').write_text(text)

documents = []
files = ['00-namespace.yaml', '10-telemetry-simulator.yaml', '11-llm-gateway.yaml',
         '12-rag-service.yaml', '13-assistant-api.yaml']
for filename in files:
    for doc in yaml.safe_load_all((Path('infra/k8s/base') / filename).read_text()):
        if not doc or doc['kind'] in ('Namespace', 'Secret', 'HorizontalPodAutoscaler'):
            continue
        if doc['kind'] == 'ConfigMap':
            doc['data']['AURORA_GATEWAY_PROVIDER_PRIORITY'] = 'mock'
            doc['data']['AURORA_GATEWAY_PROVIDER_WEIGHTS'] = 'mock:1'
            doc['data']['AURORA_ASSISTANT_PROACTIVE_ENABLED'] = 'false'
            doc['data'].pop('AURORA_ASSISTANT_PROMPT_VERSION', None)
        if doc['kind'] == 'Deployment':
            doc['spec']['replicas'] = 1
            # Demo state is process-local. Avoid overlapping revisions.
            doc['spec']['strategy'] = {'type': 'Recreate'}
            for container in doc['spec']['template']['spec']['containers']:
                container['image'] = f"{os.environ['ACR_LOGIN_SERVER']}/{container['name']}:{os.environ['IMAGE_TAG']}"
                container['envFrom'] = [item for item in container.get('envFrom', []) if 'secretRef' not in item]
                container['startupProbe'] = {
                    'httpGet': {'path': '/healthz', 'port': 'http'},
                    'periodSeconds': 5, 'failureThreshold': 60,
                }
        documents.append(doc)

documents.append({
    'apiVersion': 'v1', 'kind': 'Service',
    'metadata': {'name': 'aurora-web', 'namespace': 'aurora'},
    'spec': {'type': 'LoadBalancer',
             'selector': {'app.kubernetes.io/part-of': 'aurora',
                          'app.kubernetes.io/component': 'assistant-api'},
             'ports': [{'name': 'http', 'port': 80, 'targetPort': 8100}]},
})
(OUT / 'deploy.yaml').write_text(yaml.safe_dump_all(documents, sort_keys=False))
print('Prepared demo Dockerfiles and work/azure/deploy.yaml')
