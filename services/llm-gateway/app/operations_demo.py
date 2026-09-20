"""Fault injection uses the production router with isolated synthetic providers.
No live provider, cache, breaker, or vehicle state is modified.
"""
import asyncio
import time
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Request
from vehicle_ai.schemas import GatewayRequest, Message
from .router import LlmRouter
from .providers.base import BaseProvider, ProviderError, ProviderResult

router = APIRouter()

class DemoProvider(BaseProvider):
    models = ['synthetic-health-probe']
    def __init__(self, name):
        self.name = name
        self.fail = False
    def is_configured(self):
        return True
    async def complete(self, req):
        if self.fail:
            try:
                await asyncio.wait_for(asyncio.sleep(1), timeout=.08)
            except asyncio.TimeoutError:
                raise ProviderError('Injected 80 ms timeout')
        return ProviderResult(content='Synthetic routing probe completed.', model=self.models[0], region='isolated-demo')

async def run_demo():
    primary = DemoProvider('demo-primary'); backup = DemoProvider('demo-backup')
    isolated = LlmRouter(providers={'demo-primary':primary,'demo-backup':backup},
                         priority=['demo-primary','demo-backup'], weights={},cache_ttl=0,cache_max=1,
                         breaker_threshold=1,breaker_cooldown=1)
    timeline = []
    async def probe(label):
        result = await isolated.route(GatewayRequest(messages=[Message(role='user',content='Hello')], cache_key=uuid4().hex))
        timeline.append({'step':label, 'provider':result.provider,'latency_ms':result.latency_ms,
                         'fallback_chain':result.fallback_chain, 'breaker':isolated.health()['demo-primary']['circuit'],
                         'refused':result.refused})
    await probe('1 · Healthy primary')
    primary.fail = True
    await probe('2 · Inject timeout → fallback')
    await probe('3 · Open circuit skips primary')
    primary.fail = False
    await asyncio.sleep(1.05)
    await probe('4 · Cooldown expires → primary recovers')
    checks = {'healthy_primary': timeline[0]['provider']=='demo-primary',
              'timeout_fallback': timeline[1]['provider']=='demo-backup' and bool(timeline[1]['fallback_chain']),
              'circuit_skipped': any('circuit-open' in x for x in timeline[2]['fallback_chain']),
              'recovered': timeline[3]['provider']=='demo-primary' and timeline[3]['breaker']=='closed'}
    return {'id':uuid4().hex, 'at':time.time(), 'scope':'Synthetic providers; production LlmRouter; isolated from driver traffic.',
            'timeline':timeline, 'checks':checks, 'passed':all(checks.values()), 'injected_timeout_ms':80,'cooldown_s':1}

@router.post('/v1/ops/failure-demo')
async def failure_demo(request: Request):
    if request.client is None or request.client.host not in ('127.0.0.1','::1','testclient') or request.headers.get('origin'):
        raise HTTPException(403, 'Local service calls only')
    return await run_demo()
