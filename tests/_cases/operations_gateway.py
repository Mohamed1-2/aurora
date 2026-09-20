import asyncio
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'services/llm-gateway'),str(ROOT/'shared')]
from app.operations_demo import run_demo, DemoProvider
from app.router import LlmRouter
from vehicle_ai.schemas import GatewayRequest, Message
async def main():
 result=await run_demo();assert result['passed'],result
 p=DemoProvider('demo');router=LlmRouter({'demo':p},['demo'],{},60,20,2,1)
 req=GatewayRequest(messages=[Message(role='user',content='Hello')])
 first=await router.route(req);second=await router.route(req.model_copy(update={'trace_id':'second'}))
 assert second.cache_hit and second.trace_id=='second'
 assert not first.cache_hit and first.trace_id==req.trace_id
 assert router._cache_key(req)!=router._cache_key(req.model_copy(update={'temperature':.1}))
 assert router._cache_key(req)!=router._cache_key(req.model_copy(update={'max_tokens':96}))
 print('Gateway: timeout, fallback, circuit skip, recovery, cache trace identity and generation isolation verified')
asyncio.run(main())
