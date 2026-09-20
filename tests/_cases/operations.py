"""Isolated assistant operations assertions (no model or simulator calls)."""
import asyncio
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'services/assistant-api'),str(ROOT/'shared')]
from app.api import operations as ops
from app.core.observability import traced, span, traces
from fastapi import HTTPException
from starlette.requests import Request
from vehicle_ai.schemas import AssistantRequest, AssistantResponse

candidate={'provider':'local','model':'test'}
rows=[{'status':'passed','track':'deterministic'} for _ in range(18)]
rows += [{'status':'passed','track':'model','latency_ms':100,'response':{'provider':'local','model':'test'},'gateway_exchanges':[{'response':{'cache_hit':False}}]} for _ in range(3)]
assert all(ops.gate({'results':rows},candidate).values())
for field,value in [('provider','mock'),('model','other')]:
 bad=copy.deepcopy(rows);bad[-1]['response'][field]=value
 assert not all(ops.gate({'results':bad},candidate).values())
for change in ['missing','cache','latency','failed']:
 bad=copy.deepcopy(rows)
 if change=='missing':bad.pop()
 if change=='cache':bad[-1]['gateway_exchanges'][0]['response']['cache_hit']=True
 if change=='latency':bad[-1]['latency_ms']=36000
 if change=='failed':bad[0]['status']='failed'
 assert not all(ops.gate({'results':bad},candidate).values())
for host,origin,allowed in [('127.0.0.1',None,True),('127.0.0.1','http://localhost:8100',True),('127.0.0.1','https://evil.example',False),('10.0.0.2',None,False)]:
 request=Request({'type':'http','client':(host,123),'headers':[(b'origin',origin.encode())] if origin else []})
 try:ops.local_only(request);assert allowed
 except HTTPException:assert not allowed

class Pipe:
 async def _ask(self,req):
  with span('isolated',question=req.question):await asyncio.sleep(.001)
  return AssistantResponse(question=req.question,answer=req.question,provider='telemetry',model='facts')

async def main():
 traces.clear();pipe=Pipe()
 answers=await asyncio.gather(*(traced(pipe,AssistantRequest(question=str(i))) for i in range(10)))
 assert len({a.trace_id for a in answers})==10
 assert all(r['stages'][0]['question']==r['question'] for r in traces)
 with tempfile.TemporaryDirectory() as tmp:
  ops.STORE=Path(tmp);ops.STATE=ops.STORE/'releases.json'
  c={**candidate,'id':'test','status':'unevaluated'}
  ops.save({'active':None,'previous':None,'candidates':[c],'history':[]})
  request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(pipeline=pipe)))
  try:await ops.activate('test',request);assert False,'ungated activation allowed'
  except HTTPException as e:assert e.status_code==409
  report=ops.STORE/'report.json';report.write_text('{}')
  import hashlib,time
  c.update(status='passed',fingerprint=ops.fingerprint(),evidence=str(report),evidence_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),evaluated_at=time.time(),temperature=.1,max_tokens=96)
  ops.save({'active':None,'previous':None,'candidates':[c],'history':[]})
  async def providers():return {'local':{'configured':True,'models':['test']}}
  ops.providers=providers
  await ops.activate('test',request);assert pipe.release['id']=='test'
  restored=Pipe();ops.restore_release(restored);assert restored.release['id']=='test'
  await ops.rollback(request);assert pipe.release is None
  report.write_text('tampered')
  try:await ops.activate('test',request);assert False,'tampered evidence activated'
  except HTTPException as e:assert e.status_code==409
 print('Operations: gate failures, origin guard, concurrent tracing, activation, restore, rollback, tamper protection verified')
asyncio.run(main())
