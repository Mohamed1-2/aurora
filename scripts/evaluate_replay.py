"""Controlled-time replay of the production AssistantPipeline (no simulator writes)."""
from __future__ import annotations
import argparse,asyncio,hashlib,json,platform,re,statistics,sys,time,uuid
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
UNITS={'%':'percent','percent':'percent','kilometres':'km','kilometers':'km','kilometre':'km','kilometer':'km','km':'km','miles':'mile','mile':'mile','kpa':'kpa','km/h':'kph','kilometres per hour':'kph','kilometers per hour':'kph','mph':'mph'}
NUMBER=re.compile(r'(?<![\w.])(-?\d+(?:\.\d+)?)\s*(kilometres per hour|kilometers per hour|km/h|kilometres|kilometers|kilometre|kilometer|miles|mile|percent|mph|kpa|km|%)(?!\w)',re.I)
def extract(text):return [{'value':float(m[1]),'unit':UNITS[m[2].lower()]} for m in NUMBER.finditer(text)]
def score(response,expected):
 text=response['answer'].lower().replace("can’t", "cannot").replace("can't", "cannot");found=extract(text)
 matches=lambda n,v: n['unit']==v['unit'] and abs(n['value']-v['value'])<=v.get('tolerance',.01)
 required=expected.get('numbers',[])
 checks={'numeric_and_units':all(any(matches(n,v) for n in found) for v in required),
 'required_concepts':all(any(word in text for word in group) for group in expected.get('required_groups',[])),
 'forbidden_phrases_absent':not any(word in text for word in expected.get('forbidden',[])),
 'behaviour':('refused' not in expected or response.get('refused')==expected['refused']) and ('provider' not in expected or response.get('provider') in expected['provider'])}
 if expected.get('no_numbers'):checks['no_invented_numbers']=not re.search(r'\d',text)
 if required or 'allowed_numbers' in expected:
  allowed=required+expected.get('allowed_numbers',[])
  checks['no_unlisted_numeric_claims']=all(any(matches(n,v) for v in allowed) for n in found)
  covered=[m.span() for m in NUMBER.finditer(text)]
  checks['no_unrecognized_numeric_claims']=not any(not any(a<=m.start() and m.end()<=b for a,b in covered) for m in re.finditer(r'(?<![\w.])-?\d+(?:\.\d+)?',text))
 return {'checks':checks,'extracted_numbers':found,'passed':all(checks.values()),'human_semantics':'pending'}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
async def run(args):
 root=Path(args.project).resolve();fixture=Path(args.cases or root/'evaluation/replay-v1/cases.json').resolve();data=json.loads(fixture.read_text())
 out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
 sys.path[:0]=[str(root/'services/assistant-api'),str(root/'shared')]
 import httpx
 from app.core import service,conversation
 from app.telemetry.ingest import TelemetryIngest
 from vehicle_ai.schemas import AssistantRequest,RAGChunk
 clock=SimpleNamespace(time=lambda:now[0],perf_counter=time.perf_counter)
 now=[data['epoch']];service.time=clock;conversation.time=clock
 telemetry=TelemetryIngest(ws_url='ws://unused',replay_path='')
 pipe=service.AssistantPipeline(telemetry);await pipe._client.aclose()
 captures=[]
 class Client:
  def __init__(self):self.client=httpx.AsyncClient(timeout=45)
  async def post(self,url,**kwargs):
   if not args.gateway:raise RuntimeError('Unexpected model route in deterministic replay; no stub answer is substituted')
   if getattr(args, 'provider', None):
    kwargs['json']['provider']=args.provider
    kwargs['json']['cache_key']='evaluation-'+uuid.uuid4().hex
   if getattr(args, 'temperature', None) is not None:kwargs['json']['temperature']=args.temperature
   if getattr(args, 'max_tokens', None) is not None:kwargs['json']['max_tokens']=args.max_tokens
   capture={'request':kwargs['json']};captures.append(capture)
   result=await self.client.post(args.gateway.rstrip('/')+'/v1/complete',**kwargs)
   capture['status']=result.status_code
   try:capture['response']=result.json()
   except Exception:capture['response_text']=result.text
   return result
 client=Client();pipe._client=client
 async def retrieve(q,intents):return [RAGChunk(**d) for d in data['documents']]
 async def prediction():return None
 pipe._retrieve=retrieve;pipe._fetch_prediction_explanation=prediction
 rows=[]
 for session in data['sessions']:
  telemetry.frames.clear();history=[]
  for idx,turn in enumerate(session['turns']):
   ident=f"{session['id']}/{idx+1}"
   if session['track']=='model' and not args.gateway:
    rows.append({'id':ident,'track':'model','status':'skipped','reason':'--gateway not supplied'});continue
   now[0]=data['epoch']+turn['at_s'];frame={'vehicle_id':'REPLAY-EV','timestamp':now[0]-turn['age_s'],'seq':idx+1,'signals':turn['signals'],'active_dtcs':[],'warnings':turn.get('warnings',[])}
   telemetry.frames.append(frame);before=len(captures);start=time.perf_counter()
   try:
    response=await pipe.ask(AssistantRequest(question=turn['question'],response_style='concise',conversation=history[-6:],session_id=session['id']))
    result=response.model_dump(mode='json');scored=score(result,turn['expected'])
    if session['track']=='model' and result['provider'] in ('mock','policy','telemetry','guide','deterministic'):
     scored['checks']['actual_model_route']=False;scored['passed']=False
    history.extend([{'role':'user','content':turn['question']},{'role':'assistant','content':result['answer'][:2000]}])
    rows.append({'id':ident,'track':session['track'],'status':'passed' if scored['passed'] else 'failed','question':turn['question'],'reference_answer':turn['reference_answer'],'expected':turn['expected'],'clock':now[0],'frames':list(telemetry.frames),'response':result,'latency_ms':round((time.perf_counter()-start)*1000,3),'scoring':scored,'gateway_exchanges':captures[before:]})
   except Exception as e:rows.append({'id':ident,'track':session['track'],'status':'error','question':turn['question'],'error':repr(e),'gateway_exchanges':captures[before:]})
 await client.client.aclose()
 files=[fixture,Path(__file__).resolve()]
 for path in ['services/assistant-api/app','shared/vehicle_ai']:
  files.extend(sorted((root/path).rglob('*.py')))
 hashes={str(p.relative_to(root)) if p.is_relative_to(root) else str(p):digest(p) for p in files}
 tracks={}
 for track in ['deterministic','model']:
  rs=[r for r in rows if r['track']==track];lat=[r['latency_ms'] for r in rs if 'latency_ms' in r]
  tracks[track]={'passed':sum(r['status']=='passed' for r in rs),'failed':sum(r['status'] in ('failed','error') for r in rs),'skipped':sum(r['status']=='skipped' for r in rs),'median_ms':statistics.median(lat) if lat else None,'max_ms':max(lat) if lat else None}
 report={'suite':data['version'],'command':[sys.executable,*sys.argv],'cwd':str(Path.cwd()),'python':platform.python_version(),'dependencies':{p:version(p) for p in ['httpx','pydantic','fastapi']},'platform':platform.platform(),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gateway':args.gateway,'scope':'Production pipeline with frozen retrieval excerpts and controlled clock. Tests no live simulator or retrieval ranking. Model generation may vary; hashes and gateway exchanges capture context. Automated passes do not imply semantic or acoustic validation.','fixture_provenance':data['provenance'],'hashes':hashes,'tracks':tracks,'release_ready':False,'human_review':'pending','audio_validation':'separate; not established by this run','results':rows}
 (out/'results.json').write_text(json.dumps(report,indent=2))
 reviews=[{'id':r['id'],'reviewer':None,'ratings':{'intent_relevance':None,'evidence_support':None,'appropriate_action':None,'followup_completion':None,'clarity':None},'unsupported_claim_spans':[],'notes':''} for r in rows if 'response' in r]
 (out/'human-review.json').write_text(json.dumps({'run_sha256':digest(out/'results.json'),'status':'pending','reviews':reviews},indent=2))
 (out/'summary.md').write_text('# Replay evaluation\n\n'+json.dumps(tracks,indent=2)+'\n\nHuman review pending. Audio validation not included. See results.json for command, hashes, raw replies and gateway exchanges.\n')
 print(json.dumps(tracks,indent=2));print(out)
 return int(any(r['status'] in ('failed','error') for r in rows))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--cases');p.add_argument('--provider');p.add_argument('--temperature',type=float);p.add_argument('--max-tokens',type=int);p.add_argument('--gateway',help='Optional real local gateway; never replaced with mock success');p.add_argument('--out',required=True,help='New directory; existing runs are never overwritten');args=p.parse_args();sys.exit(asyncio.run(run(args)))
