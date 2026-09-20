"""Live concise-assistant evaluation. Run with services up; output JSON evidence.
Uses response intent/provider as routing evidence, plus required-reference and
relevance checks. These rule-based checks do not establish semantic safety.
"""
import json,time,urllib.request,argparse,statistics
p=argparse.ArgumentParser();p.add_argument('--url',default='http://localhost:8100');p.add_argument('--out',default='/tmp/aurora-experience-eval.json');a=p.parse_args()
cases=[('Battery level?','battery'),('How much charge is remaining?','battery'),('What is my state of charge?','battery'),('How far can I drive?','range'),('how many kilometer I can drive','range'),('How much farther can we go?','range'),('What is my current range?','range'),('How fast are we going?','speed'),('What is my speed?','speed'),('Which tyre is lowest?','tyre'),('Can I reach my destination?','destination'),('Why?','clarification')]
expected={'battery':('battery_soc_pct',('charge','battery')),'range':('distance_to_empty_km',('range',)),'speed':('vehicle_speed_kph',('speed',)),'tyre':('tire_pressure_fl_kpa',('kpa',)),'destination':('destination_remaining_km',('destination','remaining','trip')),'clarification':(None,('which',))}
rows=[]
def ask(q,history=[]):
 t=time.perf_counter();req=urllib.request.Request(a.url+'/v1/ask',data=json.dumps({'question':q,'response_style':'concise','conversation':history}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=45) as r:result=json.load(r)
 return result,round((time.perf_counter()-t)*1000,2)
for q,topic in cases:
 try:
  d,ms=ask(q);key,words=expected[topic];signals=d.get('grounding',{}).get('telemetry_signals',[])
  relevant=any(w in d['answer'].lower() for w in words)
  grounded=key is None or key in signals or 'Set a destination' in d['answer']
  rows.append({'question':q,'topic':topic,'answer':d['answer'],'provider':d.get('provider'),'latency_ms':ms,'relevant':relevant,'reference_present':grounded,'passed':relevant and grounded})
 except Exception as e:rows.append({'question':q,'passed':False,'error':str(e)})
q='Can I reach my destination?';d,_=ask(q)
h=[{'role':'user','content':q},{'role':'assistant','content':d['answer']}]
for follow in ['Why?','What should I do?']:
 result,ms=ask(follow,h);text=result['answer'].lower();ok=any(w in text for w in ['trip','range','destination','charging'])
 rows.append({'question':follow,'context':q,'answer':result['answer'],'latency_ms':ms,'passed':ok,'relevant':ok})
lat=[r['latency_ms'] for r in rows if 'latency_ms' in r]
report={'created_at':time.time(),'scope':'live API; rule-based relevance and signal references, not human semantic scoring','cases':rows,'passed':sum(r['passed'] for r in rows),'total':len(rows),'latency_median_ms':statistics.median(lat),'latency_max_ms':max(lat)}
open(a.out,'w').write(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
