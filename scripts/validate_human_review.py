"""Validate review completeness against a saved run; never certify vehicle safety."""
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('review');a=p.parse_args()
raw=Path(a.results).read_bytes();run=json.loads(raw);review=json.loads(Path(a.review).read_text());errors=[]
if review.get('run_sha256')!=hashlib.sha256(raw).hexdigest():errors.append('Review is not bound to this exact results file')
expected={r['id'] for r in run['results'] if 'response' in r};entries=review.get('reviews',[])
if len({r['id'] for r in entries})!=len(entries):errors.append('Duplicate reviews')
if {r['id'] for r in entries}!=expected:errors.append('Missing or unexpected case reviews')
for r in entries:
 if not str(r.get('reviewer') or '').strip():errors.append(r['id']+': reviewer missing')
 for k in ['intent_relevance','evidence_support','appropriate_action','followup_completion','clarity']:
  value=r.get('ratings',{}).get(k)
  if type(value) is not int or value not in (0,1,2):errors.append(r['id']+': incomplete rating '+k)
completed=not errors
accepted=completed and all(all(v==2 for v in r['ratings'].values()) and not r.get('unsupported_claim_spans') for r in entries) and all(r['status']=='passed' for r in run['results'])
print(json.dumps({'complete':completed,'all_episodes_accepted':accepted,'errors':errors,'scope':'Review completeness and text episode acceptance only; no acoustic or automotive readiness claim'},indent=2))
sys.exit(0 if accepted else 1)
