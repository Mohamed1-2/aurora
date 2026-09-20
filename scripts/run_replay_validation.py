"""One command, saved subprocess commands/output, no live simulator mutation."""
import argparse,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gateway');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
commands=[[sys.executable,'scripts/evaluate_replay.py','--out',str(out/'replay')]+(['--gateway',a.gateway] if a.gateway else []),['node','tests/test_voice_contract.cjs',str(out/'voice-contract.json')],['node','tests/test_driving_ui.cjs'],[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_replay_scoring.py','tests/test_range_consistency.py','tests/test_conversation_relevance.py','tests/test_policy.py','tests/test_mock_provider.py','tests/test_proactive_engine.py','tests/test_hud.py','tests/test_control_relay.py','tests/test_gateway_safety_gate.py','tests/test_assistant_prediction_fetch.py','tests/test_driving_time.py','tests/test_telemetry_utils.py','tests/test_simulator_trip.py']]
results=[];env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
for i,command in enumerate(commands):
 start=time.perf_counter()
 try:
  proc=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=240)
  result={'command':command,'cwd':str(root),'returncode':proc.returncode,'elapsed_s':time.perf_counter()-start,'stdout':proc.stdout,'stderr':proc.stderr}
 except (subprocess.TimeoutExpired,OSError) as e:result={'command':command,'returncode':-1,'error':str(e)}
 (out/f'command-{i+1}.json').write_text(json.dumps(result,indent=2));results.append(result)
 print('PASS' if result['returncode']==0 else 'FAIL', ' '.join(command),flush=True)
 hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['scripts','tests','services/assistant-api/app/static','evaluation/replay-v1'] for f in (root/folder).rglob('*') if f.is_file() and f.suffix in ('.js','.cjs','.py','.json','.html','.md')}
 manifest={'command':[sys.executable,*sys.argv],'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'platform':platform.platform(),'hashes':hashes,'all_commands_passed':all(r['returncode']==0 for r in results),'not_established':['human semantic quality','real audio intelligibility','cabin-noise recognition','production automotive safety'],'commands':[{'command':r['command'],'returncode':r['returncode']} for r in results]}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Evidence:',out)
sys.exit(0 if manifest['all_commands_passed'] else 1)
