"""Local portfolio control plane: bounded traces and evidence-gated configuration releases.
Single assistant process only; not a fleet deployment or a production quality certification.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from app.config import settings
from app.core.observability import traces

ROOT = Path(__file__).resolve().parents[4]
STORE = ROOT / 'artifacts/operations'
STATE = STORE / 'releases.json'
lock = asyncio.Lock()
tasks = set()


def local_only(request: Request):
    origin = request.headers.get('origin')
    host = request.headers.get('host')
    if host and urlparse('//'+host).hostname not in ('localhost', '127.0.0.1', '::1', 'testserver'):
        raise HTTPException(403, 'Local hostname required')
    if request.client is None or request.client.host not in ('127.0.0.1', '::1', 'testclient'):
        raise HTTPException(403, 'Operations is available on this computer only')
    if origin and (urlparse(origin).hostname not in ('localhost', '127.0.0.1', '::1') or urlparse(origin).port != settings.port):
        raise HTTPException(403, 'Cross-origin operations are disabled')

router = APIRouter(dependencies=[Depends(local_only)])


def load():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {'active': None, 'previous': None, 'candidates': [], 'history': []}


def save(state):
    STORE.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    os.replace(temp, STATE)


def restore_release(pipeline):
    try:
        pipeline.release = load()['active']
    except (ValueError, OSError, KeyError):
        pipeline.release = None


def fingerprint():
    """Bind evidence to code + frozen cases, not merely a label or timestamp."""
    files = [ROOT/'scripts/evaluate_replay.py', ROOT/'evaluation/replay-v1/cases.json']
    for folder in ('services/assistant-api/app', 'services/llm-gateway/app', 'shared/vehicle_ai'):
        files.extend(sorted((ROOT/folder).rglob('*.py')))
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def gate(report, candidate):
    rows = report.get('results', [])
    model = [r for r in rows if r.get('track') == 'model']
    return {
        'all_21_replay_turns_pass': len(rows) == 21 and all(r.get('status') == 'passed' for r in rows),
        'three_actual_candidate_model_answers': len(model) == 3 and all(
            r.get('response', {}).get('provider') == candidate['provider'] and
            r.get('response', {}).get('model') == candidate['model'] for r in model),
        'model_requests_uncached': len(model) == 3 and all(r.get('gateway_exchanges') and all(
            x.get('response', {}).get('cache_hit') is False for x in r['gateway_exchanges']) for r in model),
        'model_turns_under_35_seconds': len(model) == 3 and all(r.get('latency_ms', float('inf')) <= 35000 for r in model),
    }


async def providers():
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(settings.gateway_url+'/providers')
        response.raise_for_status()
        return response.json()


@router.get('/status')
async def status():
    state = load()
    try:
        configured = await providers()
        gateway = 'connected'
    except httpx.HTTPError:
        configured = {}; gateway = 'unreachable'
    code = fingerprint()
    for c in state['candidates']:
        c['evidence_current'] = c.get('fingerprint') == code
        if c.get('status') == 'running' and not any(t.get_name() == c['id'] for t in tasks):
            c['status'] = 'interrupted'
    return {**state, 'providers': configured, 'gateway': gateway, 'traces': list(traces),
            'scope': 'Local configuration releases; same model weights. Human semantics and acoustic validation remain separate.',
            'trace_retention': 'Last 100 completed requests in memory. Restart clears questions and answers.'}


class CandidateRequest(BaseModel):
    provider: str = 'local'
    profile: Literal['concise', 'baseline'] = 'concise'


@router.post('/candidates')
async def create(req: CandidateRequest):
    try:
        configured = await providers()
    except httpx.HTTPError:
        raise HTTPException(503, 'Gateway unavailable')
    p = configured.get(req.provider, {})
    if req.provider in ('mock', 'none') or not p.get('configured') or not p.get('models'):
        raise HTTPException(400, 'Choose a configured language model, not the deterministic mock')
    async with lock:
        state = load()
        if len(state['candidates']) >= 30:
            raise HTTPException(409, 'Local candidate limit reached (30); archive evidence before creating more')
        baseline = req.profile == 'baseline'
        candidate = {'id': uuid.uuid4().hex[:12], 'name': 'Baseline configuration' if baseline else 'Concise response configuration',
                     'provider': req.provider, 'model': p['models'][0], 'temperature': 0.2 if baseline else 0.1, 'max_tokens': 110 if baseline else 96,
                     'status': 'unevaluated', 'created_at': time.time(), 'kind': 'same-weights configuration'}
        state['candidates'].append(candidate); save(state)
        return candidate


async def evaluate(ident):
    proc = None
    try:
        async with lock:
            state = load(); candidate = next(c for c in state['candidates'] if c['id'] == ident)
            candidate['status'] = 'running'; candidate['fingerprint'] = fingerprint(); save(state)
        run = STORE/'runs'/f'{ident}-{uuid.uuid4().hex[:8]}'
        run.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT/'scripts/evaluate_replay.py'), '--project', str(ROOT),
                   '--gateway', settings.gateway_url, '--provider', candidate['provider'],
                   '--temperature', str(candidate['temperature']), '--max-tokens', str(candidate['max_tokens']), '--out', str(run)]
        proc = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output, _ = await asyncio.wait_for(proc.communicate(), timeout=180)
        report = json.loads((run/'results.json').read_text())
        checks = gate(report, candidate)
        checks['source_unchanged_during_run'] = candidate['fingerprint'] == fingerprint()
        checks['process_success'] = proc.returncode == 0
        async with lock:
            state = load(); c = next(c for c in state['candidates'] if c['id'] == ident)
            c.update(status='passed' if all(checks.values()) else 'blocked', checks=checks,
                     evaluated_at=time.time(), tracks=report['tracks'], command=command,
                     evidence=str((run/'results.json').relative_to(ROOT)),
                     evidence_sha256=hashlib.sha256((run/'results.json').read_bytes()).hexdigest(),
                     human_review='pending', production_ready=False)
            save(state)
    except BaseException as exc:
        if proc and proc.returncode is None:
            proc.kill(); await proc.wait()
        async with lock:
            state = load(); c = next(c for c in state['candidates'] if c['id'] == ident)
            c.update(status='blocked', error=type(exc).__name__); save(state)
        if isinstance(exc, asyncio.CancelledError):
            raise


@router.post('/candidates/{ident}/evaluate')
async def run_evaluation(ident: str):
    async with lock:
        state = load()
        if not any(c['id'] == ident for c in state['candidates']):
            raise HTTPException(404, 'Candidate not found')
        if tasks:
            raise HTTPException(409, 'An evaluation is already running; wait for its result')
        task = asyncio.create_task(evaluate(ident), name=ident)
        tasks.add(task); task.add_done_callback(tasks.discard)
    return {'status': 'running', 'id': ident}


@router.get('/candidates/{ident}/evidence')
def evidence(ident: str):
    c = next((c for c in load()['candidates'] if c['id'] == ident), None)
    if not c or not c.get('evidence'):
        raise HTTPException(404, 'No evaluation evidence yet')
    return json.loads((ROOT/c['evidence']).read_text())


@router.post('/candidates/{ident}/activate')
async def activate(ident: str, request: Request):
    async with lock:
        state = load(); c = next((c for c in state['candidates'] if c['id'] == ident), None)
        if not c or c.get('status') != 'passed' or c.get('fingerprint') != fingerprint():
            raise HTTPException(409, 'Activation blocked: run a passing evaluation against current source')
        report_path = ROOT/c['evidence']
        if not report_path.exists() or hashlib.sha256(report_path.read_bytes()).hexdigest() != c['evidence_sha256']:
            raise HTTPException(409, 'Evaluation evidence is missing or changed')
        if time.time()-c['evaluated_at'] > 3600:
            raise HTTPException(409, 'Evidence is over one hour old. Evaluate again before activation.')
        try:
            configured = await providers()
        except httpx.HTTPError:
            raise HTTPException(503, 'Gateway unavailable')
        if c['model'] not in configured.get(c['provider'], {}).get('models', []):
            raise HTTPException(409, 'Provider configuration changed; evaluate again')
        if state['active'] and state['active']['id'] == ident:
            return {'active': state['active']}
        state['previous'] = state['active']
        state['active'] = {k:c[k] for k in ('id', 'provider', 'model', 'temperature', 'max_tokens')}
        state['history'] = (state['history']+[{'action':'activate', 'id':ident, 'at':time.time()}])[-50:]
        save(state); request.app.state.pipeline.release = state['active']
        return {'active':state['active'], 'scope':'Local demo only; human review pending'}


@router.post('/rollback')
async def rollback(request: Request):
    async with lock:
        state = load()
        if not state['active']:
            raise HTTPException(409, 'Already using the default configuration')
        state['active'], state['previous'] = state['previous'], state['active']
        state['history'] = (state['history']+[{'action':'rollback', 'id':state['active']['id'] if state['active'] else 'default', 'at':time.time()}])[-50:]
        save(state); request.app.state.pipeline.release = state['active']
        return {'active':state['active']}


@router.post('/failure-demo')
async def failure_demo():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(settings.gateway_url+'/v1/ops/failure-demo')
            response.raise_for_status(); return response.json()
    except httpx.HTTPError:
        raise HTTPException(503, 'Gateway unavailable; no simulated success returned')
