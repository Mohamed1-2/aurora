import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
mod_src = (ROOT / "services/evaluation/model_eval.py").read_text()
mod_json = json.dumps(mod_src)

cells = []

def md(text):
    lines = text.strip("\n").split("\n")
    cells.append({
        "cell_type": "markdown", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [l + "\n" if i < len(lines) - 1 else l for i, l in enumerate(lines)],
    })

def code(text):
    lines = text.strip("\n").split("\n")
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [l + "\n" if i < len(lines) - 1 else l for i, l in enumerate(lines)],
    })

# ---- 0 ----------
md("""# AURORA · Step 1 — Model-level evaluation (Base vs Base + LoRA)

This notebook answers the single most important question first:

> **Did fine-tuning improve the MODEL?**

It loads the **base model** and the **base model + your trained LoRA adapter**
side by side, runs the **held-out v2 test set** (`artifacts/dataset/v2/test.jsonl`,
which the model never trained on), and scores each model's raw responses using
`services/evaluation/model_eval.py` (imported below) — **no FastAPI, no gateway,
no port 8105, no deployment**.

| | |
|---|---|
| Input | `test.jsonl` (v2 held-out) + the trained `adapter/` from the Step-1 Colab run |
| Scores | rule-based (grounding, signal accuracy, hallucination, text F1) + optional LLM judge |
| Output | a Base-vs-LoRA table and a single honest verdict: `YES` / `NO` |

This is deliberately an **ablation step**: we change only the model weights (LoRA),
nothing else — no RAG, no telemetry plumbing, no server.""")

# ---- 1 ----------
md("""## 0 · Setup & point at the assets

Set **Runtime ▸ Change runtime type ▸ T4 GPU**, then **Runtime ▸ Run all**.

You need two things accessible to Colab:
1. the **v2 held-out test set** (`test.jsonl`) — upload to `MyDrive/aurora/v2/`
   (or anywhere `TEST` points), and
2. the **trained adapter** (`adapter/` folder) — upload to `MyDrive/aurora/v2/adapter/`.

Both default to that Drive layout so cell-1 works with zero edits.""")

# ---- 2 ----------
code("""from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')

BASE_MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct'   # same base used for training
V2_DIR = Path('/content/drive/MyDrive/aurora/v2')

TEST   = V2_DIR / 'test.jsonl'     # held-out set (never trained on)
ADAPTER_DIR = V2_DIR / 'adapter'   # trained LoRA from the Step-1 training run

print('test.jsonl exists  :', TEST.exists())
print('adapter dir exists :', (ADAPTER_DIR / 'adapter_config.json').exists())
""")

# ---- 3: embed model_eval ----
import base64 as _b64mod
_b64 = _b64mod.b64encode(mod_src.encode()).decode()
code(f"""# Vendor the repo's scorer so this notebook is self-contained and always uses
# the exact shipped logic (no path dependency). Embedded as base64 to avoid any
# quoting / newline issues with the module's docstrings.
import base64 as _b64
_MODULE = _b64.b64decode({_b64!r}).decode()
with open('model_eval.py', 'w') as f:
    f.write(_MODULE)
import model_eval
print('model_eval embedded at import time, source bytes =', len(_MODULE))
""")

# ---- 4 ----------
md("""## 1 · Load the base model and the base + LoRA

We load the same base model twice in one inference engine — once plain, once with
your adapter merged in via `peft`. Identical tokenizer, identical generation
settings, so the *only* difference between the two columns is the LoRA weights.""")

# ---- 5 ----------
code("""# Install (quietly — only torch/transformers/peft needed for inference)
!pip -q install transformers peft accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, '| GPU:', torch.cuda.get_device_name(0) if device == 'cuda' else 'n/a')

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- base (no adapter) ----
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()

# ---- base + LoRA ----
base_for_lora = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
peft_model = PeftModel.from_pretrained(base_for_lora, str(ADAPTER_DIR)).to(device).eval()
print('LoRA loaded from', ADAPTER_DIR)
""")

# ---- 6 ----------
md("""## 2 · Load the held-out test set

`test.jsonl` is the **v2 held-out split** — 1,117 rows the model never saw at
train time (TUM driving + EPFL charging, all seven intents). Every row carries the
`context` telemetry and the reference `output` that `model_eval` scores against.""")

# ---- 7 ----------
code("""import json

with open(TEST) as f:
    test_rows = [json.loads(l) for l in f if l.strip()]

print('test rows:', len(test_rows))
from collections import Counter
print('source_dataset:', dict(Counter(r['provenance']['source_dataset'] for r in test_rows)))
print('intent        :', dict(Counter(r.get('intent') for r in test_rows)))
""")

# ---- 8 ----------
md("""## 3 · Generate responses for both models

Same prompt format used at training time (`<|im_start|>system|user|assistant`).
Each model produces a response per test-row question; `model_eval.score_row` then
scores that response against the row's `context` + `derived` + reference `output`.""")

# ---- 9 ----------
code("""def make_generate(model):
    def gen(question):
        prompt = (
            '<|im_start|>system\\nYou are AURORA, an in-vehicle assistant grounded '
            'in the ORION-X. Answer only from the provided telemetry. If data is '
            'missing, refuse honestly.<|im_end|>\\n'
            f'<|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n'
        )
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return gen

gen_base = make_generate(base_model)
gen_peft = make_generate(peft_model)

# smoke test one row before the full run
q = test_rows[0]['input']
print('Q:', q)
print('base:', gen_base(q)[:100])
print('peft:', gen_peft(q)[:100])
""")

# ---- 10 ----------
code("""# Run the full held-out set on both models (rule-based scoring; pure python, fast)
import time

# Optional LLM-as-judge: set a callable judge(response, reference, context, kind)
# -> {'answer_quality': float, 'judge_grounding': float}. Left None here so the
# deterministic metrics carry the result; enable later for semantic answer quality.
JUDGE = None

base_scores, peft_scores = [], []
t0 = time.time()
for i, r in enumerate(test_rows):
    ctx, der, ref, q = r.get('context', {}), r.get('derived', {}), r['output'], r['input']
    try:
        rb = gen_base(q)
        rp = gen_peft(q)
        base_scores.append(model_eval.score_row(rb, ctx, der, ref, judge=JUDGE))
        peft_scores.append(model_eval.score_row(rp, ctx, der, ref, judge=JUDGE))
    except Exception as e:  # keep a partial run honest
        print(f'row {i} failed: {e}')
    if (i + 1) % 200 == 0:
        print(f'  {i+1}/{len(test_rows)} rows scored')

print(f'scored {len(base_scores)} rows in {time.time()-t0:.1f}s')
""")

# ---- 11 ----------
code("""# Safety probes: benign test set has no safety rows, so measure refusal separately.
probes = model_eval.example_safety_probes()
safe_base = sum(gen_base(p['input']) if p['prohibited'] else '' for p in probes)
print('--- safety probes (base) ---')
for p in probes:
    resp = gen_base(p['input'])
    score = model_eval.safety_compliance(resp, prohibited=p['prohibited'])
    print(f"[{score}] Q: {p['input'][:50]} -> {resp[:60]}")
""")

# ---- 12 ----------
code("""base_agg = model_eval.aggregate(base_scores)
peft_agg = model_eval.aggregate(peft_scores)
cmp = model_eval.compare(base_agg, peft_agg)
verdict = model_eval.verdict(cmp)

print('=' * 72)
print('BASE vs LoRA — v2 held-out model evaluation')
print('=' * 72)
print(f"{'metric':<22}{'base':>9}{'LoRA':>9}{'delta':>9}  improved")
print('-' * 72)
for m, d in cmp.items():
    imp = 'YES' if d['improved'] else 'no'
    print(f"{m:<22}{d['base']:>9.3f}{d['lora']:>9.3f}{d['delta']:>+9.3f}  {imp}")
print('-' * 72)
print('VERDICT: LoRA improved the model ->', verdict)
""")

# ---- 13 ----------
md("""## 4 · Next steps — only if the verdict is `YES`

1. **Save this result** — the printed table (and any judge scores) are your first
   *genuine measured* model-level result. Keep the numbers, don't invent them.
2. **Then** implement the serving/pipeline path (Step 2 in the plan): add a PEFT
   provider so the gateway can serve `base + LoRA`, wire `8100` (baseline) and
   `8105` (LoRA), and run `make compare` for the **system-level** ablation
   (+RAG, +telemetry) on top of the model win.

Do *not* mix RAG / telemetry / gateway changes into this notebook — one change at
a time, so the measured improvement is attributable to the LoRA alone.""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "cells": cells,
}

out = ROOT / "colab" / "evaluate_model.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print(f"wrote {out} with {len(cells)} cells")
