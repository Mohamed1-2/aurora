#!/usr/bin/env python3
"""Train adapter-v2.1 from base (mirrors colab/train_lora.ipynb cells 11-16).

- Uses per-row ``instruction`` exactly like the notebook's ``fmt`` (this is the
  training-serving alignement fix: all v2.1 rows, incl. policy rows, carry the
  serve-time PREAMBLE verbatim).
- LoRA r=8 / alpha=16 / dropout=0.1 on q_proj+v_proj (notebook cell 14).
- 3 epochs, lr 5e-5, batch 4, grad-accum 2, weight-decay 0.01 (notebook cell 16).
- MPS/cuda when available (notebook only used cuda/cpu; MPS is a local-only
  speedup and does not change the training recipe).

Usage:
    AURORA_V21_DIR=artifacts/dataset/v2.1 \
    AURORA_V21_OUT=artifacts/finetune/v2.1 \
    .venv/bin/python services/finetune/train_v21.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = os.environ.get("AURORA_FINETUNE_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
DATA_DIR = Path(os.environ.get("AURORA_V21_DIR", str(ROOT / "artifacts/dataset/v2.1")))
OUT_DIR = Path(os.environ.get("AURORA_V21_OUT", str(ROOT / "artifacts/finetune/v2.1")))
MAX_LEN = 512

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device:", device)


def _context_line(context: dict) -> str:
    if not context:
        return ""
    items = []
    for k, v in context.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            items.append(f"{k} {round(float(v), 1)}")
    return "Telemetry: " + "; ".join(items) if items else ""


def fmt(r: dict, pre: str) -> str:
    user = r["input"]
    ctx = _context_line(r.get("context", {}))
    if ctx:
        user = f"{ctx}\n\nQuestion: {r['input']}"
    return (f"<|im_start|>system\n{r.get('instruction', pre)}\n"
            f"<|im_start|>user\n{user}\n"
            f"<|im_start|>assistant\n{r['output']}<|im_end|>\n")


def load_rows(path: Path) -> list[dict]:
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


class TextDataset(Dataset):
    """Tokenizer-agnostic lazy dataset: tokenizes fmt() output on the fly."""

    def __init__(self, rows: list[dict], tokenizer, max_len: int = MAX_LEN):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        text = fmt(self.rows[i], "")
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_len, padding="max_length"
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(enc["input_ids"], dtype=torch.long),
        }


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_rows(DATA_DIR / "train.jsonl")
    val_rows = load_rows(DATA_DIR / "val.jsonl")
    print(f"train rows: {len(train_rows)} | val rows: {len(val_rows)}")

    train_ds = TextDataset(train_rows, tokenizer)
    val_ds = TextDataset(val_rows, tokenizer)
    print("train n:", len(train_ds))

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        evaluation_strategy="epoch",
        logging_dir=str(OUT_DIR / "logs"),
        save_total_limit=1,
        report_to=[],
        bf16=False,
        fp16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )
    trainer.train()

    adapter_dir = OUT_DIR / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved adapter to {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())