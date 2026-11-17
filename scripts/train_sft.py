"""BIRD-M SFT (LoRA + full training of the BSP token rows). Runs on CPU (smoke/pilot) or GPU.

Usage:
  python scripts/train_sft.py --model Qwen/Qwen3.5-0.8B --data data/out/skills_pilot.jsonl \
      --out runs/pilot --epochs 1 --max-steps 0 --lr 2e-4 --batch-tokens 4096

Loss only on Segment.train tokens (routing decisions, answers). Logs JSON lines to <out>/log.jsonl.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bird_m.skills import spec  # noqa: E402
from bird_m.train.data import (
    IGNORE,
    encode,
    is_trainable,
    record_to_segments,
    setup_tokenizer,
)  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--max-steps", type=int, default=0, help="0 = no cap")
ap.add_argument("--max-examples", type=int, default=0)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument(
    "--emb-lr", type=float, default=1e-3, help="lr for the new BSP token rows"
)
ap.add_argument("--lora-r", type=int, default=32)
ap.add_argument(
    "--batch-tokens",
    type=int,
    default=4096,
    help="tokens per optimizer step (packed by length)",
)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

random.seed(a.seed)
torch.manual_seed(a.seed)
if a.threads:
    torch.set_num_threads(a.threads)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
log = open(out / "log.jsonl", "a", encoding="utf-8")


def emit(**kw):
    kw["t"] = round(time.time(), 1)
    print(json.dumps(kw), flush=True)
    log.write(json.dumps(kw) + "\n")
    log.flush()


# ---------------------------------------------------------------- data
tok = AutoTokenizer.from_pretrained(a.model)
setup_tokenizer(tok)
rng = random.Random(a.seed)
examples, skipped = [], 0
for path in a.data:
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        if not is_trainable(rec):
            skipped += 1
            continue
        e = encode(tok, record_to_segments(rec, rng), a.max_len)
        if e is None:
            skipped += 1
            continue
        examples.append(e)
rng.shuffle(examples)
if a.max_examples:
    examples = examples[: a.max_examples]
emit(
    event="data",
    examples=len(examples),
    skipped=skipped,
    tokens=sum(len(e["input_ids"]) for e in examples),
)


def batches(exs):
    """Group by similar length, up to batch_tokens padded tokens per batch."""
    exs = sorted(exs, key=lambda e: len(e["input_ids"]))
    cur = []
    for e in exs:
        if cur and (len(cur) + 1) * len(e["input_ids"]) > a.batch_tokens:
            yield cur
            cur = []
        cur.append(e)
    if cur:
        yield cur


def collate(group):
    L = max(len(e["input_ids"]) for e in group)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(group), L), pad, dtype=torch.long)
    lab = torch.full((len(group), L), IGNORE, dtype=torch.long)
    att = torch.zeros((len(group), L), dtype=torch.long)
    for i, e in enumerate(group):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        lab[i, :n] = torch.tensor(e["labels"])
        att[i, :n] = 1
    return ids.to(dev), lab.to(dev), att.to(dev)


# ---------------------------------------------------------------- model
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype)
emb = model.get_input_embeddings()
if emb.weight.shape[0] < spec.FIRST_FREE_ID + len(spec.ALL_TOKENS):
    model.resize_token_embeddings(len(tok))
with (
    torch.no_grad()
):  # init BSP rows: mean embedding + small noise (rows were never trained)
    w = model.get_input_embeddings().weight
    mean = w[: spec.FIRST_FREE_ID].mean(0)
    for i in spec.TOKEN_IDS.values():
        w[i] = mean + 0.01 * torch.randn_like(mean) * w[: spec.FIRST_FREE_ID].std()
    if not model.config.get_text_config().tie_word_embeddings:
        o = model.get_output_embeddings().weight
        omean = o[: spec.FIRST_FREE_ID].mean(0)
        for i in spec.TOKEN_IDS.values():
            o[i] = omean
model.config.use_cache = False
if dev == "cuda":
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

# new token rows are trained in full; with untied embeddings (Qwen3.8-27B) the OUTPUT rows must be
# trained too, otherwise the model could never emit a BSP token
bsp_ids = list(spec.TOKEN_IDS.values())
tied = model.config.get_text_config().tie_word_embeddings
token_rows = {"embed_tokens": bsp_ids} if tied else {"embed_tokens": bsp_ids, "lm_head": bsp_ids}
emit(event="token_rows", tied=tied, modules=list(token_rows))
cfg = LoraConfig(
    r=a.lora_r,
    lora_alpha=2 * a.lora_r,
    lora_dropout=0.0,
    target_modules="all-linear",
    trainable_token_indices=token_rows,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, cfg)
model.to(dev)
trainable = [p for p in model.parameters() if p.requires_grad]
emit(event="model", device=dev, trainable_params=sum(p.numel() for p in trainable))

emb_params = [
    p for n, p in model.named_parameters() if p.requires_grad and "token_adapter" in n
]
other = [
    p
    for n, p in model.named_parameters()
    if p.requires_grad and "token_adapter" not in n
]
opt = torch.optim.AdamW(
    [{"params": other, "lr": a.lr}, {"params": emb_params, "lr": a.emb_lr}],
    weight_decay=0.0,
)

groups = list(batches(examples))
total = math.ceil(len(groups) * a.epochs)
if a.max_steps:
    total = min(total, a.max_steps)
warm = max(1, total // 20)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda s: (
        min(1.0, (s + 1) / warm)
        * 0.5
        * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))
    ),
)
emit(event="plan", steps=total, batches_per_epoch=len(groups))

# ---------------------------------------------------------------- train
model.train()
step, t0, seen = 0, time.time(), 0
while step < total:
    rng.shuffle(groups)
    for g in groups:
        if step >= total:
            break
        ids, lab, att = collate(g)
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        seen += int(att.sum())
        if step == 1 or step % 10 == 0 or step == total:
            emit(
                event="step",
                step=step,
                of=total,
                loss=round(loss.item(), 4),
                tok_per_s=round(seen / (time.time() - t0), 1),
            )

model.save_pretrained(out / "adapter")
tok.save_pretrained(out / "adapter")
emit(event="done", steps=step, seconds=round(time.time() - t0))
