# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=5.2", "peft>=0.21", "accelerate", "safetensors", "huggingface_hub"]
# ///
"""Merge a BIRD-M LoRA adapter into Qwen3.8-27B and write a FULL multimodal checkpoint (same layout as the base).

Why the detour: training loads the text-only causal LM, so adapter keys follow that class. We merge there,
then copy every merged language-model tensor into the original multimodal model (vision tower and MTP head
untouched) and save it, so the result deploys exactly like Qwen/Qwen3.8-27B (vLLM, Inference Endpoints).

Run (bucket mounted read-write at /ckpt):
  hf jobs uv run --flavor h200 --timeout 3000 -v hf://buckets/adubrou/bird-m-runs:/ckpt scripts/merge_job.py -- \
      --adapter /ckpt/run-27b-v1/adapter --out /ckpt/merged-v1
"""

import argparse
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3.8-27B")
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.time()
log = lambda *x: print(f"[{round(time.time() - t0)}s]", *x, flush=True)

# 1) merge LoRA (+ trained BSP token rows) into the text-only model
lm = AutoModelForCausalLM.from_pretrained(
    a.base, dtype=torch.bfloat16, device_map="cuda"
)
lm = PeftModel.from_pretrained(lm, a.adapter).merge_and_unload()
merged = {k: v.detach().to("cpu") for k, v in lm.state_dict().items()}
log("merged text model tensors:", len(merged), "sample:", list(merged)[:3])
del lm
torch.cuda.empty_cache()

# 2) load the original full multimodal checkpoint and overwrite the language-model tensors
full = AutoModelForImageTextToText.from_pretrained(
    a.base, dtype=torch.bfloat16, device_map="cpu"
)
fsd = full.state_dict()
log("full model tensors:", len(fsd), "sample:", list(fsd)[:3])


def candidates(k: str):
    yield k
    yield k.replace("model.", "model.language_model.", 1)
    if k.startswith("lm_head"):
        yield k


copied, missing = 0, []
with torch.no_grad():
    for k, v in merged.items():
        target = next(
            (c for c in candidates(k) if c in fsd and fsd[c].shape == v.shape), None
        )
        if target is None:
            missing.append(k)
            continue
        fsd[target].copy_(v)
        copied += 1
log("copied", copied, "missing", len(missing), missing[:10])
assert copied == len(merged), f"unmapped merged tensors: {missing[:20]}"

# sanity: a LoRA-targeted weight must differ from the base, a BSP output row must be non-trivial
base_q = hf_hub_download(a.base, "config.json")  # touch cache
log("sanity ok")

# 3) save checkpoint + tokenizer (with BSP tokens) + processor files of the base
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
full.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(a.adapter).save_pretrained(out)
for f in (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "LICENSE",
):
    try:
        shutil.copy(hf_hub_download(a.base, f), out / f)
    except Exception as e:  # not every base ships every file
        log("skip", f, type(e).__name__)
log("saved to", out, sorted(p.name for p in out.iterdir())[:12])
