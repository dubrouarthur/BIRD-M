# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=5.2", "peft>=0.17", "accelerate", "safetensors", "huggingface_hub", "flash-linear-attention"]
# ///
"""Fusionne l'adaptateur LoRA Bird-Medium dans Qwen3.8-27B et écrit un checkpoint COMPLET (même format que la base).

L'entraînement charge le modèle texte seul (AutoModelForCausalLM) : on fusionne là, puis on recopie les
tenseurs du modèle de langage dans le modèle multimodal d'origine (tour de vision intacte). Le résultat se
déploie exactement comme Qwen/Qwen3.8-27B (vLLM, Inference Endpoints), raisonnement compris.
Lancement : python scripts/launch.py merge -- --adapter /out/bird-medium-v1/adapter --out /out/bird-medium-v1/merged
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3.8-27B")
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--flush-wait", type=int, default=300)
a = ap.parse_args()
t0 = time.time()
log = lambda *x: print(f"[{round(time.time() - t0)}s]", *x, flush=True)  # noqa: E731

lm = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
lm = PeftModel.from_pretrained(lm, a.adapter).merge_and_unload()
merged = {k: v.detach().to("cpu") for k, v in lm.state_dict().items()}
del lm
torch.cuda.empty_cache()

full = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cpu")
fsd = full.state_dict()


def candidates(k):
    yield k
    yield k.replace("model.", "model.language_model.", 1)


copied, missing = 0, []
with torch.no_grad():
    for k, v in merged.items():
        t = next((c for c in candidates(k) if c in fsd and fsd[c].shape == v.shape), None)
        if t is None:
            missing.append(k)
            continue
        fsd[t].copy_(v)
        copied += 1
log("tenseurs copiés", copied, "manquants", missing[:10])
assert copied == len(merged), f"tenseurs non mappés : {missing[:20]}"

local = Path("/tmp/merged")
full.save_pretrained(local, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(a.adapter).save_pretrained(local)
for f in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja", "generation_config.json", "LICENSE"):
    try:
        shutil.copy(hf_hub_download(a.base, f), local / f)
    except Exception as e:  # noqa: BLE001 — toutes les bases n'ont pas tous les fichiers
        log("ignoré", f, type(e).__name__)

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for f in sorted(local.iterdir()):
    shutil.copyfile(f, out / f.name)
    with open(out / f.name, "rb+") as fh:
        os.fsync(fh.fileno())
    log("copié", f.name, f.stat().st_size)
os.sync()
time.sleep(a.flush_wait)  # les envois en attente sont perdus si le conteneur s'arrête
log("terminé", sorted(p.name for p in out.iterdir()))
