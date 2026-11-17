"""Construit le mélange d'entraînement et le publie (dataset privé) pour le job.

  python scripts/build_dataset.py                    # -> data/train.jsonl, data/eval.jsonl, data/stats.json
  python scripts/build_dataset.py --push             # + publie sur cfg.hub_dataset (privé)
Lancer scripts/distill.py avant pour inclure les traces de raisonnement de Qwen3.8-27B.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bird_medium import mix  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/mix.yaml")
ap.add_argument("--distill", default="data/distill.jsonl")
ap.add_argument("--out", default="data")
ap.add_argument("--push", action="store_true")
a = ap.parse_args()

cfg = mix.load_config(a.config)
recs = mix.build(cfg, a.distill)
train, ev = mix.split(recs, cfg["eval_fraction"], cfg["seed"])
out = Path(a.out)
mix.write_jsonl(train, out / "train.jsonl")
mix.write_jsonl(ev, out / "eval.jsonl")
stats = {"train": mix.stats(train), "eval": mix.stats(ev), "config": cfg}
(out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
print(json.dumps(stats["train"], ensure_ascii=False, indent=1))

if a.push:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(cfg["hub_dataset"], repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=cfg["hub_dataset"], repo_type="dataset", folder_path=str(out),
                      allow_patterns=["train.jsonl", "eval.jsonl", "stats.json"], commit_message="Bird-Medium: mix SFT")
    print("publié :", f"https://huggingface.co/datasets/{cfg['hub_dataset']}")
