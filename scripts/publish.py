"""Publie un dossier du bucket (adaptateur ou modèle fusionné) dans le repo modèle, fichier par fichier.

  python scripts/publish.py bird-medium-v1/merged            # -> cfg.hub_model (privé)
  python scripts/publish.py bird-medium-v1/adapter --repo adubrou/Bird-Medium-LoRA
Le disque local est limité : un fichier à la fois, supprimé après envoi. Les fichiers déjà publiés sont sautés.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import yaml
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--repo")
ap.add_argument("--card", default=str(ROOT / "model_card/README.md"))
a = ap.parse_args()
cfg = yaml.safe_load((ROOT / "configs/mix.yaml").read_text())
repo = a.repo or cfg["hub_model"]

api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
existing = set(api.list_repo_files(repo))
files = [f.path for f in api.list_bucket_tree(cfg["bucket"], prefix=a.prefix, recursive=True) if getattr(f, "type", "file") == "file"]
if not files:
    sys.exit(f"rien sous {cfg['bucket']}/{a.prefix}")
with tempfile.TemporaryDirectory() as tmp:
    for f in files:
        name = f[len(a.prefix):].lstrip("/")
        if name in existing:
            print("déjà publié", name)
            continue
        dst = Path(tmp) / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        api.download_bucket_files(cfg["bucket"], [(f, str(dst))])
        api.upload_file(path_or_fileobj=str(dst), path_in_repo=name, repo_id=repo, commit_message=f"Bird-Medium: {name}")
        dst.unlink()
        print("publié", name)
if Path(a.card).exists():
    api.upload_file(path_or_fileobj=a.card, path_in_repo="README.md", repo_id=repo, commit_message="Bird-Medium: fiche modèle")
print("ok :", f"https://huggingface.co/{repo}")
