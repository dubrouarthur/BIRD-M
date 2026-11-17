"""Publie un dossier du bucket (adaptateur ou modèle fusionné) dans le repo modèle, fichier par fichier.

  python scripts/publish.py bird-medium-v1/merged            # -> cfg.hub_model (adubrou/BIRD-M, privé)
  python scripts/publish.py bird-medium-v1/adapter --card model_card/ADAPTER.md   # adaptateur seul
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
ap.add_argument("--force", action="store_true", help="republie même les fichiers déjà présents (nouveau run)")
ap.add_argument("--card", default=str(ROOT / "model_card/README.md"))
a = ap.parse_args()
cfg = yaml.safe_load((ROOT / "configs/mix.yaml").read_text())
repo = a.repo or cfg["hub_model"]

api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
prefix = a.prefix.rstrip("/") + "/"  # évite de capter un dossier voisin (adapter_old/)
existing = {} if a.force else {
    f.path: f.size for f in api.list_repo_tree(repo, recursive=True) if getattr(f, "size", None) is not None
}
files = [(f.path, f.size) for f in api.list_bucket_tree(cfg["bucket"], prefix=prefix, recursive=True)
         if getattr(f, "type", "file") == "file"]
if not files:
    sys.exit(f"rien sous {cfg['bucket']}/{a.prefix}")
with tempfile.TemporaryDirectory() as tmp:
    for f, size in files:
        name = f[len(prefix):]
        if existing.get(name) == size:  # même nom ET même taille : déjà publié
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
