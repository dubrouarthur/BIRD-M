"""Lance un script (train_job.py, merge_job.py) sur Hugging Face Jobs avec les volumes Bird-Medium.

  python scripts/launch.py train --flavor h200 --timeout 6h -- --epochs 1
  python scripts/launch.py merge --flavor h200 --timeout 2h -- --adapter /out/bird-medium-v1/adapter --out /out/bird-medium-v1/merged
Les arguments après « -- » sont passés au script. Le job n'a pas besoin de jeton : données et sorties sont montées.
"""

import argparse
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, Volume

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {"train": "scripts/train_job.py", "merge": "scripts/merge_job.py"}

argv = sys.argv[1:]
extra = argv[argv.index("--") + 1:] if "--" in argv else []
argv = argv[: argv.index("--")] if "--" in argv else argv
ap = argparse.ArgumentParser()
ap.add_argument("what", choices=SCRIPTS)
ap.add_argument("--flavor", default="h200")
ap.add_argument("--timeout", default="6h")
ap.add_argument("--config", default="configs/mix.yaml")
a = ap.parse_args(argv)
cfg = yaml.safe_load((ROOT / a.config).read_text())

api = HfApi()
job = api.run_uv_job(
    str(ROOT / SCRIPTS[a.what]),
    script_args=["--model", cfg["base_model"], *extra] if a.what == "train" else ["--base", cfg["base_model"], *extra],
    flavor=a.flavor,
    timeout=a.timeout,
    env={"PYTHONUNBUFFERED": "1"},
    volumes=[
        Volume(type="dataset", source=cfg["hub_dataset"], mount_path="/dataset"),  # /data : réservé par Jobs
        Volume(type="bucket", source=cfg["bucket"], mount_path="/out", read_only=False),
    ],
    labels={"project": "bird-medium", "step": a.what},
)
print("job :", job.id, job.url if hasattr(job, "url") else "")
print("logs :", f"python -c \"from huggingface_hub import HfApi; [print(l) for l in HfApi().fetch_job_logs(job_id='{job.id}')]\"")
