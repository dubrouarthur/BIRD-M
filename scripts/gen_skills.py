"""Generate BSP skill episodes to JSONL + stats.

Usage:
  python scripts/gen_skills.py --n 3000 --seed 0 --split train --out data/out/skills_pilot.jsonl
  python scripts/gen_skills.py --n 500 --seed 1 --split bench --out data/out/skills_bench.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bird_m.data.skill_tasks import generate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--split", choices=["train", "bench"], default="train")
ap.add_argument("--out", required=True)
a = ap.parse_args()

eps = generate(a.n, seed=a.seed, split=a.split)
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for e in eps:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

stats = {
    "records": len(eps),
    "by_kind": Counter(e["kind"] for e in eps),
    "by_lang": Counter(e["lang"] for e in eps),
    "invocations": Counter(len(e["messages"][1]["invoke"]) for e in eps),
    "targets": Counter(k for e in eps for k in e["messages"][1]["invoke"]),
    "catalog_size": {
        "min": min(len(e["catalog"]) for e in eps),
        "max": max(len(e["catalog"]) for e in eps),
    },
}
out.with_suffix(".stats.json").write_text(
    json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(
    json.dumps(
        {k: stats[k] for k in ("records", "by_kind", "invocations", "catalog_size")},
        ensure_ascii=False,
    )
)
