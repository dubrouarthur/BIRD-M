"""Generate BIRD-World finance tasks to JSONL + stats.

Usage:
  python scripts/gen_finance.py --n 2000 --seed 0 --out data/out/finance.jsonl
  python scripts/gen_finance.py --n 300 --company-offset 900000 --out data/out/finance_bench.jsonl   # bench split
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bird_m.data.checks import numbers_check  # noqa: E402
from bird_m.data.finance_tasks import generate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--company-offset", type=int, default=0)
ap.add_argument("--fr-share", type=float, default=0.6)
ap.add_argument("--out", required=True)
a = ap.parse_args()

recs = generate(a.n, seed=a.seed, fr_share=a.fr_share, company_offset=a.company_offset)
bad = [
    r["id"]
    for r in recs
    if r["target"] is not None and numbers_check(r["target"], r["facts"])["score"] < 1
]
if bad:
    sys.exit(f"{len(bad)} exact targets fail their own check, e.g. {bad[:3]}")

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

stats = {
    "records": len(recs),
    "by_kind": Counter(r["kind"] for r in recs),
    "by_task": Counter(r["task_id"] for r in recs),
    "by_lang": Counter(r["lang"] for r in recs),
    "by_health": Counter(r["company"]["health"] for r in recs),
    "by_sector": Counter(r["company"]["sector"] for r in recs),
}
out.with_suffix(".stats.json").write_text(
    json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(stats, ensure_ascii=False))
