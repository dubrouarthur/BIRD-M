"""Measure an off-the-shelf model on BIRD-Bench (finance exact + skill routing), in direct and thinking modes.

Usage (needs Hugging Face Inference Providers credits):
  python scripts/eval_base.py --model "Qwen/Qwen3.8-27B:scaleway" --n 100 --modes direct,think --out reports/base_qwen38
Outputs <out>.json (per-item + aggregates) and <out>.md (summary table).
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bird_m.data.finance_tasks import generate as gen_fin  # noqa: E402
from bird_m.data.skill_tasks import generate as gen_skills  # noqa: E402
from bird_m.eval.baseline import (  # noqa: E402
    aggregate,
    finance_items,
    finance_messages,
    score_finance,
    score_skills,
    skills_messages,
)
from bird_m.llm import ChatClient  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=100, help="items per suite")
ap.add_argument("--modes", default="direct,think")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()

# Bench data is regenerated deterministically (same seeds as the published bench files).
fin = finance_items(gen_fin(1000, seed=1, company_offset=900_000))
skills = gen_skills(1000, seed=1, split="bench")
rng = random.Random(0)
fin, skills = (
    rng.sample(fin, min(a.n, len(fin))),
    rng.sample(skills, min(a.n, len(skills))),
)

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
report = {"model": a.model, "n": a.n, "results": {}}
for mode in a.modes.split(","):
    client = ChatClient(
        model=a.model,
        thinking=(mode == "think"),
        max_tokens=8192 if mode == "think" else 2048,
        cache_path=out.with_name(out.name + f".{mode}.cache.jsonl"),
    )
    rows = {}
    for suite, items, to_msgs, scorer, by in (
        ("finance", fin, finance_messages, score_finance, "task_id"),
        ("skills", skills, skills_messages, score_skills, "kind"),
    ):
        answers = client.map([to_msgs(x) for x in items], workers=a.workers)
        scored = []
        for x, ans in zip(items, answers):
            if isinstance(ans, Exception):
                if any(k in str(ans) for k in ("credits", "HTTP 402", "HTTP 401")):
                    sys.exit(f"Inference Providers credits exhausted/missing: {ans}")
                scored.append(
                    {
                        by: x.get(by, x.get("kind")),
                        "exact": False,
                        "error": str(ans)[:200],
                        "completion_tokens": None,
                    }
                )
                continue
            s = scorer(x, ans["content"])
            s["completion_tokens"] = ans["usage"].get("completion_tokens")
            scored.append(s)
        rows[suite] = {"aggregate": aggregate(scored, by), "items": scored}
    report["results"][mode] = rows

out.with_suffix(".json").write_text(
    json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
)
lines = [
    f"# Baseline {a.model}",
    "",
    "| Mode | Suite | Groupe | n | Exact | Tokens générés (moy.) | Erreurs |",
    "|---|---|---|---:|---:|---:|---:|",
]
for mode, suites in report["results"].items():
    for suite, r in suites.items():
        for g, v in r["aggregate"].items():
            lines.append(
                f"| {mode} | {suite} | {g} | {v['n']} | {v['exact_rate']:.1%} | {v['avg_completion_tokens']} | {v['errors']} |"
            )
out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
