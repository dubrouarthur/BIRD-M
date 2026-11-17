"""Build the SFT training mix (JSONL) from our generators + the open corpus + converted slides.

Usage: python scripts/build_mix.py --out data/out/mix_v1.jsonl
Caps are chosen to fit the token budget of a pilot run (~4.8M tokens on Qwen3.8-27B).
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser()
ap.add_argument("--open", default="data/out/open_corpus.jsonl")
ap.add_argument("--slides", default="data/out/slides_officesmith.jsonl")
ap.add_argument("--skills", default="data/out/skills_pilot.jsonl")
ap.add_argument("--finance", default="data/out/finance_pilot.jsonl")
ap.add_argument("--identity", default="data/out/identity.jsonl")
ap.add_argument("--no-system-share", type=float, default=0.4)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

CAPS = {  # per source key (prefix of the record id) and language
    "sinabis_emails": {"fr": 1000, "en": 300}, "smoltalk_ml8_fr": {"fr": 500}, "helpsteer3_edit": {"fr": 100, "en": 100},
    "oasst2": {"fr": 321, "en": 280}, "aya_human": {"fr": 27, "en": 373}, "smoltalk_systemchats": {"fr": 17, "en": 283},
    "sinabis_docs": {"fr": 100, "en": 100}, "structured_extraction": {"fr": 74, "en": 126}, "smoltalk_tables": {"en": 150},
    "smoltalk_aya_qwen": {"fr": 250, "en": 50}, "officesmith": {"fr": 200, "en": 100},
    "identity": {"fr": 274, "en": 126}, "skills": {"fr": 600, "en": 400}, "finance": {"fr": 300, "en": 200},
}

rng = random.Random(a.seed)
pools = defaultdict(list)


def key_of(r):
    if r["id"].startswith("officesmith-"):
        return "officesmith"
    if r.get("catalog"):
        return "skills"
    if r["id"].startswith("identity-"):
        return "identity"
    if r["id"].startswith("bw-"):
        return "finance"
    return r["id"].rsplit("-", 2)[0]


for path in (a.open, a.slides, a.skills, a.finance, a.identity):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("catalog") is None and r.get("target") is None:
            continue  # teacher answer still pending
        pools[(key_of(r), r["lang"])].append(r)

mix, stats = [], Counter()
for (k, lang), pool in sorted(pools.items()):
    n = CAPS.get(k, {}).get(lang, 0)
    rng.shuffle(pool)
    mix += pool[:n]
    stats[f"{k}/{lang}"] = min(n, len(pool))
rng.shuffle(mix)
# a share of open-source records is trained with NO system prompt, so behaviour does not depend on one
import hashlib  # noqa: E402

for r in mix:
    if key_of(r) not in ("skills", "finance", "identity") and not r.get("system"):
        if int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % 100 < a.no_system_share * 100:
            r["no_system"] = True
stats["no_system"] = sum(1 for r in mix if r.get("no_system"))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for r in mix:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
Path(a.out).with_suffix(".stats.json").write_text(json.dumps(dict(stats), indent=1), encoding="utf-8")
print(len(mix), json.dumps(dict(sorted(stats.items())), ensure_ascii=False))
