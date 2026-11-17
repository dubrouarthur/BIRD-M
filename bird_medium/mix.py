"""Assemble le mélange d'entraînement à partir de configs/mix.yaml."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import yaml

from . import finance_fr, identity, sources


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build(cfg: dict, distill_path: str | Path | None = None, log=print) -> list[dict]:
    seed, q = cfg["seed"], cfg["sources"]
    recs: list[dict] = []

    def add(name, items):
        log(f"{name}: {len(items)}")
        recs.extend(items)

    add(
        "identity",
        identity.generate(q["identity"].get("fr", 0), q["identity"].get("en", 0), seed),
    )
    add("finance_fr", finance_fr.generate(q["finance_fr"]["fr"], seed))
    add("emails", sources.emails(q["emails"], seed))
    add("slides", sources.slides(q["slides"], seed))
    add("finance_en", sources.finance_en(q["finance_en"]["en"], seed))
    add("tables_en", sources.tables_en(q["tables_en"]["en"], seed))
    add("general_fr", sources.general_fr(q["general_fr"]["fr"], seed))
    if distill_path and Path(distill_path).exists():
        d = [json.loads(l) for l in Path(distill_path).open(encoding="utf-8")]
        add("distill", d[: q.get("distill", {}).get("fr", len(d))])

    rng = random.Random(seed)
    for r in recs:  # prompt système Bird-Medium sur une partie des exemples (jamais sur l'identité sans)
        if (
            r["messages"][0]["role"] != "system"
            and rng.random() < cfg["system_prompt_share"]
        ):
            r["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": identity.SYSTEM_PROMPT[
                        r["lang"] if r["lang"] in ("fr", "en") else "fr"
                    ],
                },
            )
    rng.shuffle(recs)
    return recs


def split(recs: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Évaluation stratifiée par source."""
    rng = random.Random(seed)
    by = {}
    for r in recs:
        by.setdefault(r["source"], []).append(r)
    train, ev = [], []
    for items in by.values():
        k = max(1, round(len(items) * frac))
        idx = set(rng.sample(range(len(items)), k))
        for i, r in enumerate(items):
            (ev if i in idx else train).append(r)
    rng.shuffle(train)
    return train, ev


def stats(recs: list[dict]) -> dict:
    return {
        "n": len(recs),
        "by_source": dict(Counter(r["source"] for r in recs).most_common()),
        "by_lang": dict(Counter(r["lang"] for r in recs)),
        "thinking": sum(r["thinking"] for r in recs),
        "with_system": sum(r["messages"][0]["role"] == "system" for r in recs),
    }


def write_jsonl(recs: list[dict], path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
