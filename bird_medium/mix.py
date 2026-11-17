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
        d = [r for r in d if distill_ok(r)]
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
    recs = dedupe(recs)
    rng.shuffle(recs)
    return recs


_REPLY_RESIDUE = ("Rédige la réponse de", "Réponds à cet e-mail", "Aide-moi à répondre")


def distill_ok(r: dict) -> bool:
    """Écarte les traces distillées qui citent un autre modèle ou dont le prompt de synthèse est pollué."""
    last = r["messages"][-1]
    if sources.leaks_identity(last["content"]) or sources.leaks_identity(last.get("reasoning_content", "")):
        return False
    if r.get("task") == "summary" and any(x in r["messages"][-2]["content"] for x in _REPLY_RESIDUE):
        return False
    return True


def dedupe(recs: list[dict]) -> list[dict]:
    """Supprime les doublons exacts (dernier échange utilisateur + réponse)."""
    seen, out = set(), []
    for r in recs:
        key = (r["messages"][-2]["content"], r["messages"][-1]["content"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def split(recs: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Évaluation stratifiée par source. Aucune question d'éval n'apparaît à l'identique en train."""
    rng = random.Random(seed)
    by = {}
    for r in recs:
        by.setdefault(r["source"], []).append(r)
    train, ev = [], []
    for items in by.values():
        k = max(1, round(len(items) * frac))
        idx = set(rng.sample(range(len(items)), k))
        ev_q = {items[i]["messages"][-2]["content"] for i in idx}
        for i, r in enumerate(items):
            if i in idx:
                ev.append(r)
            elif r["source"] == "identity" or r["messages"][-2]["content"] not in ev_q:
                # identité : peu de questions, on les garde toutes en train (éval = sondes paraphrasées)
                train.append(r)
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
