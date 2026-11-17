"""Baseline evaluation of an off-the-shelf chat model on BIRD-Bench (no BSP tokens).

- finance: exact-answer tasks; score = numbers_check against computed facts
- skills : the catalog is pasted as TEXT (ablation condition "b"); the model must name the skill(s)
           to use, or AUCUNE/NONE; compared with the exact routing ground truth
Every result keeps token usage so quality AND cost are reported together (north-star metric).
"""

from __future__ import annotations

import re
from collections import defaultdict

from bird_m.data.checks import numbers_check
from bird_m.skills import parse_skill

ROUTE_INSTR = {
    "fr": (
        "Voici les procédures internes (skills) disponibles. Pour la demande de l'utilisateur, réponds UNIQUEMENT "
        "par le ou les noms de skills à appliquer, séparés par des virgules, ou par AUCUNE si aucune ne s'applique."
    ),
    "en": (
        "Here are the available internal procedures (skills). For the user's request, answer ONLY with the name(s) "
        "of the skills to apply, comma-separated, or NONE if none applies."
    ),
}


def finance_items(records: list[dict]) -> list[dict]:
    return [r for r in records if r["kind"] == "exact"]


def finance_messages(r: dict) -> list[dict]:
    return r["messages"]


def score_finance(r: dict, answer: str) -> dict:
    res = numbers_check(answer, r["facts"])
    return {
        "task_id": r["task_id"],
        "lang": r["lang"],
        "score": res["score"],
        "exact": res["score"] == 1.0,
        "missing": res["missing"],
    }


def skills_messages(e: dict) -> list[dict]:
    lang = e["lang"]
    cards = []
    for name in e["catalog"]:
        sk = parse_skill(e["skills"][name])
        if sk.is_policy:
            continue
        line = f"- {sk.name} : {sk.description}"
        if sk.when_to_use:
            line += f" (quand : {sk.when_to_use})"
        cards.append(line)
    policies = [
        parse_skill(e["skills"][n])
        for n in e["catalog"]
        if parse_skill(e["skills"][n]).is_policy
    ]
    pol = "".join(f"\n[Politique {p.name}] {p.body}" for p in policies)
    system = f"{e['system']}{pol}\n\n{ROUTE_INSTR[lang]}\n" + "\n".join(cards)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": e["messages"][0]["content"]},
    ]


_NONE_RE = re.compile(r"\b(aucune|aucun|none)\b", re.I)


def parse_route(answer: str, catalog: list[str]) -> set[str]:
    text = answer.lower()
    found = {
        n
        for n in catalog
        if re.search(rf"(?<![a-z0-9-]){re.escape(n)}(?![a-z0-9-])", text)
    }
    if not found and _NONE_RE.search(text):
        return set()
    return found


def score_skills(e: dict, answer: str) -> dict:
    want = set(e["messages"][1]["invoke"])
    got = parse_route(
        answer, [n for n in e["catalog"] if not e["skills"][n].count("type: policy")]
    )
    return {
        "kind": e["kind"],
        "lang": e["lang"],
        "exact": got == want,
        "want": sorted(want),
        "got": sorted(got),
        "false_positive": bool(got) and not want,
        "missed": bool(want) and not got,
    }


def aggregate(rows: list[dict], by: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[by]].append(r)
        groups["ALL"].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        tok = [
            r["completion_tokens"] for r in rs if r.get("completion_tokens") is not None
        ]
        out[g] = {
            "n": len(rs),
            "exact_rate": round(sum(r["exact"] for r in rs) / len(rs), 3),
            "avg_completion_tokens": round(sum(tok) / len(tok), 1) if tok else None,
            "errors": sum(1 for r in rs if r.get("error")),
        }
    return out
