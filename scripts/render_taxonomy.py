"""Generate docs/TASKS.md from data/taxonomy/tasks.yaml (single source of truth).

Usage: python scripts/render_taxonomy.py [--check]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "taxonomy" / "tasks.yaml"
DST = ROOT / "docs" / "TASKS.md"

VALID = {
    "priority": {"P0", "P1", "P2"},
    "thinking": {"direct", "short", "full"},
    "eval": {
        "judge",
        "json_schema",
        "formula_exec",
        "grammar",
        "exact",
        "routing",
        "pairwise",
        "render_judge",
    },
}


def load() -> dict:
    data = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    ids = [t["id"] for t in data["tasks"]]
    dup = [i for i, n in Counter(ids).items() if n > 1]
    if dup:
        raise SystemExit(f"duplicate task ids: {dup}")
    for t in data["tasks"]:
        if t["domain"] not in data["domains"]:
            raise SystemExit(f"{t['id']}: unknown domain {t['domain']}")
        for key, allowed in VALID.items():
            if t[key] not in allowed:
                raise SystemExit(f"{t['id']}: invalid {key}={t[key]}")
    return data


def render(data: dict) -> str:
    tasks = data["tasks"]
    prio = Counter(t["priority"] for t in tasks)
    out = [
        "# Taxonomie des tâches BIRD-M",
        "",
        "> Fichier généré par `scripts/render_taxonomy.py` depuis `data/taxonomy/tasks.yaml`. Ne pas éditer à la main.",
        "",
        f"**{len(tasks)} tâches** · {len(data['domains'])} domaines · "
        f"P0 : {prio['P0']} · P1 : {prio['P1']} · P2 : {prio['P2']} · toutes en FR et EN.",
        "",
        "Légende : **P0** cœur v1 · **P1** v1 si budget · **P2** v1.1 — "
        "*thinking* : `direct` sans raisonnement, `short` court, `full` complet — *éval* : méthode de mesure automatique.",
        "",
    ]
    for key, label in data["domains"].items():
        rows = [t for t in tasks if t["domain"] == key]
        out += [
            f"## {label}",
            "",
            "| ID | Tâche | Prio | Thinking | Éval |",
            "|---|---|---|---|---|",
        ]
        out += [
            f"| `{t['id']}` | {t['fr']} | {t['priority']} | {t['thinking']} | {t['eval']} |"
            for t in rows
        ]
        out.append("")
    return "\n".join(out)


def main() -> None:
    text = render(load())
    if "--check" in sys.argv:
        if not DST.exists() or DST.read_text(encoding="utf-8") != text:
            raise SystemExit("docs/TASKS.md is stale: run scripts/render_taxonomy.py")
        return
    DST.write_text(text, encoding="utf-8")
    print(f"wrote {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
