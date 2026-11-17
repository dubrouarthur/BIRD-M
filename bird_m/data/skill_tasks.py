"""BSP episodes: routing, abstention, composition, policy precedence, generalisation to unseen skills.

The ROUTING decision of every episode is exact by construction (we know which skill the request was
drawn for). The prose answer is left to the teacher (`content: None`) and is generated with the
invoked skill bodies in context.

Episode kinds:
  single   - one relevant skill among near-miss siblings and random distractors
  none     - no relevant skill: generic request, or a request whose skill is ABSENT from the catalog
  compose  - procedure + writing charter for an external text (2 invocations, fixed order)
  policy   - a policy forbids what the user asks: no invocation, the teacher must refuse/mask
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import yaml

from bird_m.skills import Skill, parse_skill
from bird_m.skills.protocol import MAX_INVOCATIONS
from bird_m.world.finance import company_name

ARCHETYPES = Path(__file__).resolve().parents[2] / "data" / "skills" / "archetypes.yaml"
SOURCE = "bird-world/skills@0.1"
EXTERNAL_TEXT = {
    "relance-impayes",
    "relance-devis",
    "reponse-reclamation",
    "geste-commercial",
}  # compose with the charter
KIND_WEIGHTS = {"single": 0.50, "none": 0.25, "compose": 0.12, "policy": 0.13}
SYSTEM = {
    "fr": "Tu es BIRD-M, l'assistant de {company}. Réponds dans la langue de l'utilisateur.",
    "en": "You are BIRD-M, the assistant of {company}. Answer in the user's language.",
}


@lru_cache(maxsize=1)
def library() -> dict:
    return yaml.safe_load(ARCHETYPES.read_text(encoding="utf-8"))


def _skill_md(key: str, a: dict, body: str, stype: str) -> str:
    front = {
        "name": key,
        "description": a["description"]["fr"],
        "type": stype,
        "lang": ["fr", "en"],
        "version": "1.0.0",
    }
    if a.get("when_to_use"):
        front["when_to_use"] = a["when_to_use"]["fr"]
    return (
        "---\n"
        + yaml.safe_dump(front, allow_unicode=True, sort_keys=False)
        + "---\n"
        + body.strip()
        + "\n"
    )


def instantiate(key: str, rng: random.Random) -> tuple[Skill, str]:
    """Company-specific instance of an archetype or policy: (Skill, SKILL.md text)."""
    lib = library()
    a = next((x for x in lib["archetypes"] if x["key"] == key), None)
    if a is None:
        a = next(x for x in lib["policies"] if x["key"] == key)
        md = _skill_md(key, a, a["body"], "policy")
    else:
        params = {k: rng.choice(v) for k, v in (a.get("params") or {}).items()}
        md = _skill_md(key, a, a["body"].format(**params), a["type"])
    return parse_skill(md), md


def _keys(split: str) -> list[str]:
    return [a["key"] for a in library()["archetypes"] if a["split"] == split]


def _family(key: str) -> str:
    return next(a["family"] for a in library()["archetypes"] if a["key"] == key)


def _siblings(key: str) -> list[str]:
    fam = _family(key)
    return [
        a["key"]
        for a in library()["archetypes"]
        if a["family"] == fam and a["key"] != key
    ]


def _requests(key: str, lang: str) -> list[str]:
    return next(a["requests"][lang] for a in library()["archetypes"] if a["key"] == key)


def _catalog(
    rng: random.Random, must: list[str], pool: list[str], exclude: set[str], size: int
) -> list[str]:
    others = [k for k in pool if k not in must and k not in exclude]
    rng.shuffle(others)
    return must + others[: max(0, size - len(must))]


def episode(
    rng: random.Random, split: str = "train", lang: str = "fr", kind: str | None = None
) -> dict:
    lib = library()
    kind = kind or rng.choices(list(KIND_WEIGHTS), list(KIND_WEIGHTS.values()))[0]
    targets = _keys(split)
    # distractors: in train only train skills; in bench, unseen targets mixed with seen distractors
    pool = _keys("train") + (targets if split == "bench" else [])
    size = rng.randint(3, 25)
    invoke: list[str] = []
    facts: dict = {}

    if kind == "single":
        t = rng.choice(targets)
        sib = [s for s in _siblings(t) if s in pool]
        must = [t] + rng.sample(sib, min(len(sib), rng.randint(1, 2)))
        catalog = _catalog(rng, must, pool, set(), size)
        request, invoke = rng.choice(_requests(t, lang)), [t]
    elif kind == "none":
        if rng.random() < 0.5:  # generic request
            request = rng.choice(lib["none_requests"][lang])
            catalog = _catalog(rng, [], pool, set(), size)
        else:  # the right skill exists in the company... but is NOT in this user's catalog
            absent = rng.choice(targets)
            request = rng.choice(_requests(absent, lang))
            catalog = _catalog(rng, [], pool, {absent}, size)
            facts["absent_skill"] = absent
    elif kind == "compose":
        t = (
            rng.choice(sorted(EXTERNAL_TEXT & set(_keys("train"))))
            if split == "train"
            else rng.choice(targets)
        )
        catalog = _catalog(rng, [t, "charte-redactionnelle"], pool, set(), size)
        request = rng.choice(_requests(t, lang))
        invoke = [t, "charte-redactionnelle"] if t != "charte-redactionnelle" else [t]
    elif kind == "policy":
        pol = rng.choice(lib["policies"])
        request = rng.choice(pol["violations"][lang])
        catalog = _catalog(rng, [], pool, set(), size)
        facts["policy"] = pol["key"]
    else:
        raise ValueError(kind)

    policies = [p["key"] for p in lib["policies"]]
    # policies are always active in policy episodes, and often elsewhere (they must not disturb routing)
    active_policies = (
        policies if kind == "policy" else [p for p in policies if rng.random() < 0.5]
    )
    catalog = list(dict.fromkeys(catalog + active_policies))
    rng.shuffle(catalog)

    skills_md = {}
    for key in catalog:
        _, md = instantiate(key, rng)
        skills_md[key] = md
    assert len(invoke) <= MAX_INVOCATIONS and all(k in catalog for k in invoke)

    company = company_name(rng)
    return {
        "task_id": {
            "single": "skill.route",
            "none": "skill.route",
            "compose": "skill.compose",
            "policy": "policy.obey",
        }[kind],
        "kind": kind,
        "lang": lang,
        "split": split,
        "source": SOURCE,
        "system": SYSTEM[lang].format(company=company),
        "catalog": catalog,
        "skills": skills_md,
        "messages": [
            {"role": "user", "content": request},
            {
                "role": "assistant",
                "invoke": invoke,
                "content": None,
            },  # routing exact; prose by the teacher
        ],
        "facts": facts,
    }


def generate(
    n: int, seed: int = 0, split: str = "train", fr_share: float = 0.6
) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        lang = "fr" if rng.random() < fr_share else "en"
        ep = episode(rng, split=split, lang=lang)
        ep["id"] = f"bsp-{split}-{seed}-{i:06d}"
        out.append(ep)
    return out
