"""Enterprise skill definition: a SKILL.md file (YAML front-matter + Markdown body).

Compatible with the open Agent Skills layout (`name`, `description` required);
BSP adds optional enterprise fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import spec

SKILL_TYPES = ("procedure", "style", "knowledge", "template", "policy")
COMPUTE_LEVELS = ("direct", "short", "full")  # reasoning budget applied when the skill is invoked
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_FRONT_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


class SkillError(ValueError):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    type: str = "procedure"
    when_to_use: str = ""
    lang: tuple[str, ...] = ("fr", "en")
    version: str = "1.0.0"
    owner: str = ""
    roles: tuple[str, ...] = ()  # empty = everyone
    priority: int = 0
    compute: str = "direct"  # direct | short | full (skill-driven reasoning budget)

    @property
    def is_policy(self) -> bool:
        return self.type == "policy"

    def allowed_for(self, user_roles: set[str]) -> bool:
        return not self.roles or bool(set(self.roles) & user_roles)


_RESERVED_RE = re.compile(spec.RESERVED_PATTERN)
_SINGLE_LINE = ("description", "when_to_use", "version", "owner")


def _str(meta: dict, key: str, default: str = "") -> str:
    v = meta.get(key, default)
    if not isinstance(v, str):
        raise SkillError(f"field {key!r} must be a string")
    v = v.strip()
    if key in _SINGLE_LINE and ("\n" in v or "\r" in v):
        raise SkillError(f"field {key!r} must be a single line")
    if _RESERVED_RE.search(v):
        raise SkillError(f"field {key!r} contains a reserved token-like sequence")
    return v


def _strs(meta: dict, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if key not in meta:
        return default
    v = meta[key]
    v = [v] if isinstance(v, str) else v
    if not isinstance(v, list) or not v or not all(isinstance(x, str) and x.strip() for x in v):
        raise SkillError(f"field {key!r} must be a non-empty string or list of strings")
    return tuple(x.strip() for x in v)


def parse_skill(text: str) -> Skill:
    m = _FRONT_RE.match(text)
    if not m:
        raise SkillError("SKILL.md must start with a '---' YAML front-matter block")
    try:
        # BaseLoader: every scalar stays a string (no YAML 1.1 'no' -> False, '1.10' -> 1.1).
        meta = yaml.load(m.group(1), Loader=yaml.BaseLoader) or {}
    except yaml.YAMLError as e:
        raise SkillError(f"invalid YAML front-matter: {e}") from e
    if not isinstance(meta, dict):
        raise SkillError("front-matter must be a mapping")
    body = m.group(2).strip()

    for key in ("name", "description"):
        if not _str(meta, key):
            raise SkillError(f"missing required field: {key}")
    name = _str(meta, "name")
    if not _NAME_RE.match(name):
        raise SkillError(f"invalid skill name {name!r} (lowercase, digits, '-')")
    stype = _str(meta, "type", "procedure")
    if stype not in SKILL_TYPES:
        raise SkillError(f"invalid type {stype!r}, expected one of {SKILL_TYPES}")
    if not body:
        raise SkillError("skill body is empty")
    compute = _str(meta, "compute", "direct")
    if compute not in COMPUTE_LEVELS:
        raise SkillError(f"invalid compute {compute!r}, expected one of {COMPUTE_LEVELS}")
    raw_prio = _str(meta, "priority", "0")
    if not re.fullmatch(r"-?\d+", raw_prio):
        raise SkillError(f"priority must be an integer, got {raw_prio!r}")

    return Skill(
        name=name,
        description=_str(meta, "description"),
        body=body,
        type=stype,
        when_to_use=_str(meta, "when_to_use"),
        lang=_strs(meta, "lang", ("fr", "en")),
        version=_str(meta, "version", "1.0.0"),
        owner=_str(meta, "owner"),
        roles=_strs(meta, "roles", ()),  # absent = everyone; explicit empty list is rejected
        priority=int(raw_prio),
        compute=compute,
    )


def load_skill(path: str | Path) -> Skill:
    p = Path(path)
    if p.is_dir():
        p = p / "SKILL.md"
    return parse_skill(p.read_text(encoding="utf-8"))


@dataclass
class SkillLibrary:
    skills: dict[str, Skill] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, root: str | Path) -> "SkillLibrary":
        lib = cls()
        for f in sorted(Path(root).glob("*/SKILL.md")):
            try:
                lib.add(load_skill(f))
            except SkillError as e:
                raise SkillError(f"{f}: {e}") from e
        return lib

    def add(self, skill: Skill) -> None:
        if skill.name in self.skills:
            raise SkillError(f"duplicate skill name: {skill.name}")
        self.skills[skill.name] = skill

    def visible_to(self, user_roles: set[str]) -> list[Skill]:
        return [s for s in self.skills.values() if s.allowed_for(user_roles)]
