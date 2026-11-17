"""Enterprise skill definition: a SKILL.md file (YAML front-matter + Markdown body).

Compatible with the open Agent Skills layout (`name`, `description` required);
BSP adds optional enterprise fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SKILL_TYPES = ("procedure", "style", "knowledge", "template", "policy")
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

    @property
    def is_policy(self) -> bool:
        return self.type == "policy"

    def allowed_for(self, user_roles: set[str]) -> bool:
        return not self.roles or bool(set(self.roles) & user_roles)


def parse_skill(text: str) -> Skill:
    m = _FRONT_RE.match(text)
    if not m:
        raise SkillError("SKILL.md must start with a '---' YAML front-matter block")
    meta = yaml.safe_load(m.group(1)) or {}
    if not isinstance(meta, dict):
        raise SkillError("front-matter must be a mapping")
    body = m.group(2).strip()

    for key in ("name", "description"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise SkillError(f"missing required field: {key}")
    name = meta["name"].strip()
    if not _NAME_RE.match(name):
        raise SkillError(f"invalid skill name {name!r} (lowercase, digits, '-')")
    stype = meta.get("type", "procedure")
    if stype not in SKILL_TYPES:
        raise SkillError(f"invalid type {stype!r}, expected one of {SKILL_TYPES}")
    if not body:
        raise SkillError("skill body is empty")

    def _tuple(v, default):
        if v is None:
            return default
        return (v,) if isinstance(v, str) else tuple(str(x) for x in v)

    return Skill(
        name=name,
        description=meta["description"].strip(),
        body=body,
        type=stype,
        when_to_use=str(meta.get("when_to_use", "")).strip(),
        lang=_tuple(meta.get("lang"), ("fr", "en")),
        version=str(meta.get("version", "1.0.0")),
        owner=str(meta.get("owner", "")),
        roles=_tuple(meta.get("roles"), ()),
        priority=int(meta.get("priority", 0)),
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
            lib.add(load_skill(f))
        return lib

    def add(self, skill: Skill) -> None:
        if skill.name in self.skills:
            raise SkillError(f"duplicate skill name: {skill.name}")
        self.skills[skill.name] = skill

    def visible_to(self, user_roles: set[str]) -> list[Skill]:
        return [s for s in self.skills.values() if s.allowed_for(user_roles)]
