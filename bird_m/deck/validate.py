"""BIRD-Deck validation: JSON-Schema structure + readability (x-maxWords) + semantic rules.

Errors make a deck invalid (used as a hard reward signal / data filter).
Warnings are design smells (monotone deck, missing title slide) used as soft penalties.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).with_name("schema.json")


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    path: str  # e.g. "slides[3].bullets[1]"
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.path}: {self.message}"


@lru_cache(maxsize=1)
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=32)
def _validator(ref: str) -> Draft202012Validator:
    s = schema()
    sub = s if ref == "#" else {"$defs": s["$defs"], "$ref": ref}
    return Draft202012Validator(sub)


def _fmt_path(prefix: str, parts) -> str:
    out = prefix
    for p in parts:
        out += f"[{p}]" if isinstance(p, int) else f".{p}"
    return out.lstrip(".")


def _resolve(node: dict) -> dict:
    while "$ref" in node:
        node = schema()["$defs"][node["$ref"].split("/")[-1]]
    return node


def _word_limits(node: dict, value, path: str, issues: list[Issue]) -> None:
    node = _resolve(node)
    if isinstance(value, str) and "x-maxWords" in node:
        n = len(value.split())
        if n > node["x-maxWords"]:
            issues.append(
                Issue("error", path, f"{n} words > {node['x-maxWords']} (too dense)")
            )
    elif isinstance(value, dict) and "properties" in node:
        for k, v in value.items():
            if k in node["properties"]:
                _word_limits(node["properties"][k], v, f"{path}.{k}", issues)
    elif isinstance(value, list) and "items" in node:
        for i, v in enumerate(value):
            _word_limits(node["items"], v, f"{path}[{i}]", issues)


def _semantic(slide: dict, path: str, issues: list[Issue]) -> None:
    layout = slide["layout"]
    if layout == "chart":
        ch = slide["chart"]
        n = len(ch["categories"])
        for i, s in enumerate(ch["series"]):
            if len(s["values"]) != n:
                issues.append(
                    Issue(
                        "error",
                        f"{path}.chart.series[{i}]",
                        f"{len(s['values'])} values for {n} categories",
                    )
                )
        if ch["type"] == "donut":
            if len(ch["series"]) != 1:
                issues.append(
                    Issue("error", f"{path}.chart", "donut takes exactly one series")
                )
            if any(v < 0 for s in ch["series"] for v in s["values"]):
                issues.append(
                    Issue("error", f"{path}.chart", "donut values must be >= 0")
                )
            if n > 6:
                issues.append(
                    Issue("error", f"{path}.chart", "donut readable up to 6 categories")
                )
        if ch["type"] == "stacked_bar" and len(ch["series"]) < 2:
            issues.append(
                Issue(
                    "warning", f"{path}.chart", "stacked_bar with one series: use bar"
                )
            )
        if "highlight" in ch and ch["highlight"] >= n:
            issues.append(
                Issue("error", f"{path}.chart.highlight", "index out of range")
            )
    elif layout == "table":
        w = len(slide["header"])
        for i, row in enumerate(slide["rows"]):
            if len(row) != w:
                issues.append(
                    Issue(
                        "error",
                        f"{path}.rows[{i}]",
                        f"{len(row)} cells for {w} columns",
                    )
                )
        if slide.get("highlight_row", 0) >= len(slide["rows"]):
            issues.append(Issue("error", f"{path}.highlight_row", "index out of range"))
    elif layout == "comparison":
        if sum(1 for o in slide["options"] if o.get("recommended")) > 1:
            issues.append(
                Issue("error", f"{path}.options", "at most one recommended option")
            )


def validate_deck(deck) -> list[Issue]:
    issues: list[Issue] = []
    for e in _validator("#").iter_errors(deck):
        issues.append(Issue("error", _fmt_path("", e.absolute_path) or "$", e.message))
    if issues or not isinstance(deck, dict):
        return issues  # structure broken: deeper checks would be noise

    _word_limits(schema(), deck, "", issues)
    slides = deck["slides"]
    for i, slide in enumerate(slides):
        path = f"slides[{i}]"
        before = len(issues)
        for e in _validator(f"#/$defs/layout_{slide['layout']}").iter_errors(slide):
            issues.append(Issue("error", _fmt_path(path, e.absolute_path), e.message))
        if len(issues) == before:
            _word_limits(
                {"$ref": f"#/$defs/layout_{slide['layout']}"}, slide, path, issues
            )
            _semantic(slide, path, issues)

    if slides[0]["layout"] != "title":
        issues.append(
            Issue("warning", "slides[0]", "deck should open with a title slide")
        )
    run = 0
    for i, s in enumerate(slides):
        run = run + 1 if s["layout"] == "bullets" else 0
        if run == 3:
            issues.append(
                Issue(
                    "warning", f"slides[{i}]", "3 bullet slides in a row: vary layouts"
                )
            )
    if (
        len(slides) >= 5
        and sum(s["layout"] == "bullets" for s in slides) / len(slides) > 0.5
    ):
        issues.append(
            Issue("warning", "slides", "more than half the slides are bullet lists")
        )
    return issues


def is_valid(deck) -> bool:
    return not any(i.level == "error" for i in validate_deck(deck))
