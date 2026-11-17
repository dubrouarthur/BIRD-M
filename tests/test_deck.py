import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from bird_m.deck import is_valid, render_deck, validate_deck
from bird_m.deck.render import fmt_num

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples" / "decks").glob("*.json"))
LUMEN = json.loads(
    (ROOT / "examples" / "decks" / "resultats-t3-lumen.json").read_text(
        encoding="utf-8"
    )
)


def deck(*slides, **meta):
    return {
        "version": "bird-deck/0.1",
        "meta": {"title": "T", "lang": "fr", **meta},
        "slides": list(slides),
    }


def errors(d):
    return [i for i in validate_deck(d) if i.level == "error"]


# ------------------------------------------------------------------ examples
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_examples_are_valid_and_render(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    assert validate_deck(d) == []
    html = render_deck(d)
    assert html.count('<section class="slide') == len(d["slides"])


def test_reference_deck_uses_every_layout():
    layouts = set(
        json.loads((ROOT / "bird_m" / "deck" / "schema.json").read_text())["$defs"][
            "slide"
        ]["properties"]["layout"]["enum"]
    )
    assert {s["layout"] for s in LUMEN["slides"]} == layouts


# ------------------------------------------------------------------ validation
def test_density_limits_are_enforced():
    d = deck({"layout": "bullets", "title": "x", "bullets": ["mot " * 17, "b"]})
    assert any("too dense" in i.message for i in errors(d))
    d = deck(
        {
            "layout": "bullets",
            "title": "x",
            "bullets": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )
    assert errors(d)


@pytest.mark.parametrize(
    "slide",
    [
        {"layout": "nope", "title": "x"},
        {"layout": "bullets", "title": "x", "bullets": ["a", "b"], "extra": 1},
        {
            "layout": "chart",
            "title": "x",
            "takeaway": "t",
            "chart": {
                "type": "bar",
                "categories": ["a", "b"],
                "series": [{"name": "s", "values": [1]}],
            },
        },
        {
            "layout": "chart",
            "title": "x",
            "takeaway": "t",
            "chart": {
                "type": "donut",
                "categories": ["a", "b"],
                "series": [{"name": "s", "values": [1, -2]}],
            },
        },
        {
            "layout": "chart",
            "title": "x",
            "takeaway": "t",
            "chart": {
                "type": "bar",
                "categories": ["a", "b"],
                "series": [{"name": "s", "values": [1, 2]}],
                "highlight": 5,
            },
        },
        {
            "layout": "table",
            "title": "x",
            "header": ["a", "b"],
            "rows": [["1", "2", "3"]],
        },
        {
            "layout": "comparison",
            "title": "x",
            "options": [
                {"heading": "a", "points": ["p"], "recommended": True},
                {"heading": "b", "points": ["q"], "recommended": True},
            ],
        },
        {
            "layout": "kpis",
            "title": "x",
            "kpis": [
                {"value": "123456789", "label": "l"},
                {"value": "1", "label": "m"},
            ],
        },
    ],
)
def test_invalid_slides_rejected(slide):
    assert not is_valid(deck(slide))


def test_theme_colour_must_be_hex():
    d = deck({"layout": "title", "title": "x"})
    d["theme"] = {"primary": "red;}body{display:none"}
    assert errors(d)


def test_design_warnings():
    b = {"layout": "bullets", "title": "x", "bullets": ["a", "b"]}
    issues = validate_deck(deck(b, b, b, b, b))
    msgs = " ".join(i.message for i in issues if i.level == "warning")
    assert "title slide" in msgs and "in a row" in msgs and "half" in msgs
    assert is_valid(deck(b, b, b))  # warnings never invalidate


def test_garbage_input_does_not_crash():
    for bad in (
        None,
        [],
        "deck",
        {"version": "bird-deck/0.1"},
        {"version": "bird-deck/0.1", "meta": {}, "slides": [{}]},
    ):
        assert errors(bad)


# ------------------------------------------------------------------ rendering
def test_render_escapes_all_text():
    d = deck(
        {
            "layout": "title",
            "title": "<script>alert(1)</script>",
            "subtitle": '"><img src=x onerror=1>',
        }
    )
    html = render_deck(d)
    assert "<script>alert" not in html and "<img src=x" not in html
    assert "&lt;script&gt;" in html


def test_render_ignores_invalid_colours_even_unvalidated():
    d = deck({"layout": "title", "title": "x"})
    d["theme"] = {"primary": "red;}body{display:none"}
    html = render_deck(d)
    assert "red;}body" not in html and "--primary:#1E2A78" in html  # fell back to preset


def test_numbers_formatted_per_locale():
    assert fmt_num(18400.5, "fr", "€") == "18 400,5 €"
    assert fmt_num(18400.5, "en", "€") == "18,400.5€"
    assert fmt_num(12, "fr", "%") == "12 %"


def test_every_chart_type_renders():
    for t in ("bar", "line", "stacked_bar", "donut"):
        series = [{"name": "a", "values": [1, 2, 3]}] + (
            [] if t == "donut" else [{"name": "b", "values": [2, -1, 4]}]
        )
        d = deck(
            {
                "layout": "chart",
                "title": "x",
                "takeaway": "y",
                "chart": {"type": t, "categories": ["a", "b", "c"], "series": series},
            }
        )
        if t == "donut":
            assert is_valid(d)
        assert "<svg" in render_deck(d)


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_reference_deck_has_no_visual_overflow(tmp_path):
    """End-to-end: render in Chromium and measure clipping / overflow on every slide."""
    probe = subprocess.run(["node", "-e", "require('playwright')"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("playwright not available")
    html = tmp_path / "deck.html"
    html.write_text(render_deck(LUMEN), encoding="utf-8")
    r = subprocess.run(
        [
            "node",
            str(ROOT / "scripts" / "deck_check.mjs"),
            str(html),
            str(tmp_path / "png"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(list((tmp_path / "png").glob("slide-*.png"))) == len(LUMEN["slides"])


def test_overflow_detector_catches_a_real_overflow(tmp_path):
    """The checker must fail when text overflows; otherwise the reward signal is worthless."""
    if (
        not shutil.which("node")
        or subprocess.run(
            ["node", "-e", "require('playwright')"], capture_output=True
        ).returncode
    ):
        pytest.skip("node/playwright not available")
    d = copy.deepcopy(LUMEN)
    d["slides"] = [
        {"layout": "bullets", "title": "x", "bullets": ["mot " * 16] * 6}
    ]  # bypasses nothing: valid but maximal
    html = render_deck(d).replace(
        "font-size:23px", "font-size:60px"
    )  # force an overflow
    p = tmp_path / "over.html"
    p.write_text(html, encoding="utf-8")
    r = subprocess.run(
        ["node", str(ROOT / "scripts" / "deck_check.mjs"), str(p), str(tmp_path / "o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 1, r.stdout


def test_core_skill_bird_deck_is_valid_and_matches_schema():
    from bird_m.skills import load_skill

    sk = load_skill(ROOT / "skills" / "core" / "bird-deck")
    layouts = json.loads((ROOT / "bird_m" / "deck" / "schema.json").read_text())["$defs"]["slide"]["properties"]["layout"]["enum"]
    for layout in layouts:  # the skill documents every layout the schema accepts
        assert f"`{layout}`" in sk.body, layout
