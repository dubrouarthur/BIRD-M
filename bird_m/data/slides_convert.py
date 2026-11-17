"""Convert OfficeSmith `pptx-ir/v0.1` decks (Apache-2.0, Qwen3.5-397B teacher) into BIRD-Deck.

There is no industry-standard JSON for slides (the standard is the .pptx file itself); pptx-ir is one
dataset's own format. Layouts map closely to BIRD-Deck; our density rules are enforced and every
converted deck must pass bird_m.deck.validate, otherwise the offending slides are dropped.
"""

from __future__ import annotations

import json
import re

from bird_m.deck.validate import validate_deck

_BULLET = re.compile(r"^[•\-–*·]\s*")


def _lines(elements, slots=None) -> list[str]:
    out = []
    for e in elements:
        if e["type"] != "text" or e.get("slot") == "source_note" or (slots and e.get("slot") not in slots):
            continue
        for ln in (e.get("text") or "").split("\n"):
            ln = _BULLET.sub("", ln).strip().rstrip(".")
            if ln and not re.match(r"(source|sources)\s*:", ln, re.I) and "SYNTH-" not in ln:
                out.append(ln)
    return out


def _short(lines, max_words):
    return [ln for ln in lines if len(ln.split()) <= max_words]


def _title(t: str) -> str:
    words = t.split()
    return " ".join(words[:14])


def convert_slide(s: dict) -> dict | None:
    lay, title, els = s["layout"], _title(s.get("title") or ""), s.get("elements", [])
    if not title:
        return None
    tables = [e for e in els if e["type"] == "table"]
    if lay == "table" and tables:
        t = tables[0]
        cols = [str(c) for c in t.get("columns", [])][:5]
        rows = [[str(c) for c in r][: len(cols)] for r in t.get("rows", [])][:7]
        if len(cols) >= 2 and rows and all(len(r) == len(cols) for r in rows):
            return {"layout": "table", "title": title, "header": cols, "rows": rows}
        return None
    if lay in ("two_column", "comparison"):
        left, right = _lines(els, {"left"}), _lines(els, {"right"})
        if len(left) >= 2 and len(right) >= 2 and len(left[0].split()) <= 5 and len(right[0].split()) <= 5:
            lb, rb = _short(left[1:], 12)[:5], _short(right[1:], 12)[:5]
            if lb and rb:
                return {"layout": "two_columns", "title": title, "left": {"heading": left[0], "bullets": lb},
                        "right": {"heading": right[0], "bullets": rb}}
    if lay == "closing":
        b = _short(_lines(els), 14)[:3]
        return {"layout": "closing", "title": title, **({"bullets": b} if b else {})}
    b = _short(_lines(els), 16)[:6]
    if len(b) >= 2:
        return {"layout": "bullets", "title": title, "bullets": b}
    return None


def convert_deck(ir: dict, lang: str, subtitle: str = "") -> dict | None:
    slides = [c for c in (convert_slide(s) for s in ir.get("slides", [])) if c]
    if len(slides) < 3:
        return None
    first = slides[0]["title"]
    deck = {
        "version": "bird-deck/0.1",
        "meta": {"title": " ".join(first.split()[:12]), "lang": lang},
        "theme": {"preset": "bird"},
        "slides": [{"layout": "title", "title": first, **({"subtitle": subtitle} if subtitle else {})}] + slides,
    }
    for _ in range(len(deck["slides"])):  # drop invalid slides until the deck validates
        errs = [i for i in validate_deck(deck) if i.level == "error"]
        if not errs:
            return deck if len(deck["slides"]) >= 4 else None
        m = re.match(r"slides\[(\d+)\]", errs[0].path)
        if not m or int(m.group(1)) == 0:
            return None
        deck["slides"].pop(int(m.group(1)))
    return None


def officesmith_records(path) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        lang = r["language"] if r["language"] in ("fr", "en") else None
        if not lang:
            continue
        m = re.search(r"\{.*\}", r["messages"][-1]["content"], re.S)
        try:
            ir = json.loads(m.group(0))
        except Exception:
            continue
        brief = re.sub(r"</?PLAN>", "", r["messages"][0]["content"]).strip()
        aud = ir.get("audience") or ""
        sub = (f"Présentation pour : {aud}" if lang == "fr" else f"Presentation for: {aud}") if aud and len(aud.split()) <= 8 else ""
        deck = convert_deck(ir, lang, sub)
        if not deck:
            continue
        ask = "\n\nRéponds uniquement avec le deck au format BIRD-Deck (JSON)." if lang == "fr" else "\n\nAnswer only with the deck in BIRD-Deck format (JSON)."
        out.append({"id": f"officesmith-{r['id']}", "task_id": "slides.deck", "lang": lang, "source": "hf:Benitoow/OfficeSmith-PPTX-IR",
                    "license": "apache-2.0", "messages": [{"role": "user", "content": brief + ask}],
                    "target": json.dumps(deck, ensure_ascii=False)})
    return out
