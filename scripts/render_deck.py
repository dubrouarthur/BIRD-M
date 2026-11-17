"""Validate a BIRD-Deck JSON and write its HTML.

Usage: python scripts/render_deck.py deck.json out.html
Then:  node scripts/deck_check.mjs out.html out_dir   (PNG per slide + overflow report)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bird_m.deck import render_deck, validate_deck  # noqa: E402

deck = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
issues = validate_deck(deck)
for i in issues:
    print(i, file=sys.stderr)
if any(i.level == "error" for i in issues):
    sys.exit(1)
Path(sys.argv[2]).write_text(render_deck(deck), encoding="utf-8")
print(f"ok: {len(deck['slides'])} slides -> {sys.argv[2]}")
