import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_taxonomy_valid_and_docs_fresh():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "render_taxonomy.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
