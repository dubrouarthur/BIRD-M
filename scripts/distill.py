"""Auto-distillation : Qwen3.8-27B (mode raisonnement) répond lui-même à des tâches d'entreprise en français.

But : garder le raisonnement natif du modèle de base pendant le fine-tune, au lieu de lui apprendre
les traces d'un autre modèle. Les prompts viennent des sources e-mails / slides (seed différente du mix)
plus des demandes de synthèse de fils d'e-mails, qui n'ont pas de réponse de référence.

Usage (Inference Providers, ~0,0003 $ / requête) :
  python scripts/distill.py --n 1500 --out data/distill.jsonl
Reprend là où il s'est arrêté (les ids déjà présents dans --out sont sautés).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bird_medium.identity import NAME  # noqa: E402
from bird_medium.sources import emails, lang_of, leaks_identity, slides  # noqa: E402

URL = "https://router.huggingface.co/v1/chat/completions"
SUMMARY = [
    "Résume cet échange d'e-mails en 3 points maximum, puis liste les actions à mener avec leur responsable.",
    "Fais une synthèse courte de ce fil de discussion et indique les décisions prises et les points en suspens.",
    "Je n'ai pas le temps de tout lire : résume-moi ces mails et dis-moi ce que je dois faire.",
]


def build_prompts(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    n_mail, n_slides = round(n * 0.8), n - round(n * 0.8)
    out = []
    for r in emails({"fr": n_mail}, seed):
        user = r["messages"][0]["content"]
        if (
            "reply" in r["id"] and rng.random() < 0.5
        ):  # synthèse du fil : message précédent + réponse
            thread = (
                user.split("\n\n", 1)[-1] + "\n\n---\n\n" + r["messages"][1]["content"]
            )
            user = f"{rng.choice(SUMMARY)}\n\n{thread}"
            out.append(
                {"id": f"distill-summary-{len(out)}", "task": "summary", "user": user}
            )
        else:
            out.append(
                {"id": f"distill-email-{len(out)}", "task": "email", "user": user}
            )
    for r in slides({"fr": n_slides}, seed):
        out.append(
            {
                "id": f"distill-slides-{len(out)}",
                "task": "slides",
                "user": r["messages"][0]["content"],
            }
        )
    rng.shuffle(out)
    return out[:n]


def ask(user: str, model: str, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN', 'hf_proxy')}"}
    for attempt in range(4):
        try:
            r = requests.post(URL, json=body, headers=headers, timeout=300)
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            err = repr(e)
        time.sleep(2**attempt * 3)
    raise RuntimeError(err)


def to_record(p: dict, resp: dict) -> dict | None:
    ch = resp["choices"][0]
    msg = ch["message"]
    reasoning, content = (
        (msg.get("reasoning") or "").strip(),
        (msg.get("content") or "").strip(),
    )
    if ch.get("finish_reason") != "stop" or not reasoning or not content:
        return None
    if lang_of(content) != "fr" or leaks_identity(content):
        return None
    return {
        "id": p["id"],
        "source": "distill",
        "lang": "fr",
        "thinking": True,
        "task": p["task"],
        "messages": [
            {"role": "user", "content": p["user"]},
            {"role": "assistant", "content": content, "reasoning_content": reasoning},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument(
        "--seed", type=int, default=1000
    )  # différente du mix pour éviter les doublons
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--out", default="data/distill.jsonl")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    prompts = [p for p in build_prompts(a.n, a.seed) if p["id"] not in done]
    print(
        f"{len(done)} déjà faits, {len(prompts)} à générer avec {a.model} ({NAME} distill)",
        flush=True,
    )
    kept = dropped = 0
    with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(ask, p["user"], a.model, a.max_tokens): p for p in prompts}
        for i, f in enumerate(as_completed(futs), 1):
            p = futs[f]
            try:
                rec = to_record(p, f.result())
            except Exception as e:  # noqa: BLE001 — on journalise et on continue
                print("erreur", p["id"], e, flush=True)
                rec = None
            if rec:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                kept += 1
            else:
                dropped += 1
            if i % 50 == 0:
                print(
                    json.dumps({"done": i, "kept": kept, "dropped": dropped}),
                    flush=True,
                )
    print(json.dumps({"kept": kept, "dropped": dropped, "out": str(out)}))


if __name__ == "__main__":
    main()
