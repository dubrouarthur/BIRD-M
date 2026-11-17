"""Evaluate a BSP-trained model locally (transformers): skill routing with constrained decoding + finance exact.

Routing: the prompt (system + catalog + user) is built with STABLE binding exactly like production;
RoutingSession.allowed_next() restricts each decision token; the model's argmax among allowed ids is fed back.

Usage:
  python scripts/eval_bsp.py --model Qwen/Qwen3.5-2B --adapter runs/pilot/adapter --n-skills 500 --n-fin 150 --out runs/pilot/eval.json
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bird_m.data.checks import numbers_check  # noqa: E402
from bird_m.data.finance_tasks import generate as gen_fin  # noqa: E402
from bird_m.data.skill_tasks import generate as gen_skills  # noqa: E402
from bird_m.skills import RoutingSession, bind_slots, parse_skill, sanitize  # noqa: E402
from bird_m.skills.protocol import render_system  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", default="")
ap.add_argument("--n-skills", type=int, default=200)
ap.add_argument("--n-fin", type=int, default=100)
ap.add_argument("--max-new", type=int, default=700)
ap.add_argument("--out", required=True)
ap.add_argument("--skip-skills", action="store_true", help="base model without BSP tokens: finance only")
a = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
src = a.adapter or a.model
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype)
if a.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, a.adapter)
model.to(dev).eval()


@torch.no_grad()
def next_logits(text: str) -> torch.Tensor:
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
    return model(input_ids=ids).logits[0, -1].float()


def route(e: dict) -> tuple[list[str], float]:
    skills = {k: parse_skill(md) for k, md in e["skills"].items()}
    policies = [skills[k] for k in e["catalog"] if skills[k].is_policy]
    invocable = [skills[k] for k in e["catalog"] if not skills[k].is_policy]
    binding = bind_slots(
        [s.name for s in invocable]
    )  # stable binding, as in production
    prompt = render_system(e["system"], policies, invocable, binding)
    prompt += f"<|im_start|>user\n{sanitize(e['messages'][0]['content'])}<|im_end|>\n<|im_start|>assistant\n"
    s = RoutingSession(binding, skills)
    first_conf = None
    while not s.done:
        allowed = s.allowed_next()
        logits = next_logits(prompt)
        probs = torch.softmax(logits[allowed], 0)
        k = int(probs.argmax())
        if first_conf is None:
            first_conf = float(probs[k])
        tid = allowed[k]
        prompt += tok.convert_ids_to_tokens(tid) + s.feed(tid)
    return s.invoked, first_conf


@torch.no_grad()
def answer(user: str, lang: str) -> str:
    system = (
        "Tu es BIRD-M, un assistant pour les entreprises."
        if lang == "fr"
        else "You are BIRD-M, an assistant for businesses."
    )
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{sanitize(user)}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        dev
    )
    out = model.generate(
        input_ids=ids,
        max_new_tokens=a.max_new,
        do_sample=False,
        eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
    )
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


report = {"model": a.model, "adapter": a.adapter, "skills": {}, "finance": {}}
t0 = time.time()

# ------------------------------------------------ skill routing (bench = unseen skills; train-split = seen skills, fresh episodes)
for split, seed in (() if a.skip_skills else (("bench", 1), ("train", 99))):
    eps = random.Random(0).sample(
        gen_skills(max(a.n_skills, 50) * 2, seed=seed, split=split), a.n_skills
    )
    by = defaultdict(list)
    for e in eps:
        got, conf = route(e)
        want = e["messages"][1]["invoke"]
        ok = got == want
        by[e["kind"]].append(ok)
        by["ALL"].append(ok)
    report["skills"][split] = {
        k: {"n": len(v), "exact": round(sum(v) / len(v), 3)}
        for k, v in sorted(by.items())
    }
    print(
        json.dumps(
            {
                "skills": split,
                **report["skills"][split]["ALL"],
                "secs": round(time.time() - t0),
            }
        ),
        flush=True,
    )

# ------------------------------------------------ finance exact (held-out companies)
fin = [r for r in gen_fin(1000, seed=1, company_offset=900_000) if r["kind"] == "exact"]
fin = random.Random(0).sample(fin, min(a.n_fin, len(fin)))
by = defaultdict(list)
for r in fin:
    res = numbers_check(answer(r["messages"][0]["content"], r["lang"]), r["facts"])
    by[r["task_id"]].append(res["score"] == 1.0)
    by["ALL"].append(res["score"] == 1.0)
report["finance"] = {
    k: {"n": len(v), "exact": round(sum(v) / len(v), 3)} for k, v in sorted(by.items())
}
print(
    json.dumps({"finance": report["finance"]["ALL"], "secs": round(time.time() - t0)}),
    flush=True,
)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
print("BIRD_EVAL_REPORT " + json.dumps(report), flush=True)
