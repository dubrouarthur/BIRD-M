"""Turn BIRD records into token ids + loss labels for SFT.

- setup_tokenizer: aligns any Qwen3.5/3.8 tokenizer on the BSP id layout (ids 248077..248287) by adding
  reserved placeholders first when the base has fewer added tokens (Qwen3.5 lacks 7 audio tokens).
- record_to_segments: finance/skills JSONL records -> BSP loss-masked segments.
  Skills records whose prose answer is not generated yet (content None) are trained on ROUTING ONLY:
  the sequence stops right after <|skill_none|>.
- encode: per-segment tokenisation; labels = -100 on non-trained segments.
"""

from __future__ import annotations

import random

from bird_m.skills import Segment, parse_skill, render_conversation, spec

IGNORE = -100


def setup_tokenizer(tok) -> None:
    """Make len(tok) == FIRST_FREE_ID, then add BSP tokens so their ids match spec.TOKEN_IDS exactly."""
    missing = spec.FIRST_FREE_ID - len(tok)
    if missing < 0:
        raise ValueError(
            f"tokenizer already has {len(tok)} tokens > {spec.FIRST_FREE_ID}"
        )
    if missing:
        tok.add_special_tokens(
            {
                "additional_special_tokens": [
                    f"<|bird_reserved_{i}|>" for i in range(missing)
                ]
            },
            replace_extra_special_tokens=False,
        )
    tok.add_special_tokens(
        {"additional_special_tokens": list(spec.ALL_TOKENS)},
        replace_extra_special_tokens=False,
    )
    for t, i in spec.TOKEN_IDS.items():
        got = tok.convert_tokens_to_ids(t)
        if got != i:
            raise ValueError(f"{t}: id {got} != expected {i}")


def record_to_segments(rec: dict, rng: random.Random) -> list[Segment]:
    if "catalog" in rec:  # BSP episode
        skills = {k: parse_skill(md) for k, md in rec["skills"].items()}
        asst = rec["messages"][1]
        routing_only = asst.get("content") is None
        msgs = [rec["messages"][0], {**asst, "content": asst.get("content") or "x"}]
        segs = render_conversation(rec["system"], skills, rec["catalog"], msgs, rng=rng)
        if routing_only:
            cut = max(i for i, s in enumerate(segs) if s.text == spec.NONE)
            segs = segs[: cut + 1]
        return segs
    # plain task record (finance exact, ...): no catalog, the target is the assistant answer
    if rec.get("target") is None:
        raise ValueError(f"record {rec.get('id')} has no target yet (teacher pending)")
    default = (
        "Tu es BIRD-M, un assistant pour les entreprises."
        if rec["lang"] == "fr"
        else "You are BIRD-M, an assistant for businesses."
    )
    if rec.get("no_system"):
        system = ""  # trained WITHOUT any system prompt: identity and behaviour must not depend on one
    else:
        system = rec.get("system") or default
    msgs = rec["messages"] + [{"role": "assistant", "content": rec["target"]}]
    return render_conversation(system, {}, [], msgs)


def encode(tok, segments: list[Segment], max_len: int) -> dict | None:
    ids, labels = [], []
    for s in segments:
        t = tok(s.text, add_special_tokens=False)["input_ids"]
        ids += t
        labels += t if s.train else [IGNORE] * len(t)
    if len(ids) > max_len or not any(l != IGNORE for l in labels):
        return None  # too long or nothing to learn: skipped and counted by the caller
    return {"input_ids": ids, "labels": labels}


def is_trainable(rec: dict) -> bool:
    return "catalog" in rec or rec.get("target") is not None
