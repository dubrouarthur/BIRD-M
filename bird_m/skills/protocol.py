"""BIRD-M Skill Protocol: slot binding, rendering to training text, runtime helpers.

Conversation layout (Qwen ChatML + BSP tokens):

    <|im_start|>system
    {system}
    <|policy_start|>{policy body}<|policy_end|>          # always on, highest priority
    <|skill_catalog_start|>
    <|skill_start|><|skill_017|>
    name: ... / type: ... / description: ... / when_to_use: ...
    <|skill_end|>
    ...
    <|skill_catalog_end|><|im_end|>
    <|im_start|>user
    ...<|im_end|>
    <|im_start|>assistant
    <|skill_invoke|><|skill_017|>                         # generated: 1-token routing decision
    <|skill_body_start|>...body...<|skill_body_end|>      # injected by runtime, never trained
    <think>...</think>                                    # optional reasoning
    answer<|im_end|>

When no skill applies, the model emits <|skill_none|> instead of invoking.
Slots are *pointers* bound per request: during training the binding is random,
so a slot token carries no fixed meaning and the model must read the catalog.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from . import spec
from .skill import Skill

MAX_INVOCATIONS = 3
_RESERVED_RE = re.compile(r"<\|([a-z_0-9]+)\|>")


class ProtocolError(ValueError):
    pass


# ----------------------------------------------------------------- binding
def bind_slots(names: list[str], rng: random.Random | None = None) -> dict[str, int]:
    """Map skill names to distinct slot indices.

    rng given  -> random binding (training: forces in-context pointer reading).
    rng None   -> stable hash binding (serving: same catalog => same prefix => KV cache hits).
    """
    if len(set(names)) != len(names):
        raise ProtocolError("duplicate skill names in catalog")
    if len(names) > spec.NUM_SLOTS:
        raise ProtocolError(f"catalog too large: {len(names)} > {spec.NUM_SLOTS} slots")
    if rng is not None:
        return dict(zip(names, rng.sample(range(spec.NUM_SLOTS), len(names))))
    taken: set[int] = set()
    out: dict[str, int] = {}
    for name in sorted(names):
        i = int(hashlib.sha256(name.encode()).hexdigest(), 16) % spec.NUM_SLOTS
        while i in taken:
            i = (i + 1) % spec.NUM_SLOTS
        taken.add(i)
        out[name] = i
    return out


# ----------------------------------------------------------------- safety
def sanitize(untrusted: str) -> str:
    """Neutralise control-token look-alikes in user / document text.

    The base tokenizer has split_special_tokens=False, so a literal
    '<|skill_invoke|>' inside an email would otherwise become the real token.
    A zero-width space after '<' breaks the match while staying readable.
    """
    return _RESERVED_RE.sub(lambda m: "<​|" + m.group(1) + "|>", untrusted)


def logit_bias(binding: dict[str, int]) -> dict[int, float]:
    """Ban every slot not bound in this request (no hallucinated skills, RBAC-safe)."""
    bound = set(binding.values())
    return {spec.SLOT_IDS[i]: -100.0 for i in range(spec.NUM_SLOTS) if i not in bound}


# ----------------------------------------------------------------- rendering
@dataclass(frozen=True)
class Segment:
    text: str
    train: bool  # True => contributes to the SFT loss


def render_card(skill: Skill, slot: int) -> str:
    lines = [
        f"name: {skill.name}",
        f"type: {skill.type}",
        f"description: {skill.description}",
    ]
    if skill.when_to_use:
        lines.append(f"when_to_use: {skill.when_to_use}")
    return (
        f"{spec.SKILL_START}{spec.slot_token(slot)}\n"
        + "\n".join(lines)
        + f"\n{spec.SKILL_END}\n"
    )


def render_body(skill: Skill) -> str:
    return f"{spec.BODY_START}\n{skill.body}\n{spec.BODY_END}\n"


def render_system(
    system: str, policies: list[Skill], catalog: list[Skill], binding: dict[str, int]
) -> str:
    out = f"<|im_start|>system\n{system.strip()}\n"
    for p in sorted(policies, key=lambda s: -s.priority):
        out += f"{spec.POLICY_START}{p.name}\n{p.body}\n{spec.POLICY_END}\n"
    if catalog:
        out += f"{spec.CATALOG_START}\n"
        out += "".join(render_card(s, binding[s.name]) for s in catalog)
        out += f"{spec.CATALOG_END}"
    return out.rstrip("\n") + "<|im_end|>\n"


def render_conversation(
    system: str,
    skills: dict[str, Skill],
    catalog: list[str],
    messages: list[dict],
    rng: random.Random | None = None,
) -> list[Segment]:
    """Render one training example into loss-masked segments.

    messages: {"role": "user", "content": str}
              {"role": "assistant", "content": str, "invoke": [names], "thinking": str|None}
    Policy skills in `catalog` are always injected and are not invocable.
    """
    unknown = [n for n in catalog if n not in skills]
    if unknown:
        raise ProtocolError(f"unknown skills in catalog: {unknown}")
    policies = [skills[n] for n in catalog if skills[n].is_policy]
    invocable = [skills[n] for n in catalog if not skills[n].is_policy]
    if rng is not None:
        invocable = rng.sample(
            invocable, len(invocable)
        )  # order must not leak the answer
    binding = bind_slots([s.name for s in invocable], rng)

    segs = [Segment(render_system(system, policies, invocable, binding), False)]
    for msg in messages:
        role = msg["role"]
        if role == "user":
            segs.append(
                Segment(
                    f"<|im_start|>user\n{sanitize(msg['content'])}<|im_end|>\n", False
                )
            )
            continue
        if role != "assistant":
            raise ProtocolError(f"unsupported role: {role}")
        segs.append(Segment("<|im_start|>assistant\n", False))
        invoked = msg.get("invoke") or []
        if len(invoked) > MAX_INVOCATIONS:
            raise ProtocolError(f"too many invocations: {len(invoked)}")
        if invocable:
            if not invoked:
                segs.append(Segment(spec.NONE, True))
            for name in invoked:
                if name not in binding:
                    raise ProtocolError(
                        f"invoked skill not in invocable catalog: {name}"
                    )
                segs.append(Segment(spec.INVOKE + spec.slot_token(binding[name]), True))
                segs.append(Segment(render_body(skills[name]), False))
        elif invoked:
            raise ProtocolError("invocation without catalog")
        thinking = msg.get("thinking")
        if thinking:
            segs.append(Segment(f"<think>\n{thinking.strip()}\n</think>\n\n", True))
        else:
            segs.append(
                Segment("<think>\n\n</think>\n\n", False)
            )  # non-thinking prefill
        segs.append(Segment(msg["content"].strip() + "<|im_end|>\n", True))
    return segs


# ----------------------------------------------------------------- runtime
_INVOKE_TAIL_RE = re.compile(re.escape(spec.INVOKE) + r"<\|skill_(\d{3})\|>\Z")


def pending_injection(
    generated: str, binding: dict[str, int], skills: dict[str, Skill]
) -> str | None:
    """If generation just stopped on an invocation, return the body to append, else None.

    The server stops on any slot token (stop_token_ids=SLOT_IDS), calls this,
    appends the result to the assistant prefix and resumes generation.
    """
    m = _INVOKE_TAIL_RE.search(generated)
    if not m:
        return None
    if generated.count(spec.INVOKE) > MAX_INVOCATIONS:
        raise ProtocolError("invocation limit exceeded")
    slot = int(m.group(1))
    by_slot = {v: k for k, v in binding.items()}
    if slot not in by_slot:
        raise ProtocolError(f"model invoked unbound slot {slot}")
    return render_body(skills[by_slot[slot]])
