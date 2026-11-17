"""BIRD-M Skill Protocol: slot binding, rendering to training text, runtime routing.

Conversation layout (Qwen ChatML + BSP tokens):

    <|im_start|>system
    {system}
    <|policy_start|>{name}\n{policy body}<|policy_end|>   # always on, highest priority
    <|skill_catalog_start|>
    <|skill_start|><|skill_017|>
    name: ... / type: ... / description: ... / when_to_use: ...
    <|skill_end|>
    ...
    <|skill_catalog_end|><|im_end|>
    <|im_start|>user
    ...<|im_end|>
    <|im_start|>assistant
    <|skill_invoke|><|skill_017|>                         # generated
    <|skill_body_start|>...body...<|skill_body_end|>      # injected by runtime, never trained
    <|skill_none|>                                        # generated: end of routing
    <think>...</think>                                    # model (thinking) or runtime prefill
    answer<|im_end|>

Routing grammar (when a catalog is present):  ( INVOKE SLOT BODY ){0,3} NONE
Slots are *pointers* bound per request: during training the binding is random,
so a slot token carries no fixed meaning and the model must read the catalog.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field

from . import spec
from .skill import Skill

MAX_INVOCATIONS = 3
NON_THINKING_PREFILL = "<think>\n\n</think>\n\n"
_RESERVED_RE = re.compile(spec.RESERVED_PATTERN)
_CONTROL_RE = re.compile(r"<\|[A-Za-z_0-9]+\|>")


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
def sanitize(text: str) -> str:
    """Neutralise control-token look-alikes (<|...|>, <think>, <tool_call>, ...).

    The base tokenizer has split_special_tokens=False, so a literal
    '<|skill_invoke|>' or '</think>' inside an email or a skill body would
    otherwise become the real token. A zero-width space after '<' breaks the
    match while staying readable. Applied to every string the runtime renders.
    """
    return _RESERVED_RE.sub(lambda m: "<​" + m.group(0)[1:], text)


def _one_line(text: str) -> str:
    return sanitize(" ".join(text.split()))


def logit_bias(binding: dict[str, int]) -> dict[int, float]:
    """Ban unbound slots and runtime-only tokens (no hallucinated skill, no forged body).

    -100 is the OpenAI-API floor; for a hard guarantee use RoutingSession.allowed_next()
    with constrained decoding (vLLM `allowed_token_ids`) during the routing phase.
    """
    bound = set(binding.values())
    bias = {spec.SLOT_IDS[i]: -100.0 for i in range(spec.NUM_SLOTS) if i not in bound}
    bias.update({spec.TOKEN_IDS[t]: -100.0 for t in spec.RUNTIME_ONLY_TOKENS})
    return bias


# ----------------------------------------------------------------- rendering
@dataclass(frozen=True)
class Segment:
    text: str
    train: bool  # True => contributes to the SFT loss


def render_card(skill: Skill, slot: int) -> str:
    lines = [
        f"name: {_one_line(skill.name)}",
        f"type: {_one_line(skill.type)}",
        f"description: {_one_line(skill.description)}",
    ]
    if skill.when_to_use:
        lines.append(f"when_to_use: {_one_line(skill.when_to_use)}")
    return (
        f"{spec.SKILL_START}{spec.slot_token(slot)}\n"
        + "\n".join(lines)
        + f"\n{spec.SKILL_END}\n"
    )


def render_body(skill: Skill) -> str:
    return f"{spec.BODY_START}\n{sanitize(skill.body)}\n{spec.BODY_END}\n"


def render_system(
    system: str, policies: list[Skill], catalog: list[Skill], binding: dict[str, int]
) -> str:
    out = f"<|im_start|>system\n{sanitize(system.strip())}\n"
    for p in sorted(policies, key=lambda s: -s.priority):
        out += f"{spec.POLICY_START}{_one_line(p.name)}\n{sanitize(p.body)}\n{spec.POLICY_END}\n"
    if catalog:
        out += f"{spec.CATALOG_START}\n"
        out += "".join(render_card(s, binding[s.name]) for s in catalog)
        out += f"{spec.CATALOG_END}"
    return out.rstrip("\n") + "<|im_end|>\n"


def _check_target(text: str, what: str, pattern: re.Pattern) -> str:
    """Trained text must not contain control tokens: reject the example, don't silently fix it."""
    if pattern.search(text):
        raise ProtocolError(f"{what} contains reserved tokens")
    return text.strip()


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
    The assistant layout is byte-identical to what RoutingSession produces at inference.
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
        invoked = list(msg.get("invoke") or [])
        if len(invoked) > MAX_INVOCATIONS:
            raise ProtocolError(f"too many invocations: {len(invoked)}")
        if len(set(invoked)) != len(invoked):
            raise ProtocolError("same skill invoked twice")
        if invocable:
            for name in invoked:
                if name not in binding:
                    raise ProtocolError(
                        f"invoked skill not in invocable catalog: {name}"
                    )
                segs.append(Segment(spec.INVOKE + spec.slot_token(binding[name]), True))
                segs.append(Segment(render_body(skills[name]), False))
            segs.append(Segment(spec.NONE, True))  # end of routing, always supervised
        elif invoked:
            raise ProtocolError("invocation without catalog")
        thinking = msg.get("thinking")
        if thinking:
            thinking = _check_target(thinking, "thinking", _RESERVED_RE)
            segs.append(Segment(f"<think>\n{thinking}\n</think>\n\n", True))
        else:
            segs.append(Segment(NON_THINKING_PREFILL, False))
        content = _check_target(msg["content"], "assistant content", _CONTROL_RE)
        segs.append(Segment(content + "<|im_end|>", True))
        segs.append(Segment("\n", False))
    return segs


# ----------------------------------------------------------------- runtime
@dataclass
class RoutingSession:
    """Token-level state machine for the routing phase of one assistant turn.

    Works on token ids (not decoded text, which servers strip of special tokens).
    Loop: ids = allowed_next(); generate ONE token constrained to ids;
          text = feed(token); append text to the prefix; repeat until done.
    Then generate the answer freely with logit_bias(binding).
    """

    binding: dict[str, int]
    skills: dict[str, Skill]
    enable_thinking: bool = False
    max_invocations: int = MAX_INVOCATIONS
    invoked: list[str] = field(default_factory=list)  # audit log, in order
    _state: str = "decide"  # decide -> (slot -> decide)* -> done

    @property
    def done(self) -> bool:
        return self._state == "done"

    def allowed_next(self) -> list[int]:
        if self._state == "decide":
            ids = [spec.TOKEN_IDS[spec.NONE]]
            if len(self.invoked) < self.max_invocations and len(self.invoked) < len(
                self.binding
            ):
                ids.append(spec.TOKEN_IDS[spec.INVOKE])
            return ids
        if self._state == "slot":
            return [
                spec.SLOT_IDS[i]
                for n, i in self.binding.items()
                if n not in self.invoked
            ]
        raise ProtocolError("routing already finished")

    def feed(self, token_id: int) -> str:
        """Consume one generated token; return the text the runtime must append."""
        if token_id not in self.allowed_next():
            raise ProtocolError(f"token {token_id} not allowed in state {self._state}")
        if self._state == "decide":
            if token_id == spec.TOKEN_IDS[spec.NONE]:
                self._state = "done"
                return "" if self.enable_thinking else NON_THINKING_PREFILL
            self._state = "slot"
            return ""
        slot = spec.SLOT_IDS.index(token_id)
        name = next(n for n, i in self.binding.items() if i == slot)
        self.invoked.append(name)
        self._state = "decide"
        return render_body(self.skills[name])
