"""Token specification for the BIRD-M Skill Protocol (BSP).

Qwen3.8-27B pads its embedding / LM-head matrices to 248_320 rows while the
tokenizer only uses ids 0..248_076. New special tokens therefore land in
already-allocated rows: no resize, no architecture change, and the checkpoint
stays loadable by vLLM / SGLang / llama.cpp unchanged.
"""

from __future__ import annotations

FIRST_FREE_ID = 248_077  # first id after the base tokenizer's last added token
PADDED_VOCAB = 248_320  # rows in embed_tokens / lm_head (config.text_config.vocab_size)
NUM_SLOTS = 200

CATALOG_START = "<|skill_catalog_start|>"
CATALOG_END = "<|skill_catalog_end|>"
SKILL_START = "<|skill_start|>"
SKILL_END = "<|skill_end|>"
INVOKE = "<|skill_invoke|>"
NONE = "<|skill_none|>"
BODY_START = "<|skill_body_start|>"
BODY_END = "<|skill_body_end|>"
POLICY_START = "<|policy_start|>"
POLICY_END = "<|policy_end|>"
MEM_PAD = "<|skill_mem_pad|>"  # placeholder for compiled-skill embeddings (like <|image_pad|>)

CONTROL_TOKENS: tuple[str, ...] = (
    CATALOG_START,
    CATALOG_END,
    SKILL_START,
    SKILL_END,
    INVOKE,
    NONE,
    BODY_START,
    BODY_END,
    POLICY_START,
    POLICY_END,
    MEM_PAD,
)


def slot_token(i: int) -> str:
    if not 0 <= i < NUM_SLOTS:
        raise ValueError(f"slot index out of range: {i}")
    return f"<|skill_{i:03d}|>"


SLOT_TOKENS: tuple[str, ...] = tuple(slot_token(i) for i in range(NUM_SLOTS))
ALL_TOKENS: tuple[str, ...] = CONTROL_TOKENS + SLOT_TOKENS

# Deterministic id allocation, in the order tokens are appended to the tokenizer.
TOKEN_IDS: dict[str, int] = {tok: FIRST_FREE_ID + i for i, tok in enumerate(ALL_TOKENS)}
SLOT_IDS: tuple[int, ...] = tuple(TOKEN_IDS[t] for t in SLOT_TOKENS)

assert FIRST_FREE_ID + len(ALL_TOKENS) <= PADDED_VOCAB, (
    "BSP tokens overflow padded vocab"
)
