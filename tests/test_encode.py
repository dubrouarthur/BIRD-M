"""Rendu avec le vrai chat template de Qwen3.8-27B (tokenizer seul, ~10 Mo)."""

import importlib.util
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
spec = importlib.util.spec_from_file_location("train_job", Path(__file__).parents[1] / "scripts/train_job.py")
tj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tj)


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
    except Exception as e:  # pas de réseau
        pytest.skip(f"tokenizer indisponible: {e}")


def rec(thinking, system=False):
    m = [{"role": "user", "content": "Qui es-tu ?"},
         {"role": "assistant", "content": "Je suis Bird-Medium.", "reasoning_content": "On me demande mon identité."}]
    if system:
        m.insert(0, {"role": "system", "content": "Tu es Bird-Medium."})
    return {"messages": m, "thinking": thinking}


def target(tok, e):
    return tok.decode([i for i, l in zip(e["input_ids"], e["labels"]) if l != tj.IGNORE])


def test_mode_raisonnement(tok):
    e = tj.encode(tok, rec(True), 4096)
    full = tok.decode(e["input_ids"])
    assert full.rstrip().endswith("<|im_assistant|>") is False
    assert "<think>\n" in full and full.endswith("<|im_end|>")
    assert target(tok, e) == "On me demande mon identité.\n</think>\n\nJe suis Bird-Medium.<|im_end|>"
    assert "Reasoning effort is set to xhigh" in full  # même préfixe qu'à l'inférence par défaut


def test_mode_sans_raisonnement(tok):
    e = tj.encode(tok, rec(False), 4096)
    full = tok.decode(e["input_ids"])
    assert "<think>\n\n</think>\n\n" in full and "Reasoning effort" not in full
    assert target(tok, e) == "Je suis Bird-Medium.<|im_end|>"  # la trace est ignorée hors mode raisonnement


def test_prompt_systeme_et_multitour(tok):
    r = rec(True, system=True)
    r["messages"] = r["messages"][:1] + [{"role": "user", "content": "Bonjour"}, {"role": "assistant", "content": "Bonjour !"}] + r["messages"][1:]
    e = tj.encode(tok, r, 4096)
    assert target(tok, e).endswith("Je suis Bird-Medium.<|im_end|>")
    assert "Bonjour !" not in target(tok, e)  # seul le dernier tour est appris


def test_trop_long(tok):
    assert tj.encode(tok, rec(True), 10) is None
