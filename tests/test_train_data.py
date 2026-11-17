import random

from bird_m.train.data import record_to_segments


def _text(rec):
    return "".join(s.text for s in record_to_segments(rec, random.Random(0)))


def test_record_system_prompt_is_used_and_can_be_omitted():
    base = {"id": "x", "lang": "fr", "messages": [{"role": "user", "content": "Qui es-tu ?"}], "target": "Je suis BIRD-M."}
    assert "Tu es BIRD-M" in _text(base)  # default system
    assert "Tu es Lia" in _text({**base, "system": "Tu es Lia."})  # source system prompt kept
    assert "<|im_start|>system" not in _text({**base, "no_system": True})
