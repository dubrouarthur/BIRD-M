import json

import pytest

from bird_m.data.finance_tasks import generate as gen_fin
from bird_m.data.skill_tasks import generate as gen_skills
from bird_m.eval.baseline import aggregate, parse_route, score_finance, score_skills, skills_messages
from bird_m.llm import ChatClient, LLMError


def fake_transport(answers):
    calls = []

    def send(url, payload, headers, timeout):
        calls.append(payload)
        a = answers[len(calls) - 1] if isinstance(answers, list) else answers
        if isinstance(a, Exception):
            raise a
        return {"choices": [{"message": {"content": a}}], "usage": {"completion_tokens": 7}}

    return send, calls


def test_client_payload_cache_and_thinking_flag(tmp_path):
    send, calls = fake_transport("ok")
    c = ChatClient(model="m", thinking=False, transport=send, cache_path=tmp_path / "c.jsonl")
    assert c.chat([{"role": "user", "content": "x"}])["content"] == "ok"
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    c.chat([{"role": "user", "content": "x"}])
    assert len(calls) == 1  # cached in memory
    c2 = ChatClient(model="m", thinking=False, transport=send, cache_path=tmp_path / "c.jsonl")
    c2.chat([{"role": "user", "content": "x"}])
    assert len(calls) == 1  # cached on disk: resumed runs never pay twice


def test_client_does_not_retry_missing_credits():
    send, calls = fake_transport([LLMError("Purchase pre-paid credits to continue")] * 5)
    c = ChatClient(model="m", transport=send, retries=4)
    with pytest.raises(LLMError):
        c.chat([{"role": "user", "content": "x"}])
    assert len(calls) == 1
    out = c.map([[{"role": "user", "content": "y"}]], workers=1)
    assert isinstance(out[0], LLMError)


def test_finance_scoring_rewards_exact_target_only():
    r = next(x for x in gen_fin(50, seed=1, company_offset=900_000) if x["kind"] == "exact")
    assert score_finance(r, r["target"])["exact"]
    assert not score_finance(r, "Je ne sais pas.")["exact"]


def test_route_parsing_and_scoring():
    cat = ["relance-impayes", "relance-devis", "charte-redactionnelle"]
    assert parse_route("relance-impayes", cat) == {"relance-impayes"}
    assert parse_route("Skills : relance-impayes, charte-redactionnelle.", cat) == {"relance-impayes", "charte-redactionnelle"}
    assert parse_route("AUCUNE", cat) == set() and parse_route("none applies", cat) == set()
    e = next(x for x in gen_skills(80, seed=1, split="bench") if x["kind"] == "single")
    target = e["messages"][1]["invoke"][0]
    assert score_skills(e, target)["exact"] and not score_skills(e, "AUCUNE")["exact"]
    msgs = skills_messages(e)
    assert target in msgs[0]["content"] and "<|" not in msgs[0]["content"]  # plain-text catalog, no BSP tokens


def test_aggregate():
    rows = [{"kind": "a", "exact": True, "completion_tokens": 10}, {"kind": "a", "exact": False, "completion_tokens": 30}]
    agg = aggregate(rows, "kind")
    assert agg["a"]["exact_rate"] == 0.5 and agg["ALL"]["avg_completion_tokens"] == 20
