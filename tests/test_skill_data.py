import random
from collections import Counter

import pytest

from bird_m.data.skill_tasks import episode, generate, instantiate, library
from bird_m.skills import (
    NON_THINKING_PREFILL,
    RoutingSession,
    bind_slots,
    parse_skill,
    render_conversation,
)
from bird_m.skills import spec

BENCH = {a["key"] for a in library()["archetypes"] if a["split"] == "bench"}
POLICIES = {p["key"] for p in library()["policies"]}


def test_every_archetype_instantiates_to_a_valid_skill():
    for a in library()["archetypes"] + library()["policies"]:
        for seed in range(5):
            sk, md = instantiate(a["key"], random.Random(seed))
            assert sk.name == a["key"] and parse_skill(md) == sk
            assert "{" not in sk.body  # every template parameter was filled


def test_families_provide_near_miss_distractors():
    fams = Counter(
        a["family"] for a in library()["archetypes"] if a["split"] == "train"
    )
    assert sum(1 for n in fams.values() if n >= 2) >= 6


@pytest.fixture(scope="module")
def train():
    return generate(1500, seed=0)


def test_kind_mix_and_routing_ground_truth(train):
    kinds = Counter(e["kind"] for e in train)
    assert 0.40 < kinds["single"] / len(train) < 0.60
    assert 0.18 < kinds["none"] / len(train) < 0.32
    for e in train:
        inv = e["messages"][1]["invoke"]
        assert all(k in e["catalog"] for k in inv)
        assert not set(inv) & POLICIES  # policies are never invoked
        if e["kind"] in ("none", "policy"):
            assert inv == []
        if e["kind"] == "policy":
            assert e["facts"]["policy"] in e["catalog"]
        if "absent_skill" in e["facts"]:
            assert e["facts"]["absent_skill"] not in e["catalog"]


def test_bench_skills_never_leak_into_training(train):
    for e in train:
        assert not set(e["catalog"]) & BENCH


def test_bench_targets_are_unseen_skills():
    bench = generate(400, seed=1, split="bench")
    targets = {k for e in bench for k in e["messages"][1]["invoke"]} - {
        "charte-redactionnelle"
    }
    assert targets and targets <= BENCH


def test_episodes_render_through_the_protocol(train):
    """Every episode must be renderable by BSP and replayable by the runtime state machine."""
    rng = random.Random(0)
    for e in train[:300]:
        skills = {k: parse_skill(md) for k, md in e["skills"].items()}
        msgs = [e["messages"][0], {**e["messages"][1], "content": "réponse"}]
        segs = render_conversation(e["system"], skills, e["catalog"], msgs, rng=rng)
        trained = "".join(s.text for s in segs if s.train)
        assert trained.count(spec.INVOKE) == len(e["messages"][1]["invoke"])
        assert spec.NONE in trained
        # runtime replay with stable binding
        invocable = [k for k in e["catalog"] if not skills[k].is_policy]
        b = bind_slots(invocable)
        s = RoutingSession(b, skills)
        for name in e["messages"][1]["invoke"]:
            s.feed(spec.TOKEN_IDS[spec.INVOKE])
            s.feed(spec.SLOT_IDS[b[name]])
        assert (
            s.feed(spec.TOKEN_IDS[spec.NONE]) == NON_THINKING_PREFILL
            and s.invoked == e["messages"][1]["invoke"]
        )


def test_deterministic():
    assert generate(50, seed=7) == generate(50, seed=7)
    a = episode(random.Random(3), kind="compose")
    assert a["messages"][1]["invoke"][-1] == "charte-redactionnelle"
