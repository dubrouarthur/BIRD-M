from bird_medium import identity, mix


def _r(src, q, a, i=0):
    return {"id": f"{src}-{i}", "source": src, "lang": "fr", "thinking": False,
            "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}


def test_dedupe():
    recs = identity.generate(700, 200, seed=0)
    d = mix.dedupe(recs)
    assert len(d) < len(recs) and len({(r["messages"][-2]["content"], r["messages"][-1]["content"]) for r in d}) == len(d)


def test_split_sans_fuite_hors_identite():
    recs = [_r("emails", f"q{i % 50}", f"a{i}", i) for i in range(200)]
    train, ev = mix.split(recs, 0.1, 0)
    assert ev and not ({r["messages"][0]["content"] for r in ev} & {r["messages"][0]["content"] for r in train})


def test_split_garde_l_identite_en_train():
    recs = mix.dedupe(identity.generate(300, 0, seed=0))
    train, ev = mix.split(recs, 0.05, 0)
    assert any("qui es-tu" in r["messages"][-2]["content"].lower() for r in train)


def test_distill_ok():
    ok = {"task": "summary", "messages": [{"role": "user", "content": "Résume ce fil"},
                                          {"role": "assistant", "content": "x", "reasoning_content": "fine"}]}
    assert mix.distill_ok(ok)
    bad = {**ok, "messages": [ok["messages"][0], {"role": "assistant", "content": "x", "reasoning_content": "as Qwen we"}]}
    assert not mix.distill_ok(bad)
    residue = {**ok, "messages": [{"role": "user", "content": "Résume\n\nRédige la réponse de Ruth."}, ok["messages"][1]]}
    assert not mix.distill_ok(residue)
