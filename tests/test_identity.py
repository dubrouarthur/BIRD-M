from bird_medium import identity as I


def test_nom_et_francais_partout():
    recs = I.generate(300, 100, seed=0)
    assert len(recs) == 400
    for r in recs:
        a = r["messages"][-1]["content"]
        assert a.strip()
        if r["lang"] == "fr":
            assert "Bird-Medium" in a or "Je " in a or "Oui" in a or "Non" in a


def test_jamais_pretendre_etre_un_autre_modele():
    for r in I.generate(500, 200, seed=1):
        q, a = r["messages"][-2]["content"].lower(), r["messages"][-1]["content"]
        for other in ("chatgpt", "claude", "mistral", "gemini", "llama"):
            if other in q:
                assert a.startswith("No"), (q, a)  # « Non, je ne suis pas… » / « No, I'm not… »


def test_qui_es_tu_donne_le_nom():
    recs = [r for r in I.generate(400, 0, seed=2) if r["messages"][-2]["content"].lower().startswith("qui es-tu")]
    assert recs and all("Bird-Medium" in r["messages"][-1]["content"] for r in recs)


def test_part_de_raisonnement():
    recs = I.generate(400, 100, seed=0)
    share = sum(r["thinking"] for r in recs) / len(recs)
    assert 0.4 < share < 0.6
    assert all(("reasoning_content" in r["messages"][-1]) == r["thinking"] for r in recs)
