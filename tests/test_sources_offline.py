from bird_medium import sources as S


def test_lang_of():
    assert S.lang_of("Bonjour, je vous écris pour le rapport que nous avons reçu dans la semaine.") == "fr"
    assert S.lang_of("Hello, this is the report that you have asked for with the numbers.") == "en"


def test_split_think():
    assert S.split_think("<think>\nr\n</think>\n\na") == ("r", "a")
    assert S.split_think("a") == ("", "a")


def test_clean_email():
    body = ("Bonjour Ruth,\n\nMerci pour ton retour, je valide le planning proposé pour jeudi.\n\nCordialement,\n\nMilan\n\n"
            "-- \nUnger, Milan\nAnalyste, Sinabis Oil & Gas GmbH\n\nMode vert : n'imprimez ce courriel que si nécessaire.")
    out = S.clean_email(body)
    assert out.endswith("Milan") and "Sinabis" not in out and "Mode vert" not in out


def test_sign_et_double_politesse():
    assert S.sign("Merci.\n\nBien cordialement", "Walter Seidel").endswith("Walter")
    assert S.clean_email("Texte assez long pour passer le seuil de quatre-vingts caractères, voilà.\n\nCordialement,\nZoe\n\nBien cordialement") \
        .endswith("Zoe")


def test_deck_to_markdown():
    text = "[Slide 1]\nRésultats T3\nSous-titre\nSinabis Lyon · 2024\n\n[Slide 2]\nOrdre du jour\nDiapositive 2 / 5\n01   Ventes\n2\n✓ Marges"
    md = S.deck_to_markdown(text, "fr", "Résultats T3")
    assert md.startswith("## Diapositive 1 — Résultats T3\n- Sous-titre")
    assert "Diapositive 2 / 5" not in md and "- 2\n" not in md and "- Marges" in md


def test_leak():
    assert S.leaks_identity("Je suis Qwen, créé par Alibaba")
    assert not S.leaks_identity("Je suis Bird-Medium")


def test_autres_langues_romanes_rejetees():
    for t in ["¿Cuál es la forma más clara de explicar esto a los clientes de la empresa?",
              "La tua risposta deve contenere almeno tre punti e non più di cinque, per questo è importante.",
              "Implementa una pila con las operaciones para el sistema y como es muy simple de usar."]:
        assert S.lang_of(t) == "other", t


def test_fuite_dans_le_raisonnement():
    assert S.leaks_identity("But as Qwen we should produce answer.") and S.leaks_identity("GPT-4o")


def test_pas_de_double_signature():
    b = "Salut, je passe te voir demain pour le dossier des pompes, on en parle à ce moment-là.\n\nà tout à l'heure,\nmarek\n\nCordialement"
    out = S.sign(S.clean_email(b), "Marek Wolf")
    assert out.endswith("marek") and "Cordialement" not in out
