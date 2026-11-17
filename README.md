# BIRD-M

Modèle de langage pour les entreprises françaises (FR/EN), dérivé de [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0), avec le **BIRD-M Skill Protocol** : des compétences d'entreprise branchables sans ré-entraînement, routées en un token, gouvernables et auditables.

| Document | Contenu |
|---|---|
| [docs/BLUEPRINT.md](docs/BLUEPRINT.md) | Plan global : objectifs, données, entraînement, budget, phases, risques |
| [docs/TASKS.md](docs/TASKS.md) | Les 81 tâches d'entreprise ciblées (générée depuis `data/taxonomy/tasks.yaml`) |
| [docs/SKILL_PROTOCOL.md](docs/SKILL_PROTOCOL.md) | Spécification de l'innovation BSP |
| [docs/SKILLS_STRATEGY.md](docs/SKILLS_STRATEGY.md) | Pourquoi et comment les skills rendent le modèle plus performant |
| [docs/DATASETS.md](docs/DATASETS.md) | Les datasets à créer (BIRD-World, 11 configs, BIRD-Bench) |
| [docs/INNOVATIONS.md](docs/INNOVATIONS.md) | Innovations techniques priorisées (coût, vitesse, qualité) |

Hugging Face (privé) : modèle [`adubrou/BIRD-M`](https://huggingface.co/adubrou/BIRD-M) · données [`adubrou/bird-m-sft`](https://huggingface.co/datasets/adubrou/bird-m-sft) · éval [`adubrou/bird-bench`](https://huggingface.co/datasets/adubrou/bird-bench)

## Démarrage

```bash
pip install -e '.[dev]'
pytest -q
# vérification contre le vrai tokenizer Qwen3.8 :
BIRD_M_TOKENIZER=path/to/tokenizer.json pytest -q
python scripts/render_taxonomy.py   # après modification de tasks.yaml
# slides : valider + rendre un deck, puis captures PNG et contrôle de débordement
python scripts/render_deck.py examples/decks/resultats-t3-lumen.json /tmp/deck.html
node scripts/deck_check.mjs /tmp/deck.html /tmp/deck_png
```

## Structure

```
bird_m/skills/      implémentation de référence BSP (spec, parsing SKILL.md, rendu SFT, runtime)
bird_m/deck/        BIRD-Deck : schéma JSON des slides, validateur, moteur de rendu HTML/SVG
skills/core/        BIRD Core Skills livrées avec le modèle (bird-deck, …)
data/taxonomy/      taxonomie des tâches (source de vérité)
examples/skills/    exemples de skills d'entreprise
docs/               blueprint, taxonomie, spécification
```

## Licence

Apache 2.0. Modèle de base © Qwen Team, Alibaba Cloud (Apache 2.0). Les modifications apportées sont décrites dans `docs/`.
