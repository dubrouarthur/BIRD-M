# Bird-Medium

Modèle de langage **français** adapté aux tâches courantes de l'entreprise (e-mails, présentations, finance,
synthèses), fine-tuné en LoRA depuis [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0)
**en conservant son raisonnement** (`<think>`) et son mode sans raisonnement (`enable_thinking=False`).

## Pipeline

```
configs/mix.yaml ──► scripts/distill.py ──► scripts/build_dataset.py --push ──► scripts/launch.py train ──► launch.py merge ──► scripts/publish.py
 (quotas, base)      Qwen3.8-27B répond     mix + split éval → dataset privé     HF Jobs (H200), LoRA     modèle complet      bucket → repo modèle
                     lui-même (raisonnement)  adubrou/bird-medium-sft             → bucket bird-m-runs     (vision intacte)    adubrou/Bird-Medium
```

| Étape | Commande |
|---|---|
| Tests | `pip install -e '.[dev]' && pytest -q` |
| Distillation (~5 $) | `python scripts/distill.py --n 1700` |
| Dataset | `python scripts/build_dataset.py --push` |
| Entraînement | `python scripts/launch.py train --flavor h200 --timeout 6h` |
| Fusion | `python scripts/launch.py merge --timeout 2h -- --adapter /out/bird-medium-v1/adapter --out /out/bird-medium-v1/merged` |
| Publication | `python scripts/publish.py bird-medium-v1/merged` |

## Données (voir `configs/mix.yaml`)

| Source | Quota | Mode | Licence |
|---|---|---|---|
| Identité Bird-Medium (généré, `bird_medium/identity.py`) | 700 FR / 200 EN | 50 % raisonnement | — |
| E-mails : rédaction et réponse — [sinabis-1m-synthetic-emails](https://huggingface.co/datasets/sinabis-group/sinabis-1m-synthetic-emails) | 2 500 FR / 600 EN | direct | CC-BY-4.0 |
| Slides : contenu diapositive par diapositive — [sinabis-100k-synthetic-pptx](https://huggingface.co/datasets/sinabis-group/sinabis-100k-synthetic-pptx) | 300 FR / 500 EN | direct | CC-BY-4.0 |
| Finance FR : TVA, marges, seuil, BFR, emprunt, amortissement (généré, résultats exacts) | 1 200 FR | raisonnement | — |
| Finance d'entreprise — [Finance-Instruct-500k](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k), filtré | 1 500 EN | direct | Apache-2.0 |
| Tableaux / JSON — [smoltalk2 table_gpt](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | 500 EN | direct | Apache-2.0 |
| Conversations générales FR — [smoltalk2 multilingual8 Qwen3-32B think](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | 4 000 FR | raisonnement | Apache-2.0 |
| **Auto-distillation Qwen3.8-27B** : mails, synthèses de fils, slides | 1 500 FR | raisonnement natif | sorties du modèle de base |

35 % des exemples portent le prompt système Bird-Medium, le reste aucun : l'identité ne dépend pas du prompt.

## Structure

```
bird_medium/   identity.py, finance_fr.py (générateurs), sources.py (chargeurs Hub), mix.py
scripts/       distill.py, build_dataset.py, train_job.py, merge_job.py, launch.py, publish.py
configs/       mix.yaml (source de vérité)
model_card/    fiche du modèle publiée sur le Hub
tests/         pytest (le rendu est testé avec le vrai chat template de Qwen3.8-27B)
```

## Licence

Apache 2.0. Modèle de base © Qwen Team, Alibaba Cloud (Apache 2.0).
