---
license: apache-2.0
language: [fr, en]
base_model: Qwen/Qwen3.8-27B
library_name: peft
tags: [french, enterprise, business, reasoning, lora, qwen3.8]
---

# Bird-Medium — adaptateur LoRA

Adaptateur LoRA de **Bird-Medium**, modèle de langage français adapté aux tâches courantes de l'entreprise
(e-mails, présentations, finance, synthèses), entraîné sur [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
Le raisonnement (`<think>`) et le mode direct (`enable_thinking=False`) du modèle de base sont conservés.

## Chargement

Les clés suivent le modèle texte (`AutoModelForCausalLM`) :

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.8-27B", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "adubrou/Bird-Medium-LoRA")
tok = AutoTokenizer.from_pretrained("adubrou/Bird-Medium-LoRA")
```

Pour un checkpoint complet (vLLM, Inference Endpoints), fusionner avec `scripts/merge_job.py` du dépôt du projet.

## Entraînement (v1)

- LoRA r=32, α=64, toutes les couches linéaires, perte sur la dernière réponse uniquement, bf16.
- LR 1e-4 cosine, warmup 3 %, batch effectif 16 séquences, longueur max 6 144 tokens.
- Données : `adubrou/bird-medium-sft` plafonné pour un budget de 10 $ (mails, slides, finance FR, identité et
  auto-distillation Qwen3.8-27B en totalité ; conversations générales et finance EN sous-échantillonnées).

Voir `eval.json` pour la perte d'évaluation et des exemples de générations.
