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
model = PeftModel.from_pretrained(base, "adubrou/BIRD-M")
tok = AutoTokenizer.from_pretrained("adubrou/BIRD-M")
```

Pour un checkpoint complet (vLLM, Inference Endpoints), fusionner avec `scripts/merge_job.py` du dépôt du projet.

## Entraînement (v1)

- LoRA r=32, α=64, toutes les couches linéaires, perte sur la dernière réponse uniquement, bf16.
- LR 1e-4 cosine, warmup 3 %, batch effectif 16 séquences, longueur max 6 144 tokens.
- Données : `adubrou/bird-medium-sft` plafonné pour un budget de 10 $ (mails, slides, finance FR, identité et
  auto-distillation Qwen3.8-27B en totalité ; conversations générales et finance EN sous-échantillonnées).

**État de cette version : checkpoint du pas 100 / 150** (le job a été interrompu au pas 105, après 30 min
d'entraînement). Perte d'entraînement ≈ 0,5–0,6 (contre ≈ 2 au départ) ; learning rate 2,75e-5 au moment de la
sauvegarde (décroissance cosine aux deux tiers). Données : 2 400 exemples orientés entreprise
(mails 600, distillation Qwen3.8-27B 450, identité 300, finance FR 300, slides 250, général FR 250, finance EN 150, tableaux 100).
