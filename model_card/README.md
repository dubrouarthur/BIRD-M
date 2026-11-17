---
license: apache-2.0
language: [fr, en]
base_model: Qwen/Qwen3.8-27B
library_name: transformers
pipeline_tag: image-text-to-text
tags: [french, enterprise, business, reasoning, lora, qwen3.8]
---

# Bird-Medium

**Bird-Medium** est un modèle de langage français adapté aux tâches courantes de l'entreprise :
rédaction et réponse aux e-mails, contenu de présentations, calculs et analyses financières simples,
synthèses de fils de discussion, reformulation. Il comprend aussi l'anglais.

Il est construit à partir de [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0) par
fine-tuning LoRA, fusionné dans un checkpoint complet. Le **raisonnement** du modèle de base est conservé.

## Utilisation

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
model_id = "adubrou/Bird-Medium"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype="auto", device_map="auto")
messages = [{"role": "user", "content": "Rédige un mail de relance pour une facture impayée de 1 250 € HT."}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                       return_dict=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=800)
print(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

- `enable_thinking=True` (par défaut) : le modèle raisonne dans `<think>…</think>` avant de répondre (calculs, analyses).
- `enable_thinking=False` : réponse directe (mails, reformulation).
- vLLM : `vllm serve adubrou/Bird-Medium --reasoning-parser qwen3`.

## Données d'entraînement

Environ 13 000 exemples, à 70 % en français : e-mails synthétiques (rédaction et réponse), contenus de
présentations, exercices de finance d'entreprise à résultat exact, conversations générales en français avec
raisonnement, et auto-distillation de Qwen3.8-27B sur des tâches d'entreprise en français. Détail et licences :
voir le dépôt du projet (`configs/mix.yaml`).

## Limites

Bird-Medium peut se tromper, en particulier sur les chiffres, les points juridiques et les faits récents.
Il n'a pas accès à Internet. Relisez ses productions avant envoi ; faites valider les décisions financières
ou juridiques par un professionnel.

## Résultats

_À compléter après l'entraînement (perte d'évaluation, échantillons avant/après)._

## Licence

Apache 2.0. Modèle de base © Qwen Team, Alibaba Cloud.
