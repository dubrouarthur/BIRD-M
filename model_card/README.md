---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: finetune
library_name: transformers
pipeline_tag: image-text-to-text
language:
  - fr
  - en
tags:
  - qwen3_5
  - fine-tuned
  - enterprise
  - french
  - business-writing
  - email
  - slides
  - finance
  - bird-m
datasets:
  - adubrou/bird-m-sft
  - sinabis-group/sinabis-1m-synthetic-emails
  - sinabis-group/sinabis-100k-synthetic-docx
  - HuggingFaceTB/smoltalk2
  - CohereLabs/aya_dataset
  - OpenAssistant/oasst2
  - nvidia/HelpSteer3
  - LiquidAI/structured-extraction
  - Benitoow/OfficeSmith-PPTX-IR
model-index:
  - name: BIRD-M v0.1
    results:
      - task:
          type: text-generation
          name: Financial analysis (exact figures)
        dataset:
          type: adubrou/bird-bench
          name: BIRD-Bench finance (held-out companies)
          config: finance
          split: test
        metrics:
          - type: exact_match
            value: 43.3
            name: All figures exact (%)
      - task:
          type: text-generation
          name: Skill routing, special tokens (experimental)
        dataset:
          type: adubrou/bird-bench
          name: BIRD-Bench skills (unseen skills)
          config: skills
          split: test
        metrics:
          - type: accuracy
            value: 35.0
            name: Exact routing (%)
---

# BIRD-M v0.1: an assistant for the everyday work of French companies

**BIRD-M** is a fine-tune of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) specialised in everyday business tasks, in **French** and English:

- writing and replying to professional emails;
- business documents (letters, reports, notes);
- slide decks in a structured JSON format (BIRD-Deck);
- financial analysis;
- extracting information from documents into JSON;
- following a company's own procedures ("skills").

The goal of the project is the best French model **per euro of serving cost** for these tasks. BIRD-M answers directly, without long reasoning, by default.

> ⚠️ **v0.1 is a pilot release.** It was trained once, on a small budget: 6,351 examples, 1 epoch. It is published to share the method and the first measurements, **not as a production model**. See [Bias, risks and limitations](#bias-risks-and-limitations).

## Model details

|                        |                                                                                                                                           |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Developed by**       | Bird Models                                                                                                                               |
| **Model type**         | Decoder-only language model with a vision encoder: hybrid Gated DeltaNet + gated attention, 27B parameters, same architecture as the base |
| **Languages**          | French (primary), English                                                                                                                 |
| **License**            | Apache-2.0                                                                                                                                |
| **Fine-tuned from**    | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0)                                                                  |
| **Fine-tuning method** | LoRA, merged into the full weights. The vision tower and MTP head are unchanged.                                                          |
| **Version**            | 0.1 (pilot), November 2026                                                                                                                |

## Uses

### Direct use

- Drafting and replying to business emails, with a professional tone and correct French conventions (vouvoiement, greetings, sign-offs).
- Drafting business documents: letters, internal notes, reports, contract outlines.
- Producing slide-deck outlines as BIRD-Deck JSON, to be rendered by an editor.
- Financial ratios, intermediate management balances (SIG) and cash-flow explanations from statements. The calculation conventions must be stated in the prompt.
- Extracting fields from documents into a given JSON schema.
- General assistant use in French and English.

### Company skills (recommended: text mode)

A company describes its procedures, writing charter and policies in short `SKILL.md` files.

- **In the system prompt:** a catalogue of the skills (name, description, when to use).
- **At answer time:** the full text of the relevant skill is inserted.
- **Precedence:** policies (for example GDPR or confidentiality) take precedence over user requests.

The model's weights never need to change.

### Out-of-scope use

- Legal, tax, accounting or medical advice without review by a qualified professional.
- Fully automated decisions about people (hiring, credit, HR).
- Factual questions about current events: the model has no internet access of its own.
- Safety-critical systems.

## Bias, risks and limitations

- **Pilot scale.** The training set is small (6,351 examples, 1 epoch), so quality is uneven. Meeting minutes, proofreading and FR↔EN translation had little or no dedicated data.
- **Synthetic data.** About 60 % of the training answers were written by open-weight models (Qwen3, Qwen3.5/3.6, DeepSeek, Gemma 4) for fictional companies. They may carry those models' biases and a uniform "corporate" style.
- **Figures.** Accuracy improved on our finance tasks, but numbers can still be wrong. Always check important figures.
- **Special-token skill routing is under-trained.** The experimental BIRD-M Skill Protocol mostly declines to invoke a skill (see [Evaluation](#evaluation)). Use skills in text mode.
- **Tool calling** was not trained in this version, and the base model's native tool format may have drifted slightly.
- **Identity.** The model introduces itself as BIRD-M, by Bird Models. When an operator defines a persona in the system prompt, it adopts that persona.

**Recommendations.**

- Keep a human in the loop for anything sent outside the company.
- State calculation conventions explicitly.
- Test on your own documents before deploying.

## How to get started

**Transformers**

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer

tok = AutoTokenizer.from_pretrained("adubrou/BIRD-M")
model = AutoModelForImageTextToText.from_pretrained("adubrou/BIRD-M", dtype="bfloat16", device_map="auto")

msgs = [{"role": "user", "content": "Rédige un e-mail pour relancer un client sur une facture impayée de 4 200 € échue depuis 20 jours."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=600, temperature=0.7, top_p=0.8, top_k=20, do_sample=True)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

**vLLM / Hugging Face Inference Endpoints**

Deploy `adubrou/BIRD-M` like `Qwen/Qwen3.8-27B`: 1× A100 80 GB or 1× H200 in bf16, or a smaller GPU once quantised.

- Recommended sampling: `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`.
- Pass `chat_template_kwargs={"enable_thinking": false}`.

## Training details

### Training data

6,351 examples (4.64 M tokens). Every source was checked for license **and** for who wrote the answers. Only human answers or answers from open-weight models with permissive licenses are used. No outputs of closed models are included.

| Source                                                                                                                                         | License    | Answers written by                       |         Examples |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------------------------------------- | ---------------: |
| [sinabis-group/sinabis-1m-synthetic-emails](https://huggingface.co/datasets/sinabis-group/sinabis-1m-synthetic-emails): business email replies | CC-BY-4.0  | Qwen3.6 / DeepSeek (fictional companies) | 1,300 (1,000 FR) |
| [sinabis-group/sinabis-100k-synthetic-docx](https://huggingface.co/datasets/sinabis-group/sinabis-100k-synthetic-docx): business documents     | CC-BY-4.0  | Gemma 4 (Apache-2.0)                     |              200 |
| [Benitoow/OfficeSmith-PPTX-IR](https://huggingface.co/datasets/Benitoow/OfficeSmith-PPTX-IR): decks converted to BIRD-Deck                     | Apache-2.0 | Qwen3.5-397B                             |     300 (200 FR) |
| [HuggingFaceTB/smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2): Qwen3-32B subsets (French chat, system prompts, tables)    | Apache-2.0 | Qwen3-32B (reasoning removed)            |            1,250 |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2): best-ranked replies                                              | Apache-2.0 | Humans                                   |              601 |
| [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset)                                                               | Apache-2.0 | Humans                                   |              400 |
| [nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3): `edit`                                                                 | CC-BY-4.0  | Humans (edited responses)                |              200 |
| [LiquidAI/structured-extraction](https://huggingface.co/datasets/LiquidAI/structured-extraction)                                               | Apache-2.0 | Fictional documents                      |              200 |
| [adubrou/bird-m-sft](https://huggingface.co/datasets/adubrou/bird-m-sft): finance                                                              | Apache-2.0 | Computed by code (exact figures)         |              500 |
| [adubrou/bird-m-sft](https://huggingface.co/datasets/adubrou/bird-m-sft): skill routing                                                        | Apache-2.0 | Generated by code                        |            1,000 |
| [adubrou/bird-m-sft](https://huggingface.co/datasets/adubrou/bird-m-sft): identity                                                             | Apache-2.0 | Templates                                |              400 |

### Preprocessing

- French and English only (language detection on prompts and answers).
- `<think>` blocks removed from the targets: BIRD-M answers directly.
- Answers that mention other assistants or vendors are removed, to avoid identity contamination.
- Near-duplicate prompts are removed, and so are records containing control tokens.
- 40 % of the open-source examples are trained **without a system prompt**, so behaviour does not depend on one.
- The loss is computed **only on the assistant's answers**. Prompts and injected skill text are masked.

Excluded on purpose:

- sets with GPT-4 or Claude outputs (OpenHermes, UltraFeedback, Tulu);
- Llama-generated sets, because of the license naming terms;
- tool-calling sets in a JSON format incompatible with Qwen3.8 (xlam, When2Call).

### Training hyperparameters

|                     |                                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| Method              | LoRA r=32, α=64, dropout 0, on all linear layers of the language model (attention, DeltaNet, MLP) |
| New token rows      | 211 rows in `embed_tokens` and `lm_head` trained in full, lr 1e-3                                 |
| Optimiser           | AdamW, lr 1e-4, weight decay 0, cosine schedule, 5 % warm-up, gradient clipping 1.0               |
| Batch               | Batches grouped by length, ~16k tokens per step, 296 steps, 1 epoch                               |
| Precision           | bf16, gradient checkpointing                                                                      |
| Max sequence length | 4,096 tokens                                                                                      |
| Training loss       | 1.03 at step 10, 0.58 at step 296                                                                 |

### Speeds, sizes, times

- Training: 47.5 min on 1× NVIDIA H200 141 GB, ~1,600 tokens/s at steady state.
- LoRA merge: ~3 min on the same hardware.
- Checkpoint: 54.7 GB in bf16 (12 safetensors shards).

## Evaluation

### Testing data

**BIRD-Bench** (private). It uses fictional companies and skills that **never appear in training**:

- **Finance:** 30 tasks covering ratios, SIG tables and cash-flow bridges, on held-out companies, with the calculation conventions stated.
- **Skill routing:** 100 episodes with catalogues of 3 to 25 procedures, including 5 procedure types never seen in training.

### Metrics

- **Finance:** a task counts as correct only if every expected figure appears, within 1 % (FR or EN number formats).
- **Routing:** exact match of the set of invoked skills.

### Results

| Test                                               |   n | BIRD-M v0.1 |
| -------------------------------------------------- | --: | ----------: |
| Finance, all figures exact                         |  30 |  **43.3 %** |
| Finance: SIG and cash-flow bridge                  |  15 |      80.0 % |
| Finance: ratios                                    |  15 |       6.7 % |
| Special-token routing: "no skill needed"           |  23 |      95.7 % |
| Special-token routing: policy forbids the request  |   9 |       100 % |
| Special-token routing: pick the right unseen skill |  52 |       7.7 % |
| Special-token routing: compose two skills          |  16 |         0 % |

### Summary

- **Finance tables:** structured tables are learned well (80 %). Ratio computations are not yet reliable (6.7 %).
- **Base model reference:** Qwen3.8-27B scored 18 % on an earlier version of these prompts that did not state the conventions. A like-for-like comparison is planned for v0.2.
- **Special-token routing** is under-trained, with only 1,000 episodes: the model defaults to "no skill". Use skills in text mode; with a plain-text catalogue, the base model already picks the right skill in 88 % of cases.

## Environmental impact

- **Hardware:** 1× NVIDIA H200, on Hugging Face Jobs.
- **Hours used:** ≈ 1.5 GPU-hours for this model (training, merge and evaluation), ≈ 2.5 GPU-hours including the pilot runs.
- **Estimated emissions:** under 1 kg CO₂eq, assuming ~0.7 kW per GPU and an average grid. Not measured.

## Technical specifications

- **Architecture:** Qwen3.8-27B.
  - 64 layers: 16 blocks of 3 Gated DeltaNet layers (linear attention) followed by 1 gated full-attention layer.
  - Hidden size 5,120; vision encoder; multi-token-prediction head.
  - Native context of 262k tokens.
- **Tokenizer:** the Qwen tokenizer plus 211 special tokens for the experimental skill protocol. They occupy ids 248,077–248,287, which were unused padded rows, so the shapes are unchanged.
- **Software:** PyTorch, Transformers ≥ 5, PEFT, flash-linear-attention.

## License and attribution

- **This model:** Apache-2.0.
- **Base model:** Qwen3.8-27B © Qwen Team, Alibaba Cloud (Apache-2.0).

Training data under **CC-BY-4.0** (attribution):

- _Sinabis Group_: synthetic emails and documents;
- _NVIDIA_: HelpSteer3.

Training data under Apache-2.0:

- Hugging Face: SmolTalk2;
- Cohere Labs: Aya Dataset;
- OpenAssistant: OASST2;
- Liquid AI: Structured Extraction;
- OfficeSmith: PPTX-IR.

## Citation

```bibtex
@misc{birdm2026,
  title  = {BIRD-M: an open enterprise assistant for French companies},
  author = {{Bird Models}},
  year   = {2026},
  url    = {https://huggingface.co/adubrou/BIRD-M}
}
```

## Model card contact

Bird Models: open a discussion on this repository.
