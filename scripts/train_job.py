# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=5.2", "peft>=0.17", "accelerate", "flash-linear-attention", "huggingface_hub"]
# ///
"""Bird-Medium : fine-tune LoRA de Qwen3.8-27B (raisonnement conservé), à lancer sur Hugging Face Jobs.

Entrées (volumes montés par scripts/launch.py) :
  /data  : dataset adubrou/bird-medium-sft (train.jsonl, eval.jsonl), lecture seule
  /out   : bucket adubrou/bird-m-runs, lecture-écriture (adaptateur, logs, éval)

Chaque exemple est rendu par le chat template officiel de Qwen3.8 :
  - thinking=True  : prompt terminé par "<think>\\n", cible = raisonnement + "</think>" + réponse
  - thinking=False : prompt terminé par "<think>\\n\\n</think>\\n\\n" (enable_thinking=False), cible = réponse
La perte ne porte que sur le dernier message de l'assistant.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

IGNORE = -100


def render(tok, messages: list[dict], thinking: bool) -> tuple[str, str]:
    """(prompt, texte complet) ; le prompt est un préfixe exact du texte complet."""
    kw = {"enable_thinking": thinking, "tokenize": False}
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, **kw)
    full = tok.apply_chat_template(messages, **kw)
    if not full.startswith(prompt):
        raise ValueError(
            "le prompt n'est pas un préfixe du texte complet (chat template inattendu)"
        )
    return prompt, full


def encode(tok, rec: dict, max_len: int) -> dict | None:
    msgs = rec["messages"]
    if msgs[-1]["role"] != "assistant":
        return None
    if not rec.get("thinking"):
        msgs = msgs[:-1] + [
            {k: v for k, v in msgs[-1].items() if k != "reasoning_content"}
        ]
    prompt, full = render(tok, msgs, bool(rec.get("thinking")))
    full = full.rstrip(
        "\n"
    )  # le template ajoute "\n" après <|im_end|> : on s'arrête sur <|im_end|>
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if ids[: len(p_ids)] != p_ids or len(ids) > max_len or len(ids) <= len(p_ids):
        return None
    return {"input_ids": ids, "labels": [IGNORE] * len(p_ids) + ids[len(p_ids) :]}


EVAL_PROMPTS = [
    ("fr", True, "Qui es-tu ?"),
    ("fr", False, "Qui es-tu ?"),
    ("fr", True, "Tu es Qwen ?"),
    ("en", False, "Who are you?"),
    (
        "fr",
        False,
        "Rédige un mail court à un client pour relancer une facture impayée de 1 250 € HT, échue depuis 15 jours.",
    ),
    (
        "fr",
        True,
        "Une facture fait 4 380 € HT avec une TVA à 20 %. Quel est le montant TTC ?",
    ),
    (
        "fr",
        False,
        "Prépare le contenu d'une présentation de 6 diapositives sur les résultats commerciaux du T3.",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/out/bird-medium-v1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument(
        "--max-examples", type=int, default=0, help="0 = tout (test rapide sinon)"
    )
    ap.add_argument("--gen-tokens", type=int, default=800, help="échantillons de fin (x2 en mode raisonnement)")
    ap.add_argument("--flush-wait", type=int, default=180)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"  # cpu : test de fumée uniquement
    t0 = time.time()
    log = lambda *x: print(f"[{round(time.time() - t0)}s]", *x, flush=True)  # noqa: E731
    local = Path("/tmp/run")
    local.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load(name):
        recs = [json.loads(l) for l in open(Path(a.data) / name, encoding="utf-8")]
        if a.max_examples:
            recs = random.Random(0).sample(recs, min(a.max_examples, len(recs)))
        enc = [e for e in (encode(tok, r, a.max_len) for r in recs) if e]
        log(
            name,
            "exemples:",
            len(recs),
            "encodés:",
            len(enc),
            "tokens:",
            sum(len(e["input_ids"]) for e in enc),
        )
        return enc

    train, ev = load("train.jsonl"), load("eval.jsonl")

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id)
        lab = torch.full((len(batch), L), IGNORE)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n], lab[i, :n], att[i, :n] = (
                torch.tensor(b["input_ids"]),
                torch.tensor(b["labels"]),
                1,
            )
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32, device_map=dev
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=a.lora_r,
            lora_alpha=2 * a.lora_r,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(local / "ckpt"),
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        weight_decay=0.0,
        bf16=dev == "cuda",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        report_to="none",
        train_sampling_strategy="random",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=ev,
        data_collator=collate,
    )
    trainer.train()
    metrics = trainer.evaluate()
    log("eval", metrics)

    model.save_pretrained(local / "adapter")
    tok.save_pretrained(local / "adapter")

    # ---- échantillons de génération (identité, mail, finance, slides), avec et sans raisonnement
    model.eval()
    model.config.use_cache = True
    samples = []
    for lang, thinking, q in EVAL_PROMPTS:
        text = tok.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True,
            enable_thinking=thinking,
            tokenize=False,
        )
        x = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.no_grad():
            y = model.generate(
                **x, max_new_tokens=a.gen_tokens * (2 if thinking else 1), do_sample=False
            )
        out = tok.decode(y[0, x["input_ids"].shape[1] :], skip_special_tokens=False)
        samples.append({"lang": lang, "thinking": thinking, "prompt": q, "output": out})
        log(f"--- [{lang} think={thinking}] {q}\n{out}\n")
    (local / "eval.json").write_text(
        json.dumps(
            {"metrics": metrics, "samples": samples, "args": vars(a)},
            ensure_ascii=False,
            indent=1,
        )
    )
    for h in trainer.state.log_history:
        print(json.dumps(h), flush=True)
    (local / "log_history.json").write_text(json.dumps(trainer.state.log_history))

    # ---- copie vers le bucket monté : fichier par fichier, fsync, puis attente de la synchronisation
    dst = Path(a.out)
    for f in sorted(
        p for p in local.rglob("*") if p.is_file() and "ckpt" not in p.parts
    ):
        d = dst / f.relative_to(local)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, d)
        with open(d, "rb+") as fh:
            os.fsync(fh.fileno())
        log("copié", d)
    os.sync()
    log(f"attente de {a.flush_wait}s pour la synchronisation du bucket")
    time.sleep(a.flush_wait)
    log("terminé")


if __name__ == "__main__":
    main()
