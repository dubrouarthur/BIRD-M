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


def render(tok, messages: list[dict], thinking: bool, effort: str | None = None) -> tuple[str, str]:
    """(prompt, texte complet) ; le prompt est un préfixe exact du texte complet.

    effort : "low" pour les traces courtes (identité, calculs) afin de ne pas apprendre au modèle que
    l'effort "xhigh" par défaut veut dire raisonner en deux lignes.
    """
    kw = {"enable_thinking": thinking, "tokenize": False}
    if thinking and effort:
        kw["reasoning_effort"] = effort
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
    prompt, full = render(tok, msgs, bool(rec.get("thinking")), rec.get("reasoning_effort"))
    full = full.rstrip(
        "\n"
    )  # le template ajoute "\n" après <|im_end|> : on s'arrête sur <|im_end|>
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if ids[: len(p_ids)] != p_ids or len(ids) > max_len or len(ids) <= len(p_ids):
        return None
    return {"input_ids": ids, "labels": [IGNORE] * len(p_ids) + ids[len(p_ids) :]}


# Sondes de fin de run : formulations absentes des données d'entraînement (pas de simple mémorisation).
EVAL_PROMPTS = [
    ("fr", True, "Dis-moi en quelques mots qui tu es et à quoi tu sers."),
    ("fr", False, "Avant de commencer : tu es quel modèle, et qui t'a conçu ?"),
    ("fr", True, "Franchement, t'es pas juste ChatGPT ou Qwen avec un autre nom ?"),
    ("en", False, "Quick question before we start: what AI am I talking to?"),
    ("fr", False, "Rédige un mail court à un client pour relancer une facture impayée de 1 250 € HT, échue depuis 15 jours."),
    ("fr", True, "On a vendu pour 18 600 € TTC de prestations au taux normal. Combien de TVA doit-on reverser ?"),
    ("fr", False, "Prépare le contenu d'une présentation de 6 diapositives sur les résultats commerciaux du T3."),
]


def push(src: Path, dst: Path, log) -> None:
    """Copie un dossier vers le bucket monté, fichier par fichier ; fsync best effort (FUSE)."""
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        d = dst / f.relative_to(src)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, d)
        try:
            fd = os.open(d, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        except OSError as e:
            log("fsync ignoré", d, e)
        log("copié", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--data", default="/dataset")  # /data est réservé par HF Jobs (scripts locaux)
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
    ap.add_argument("--save-steps", type=int, default=100, help="copie de l'adaptateur vers le bucket")
    ap.add_argument("--flush-wait", type=int, default=180)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
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
        eval_steps=200,
        save_strategy="no",
        report_to="none",
        train_sampling_strategy="random",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    dst = Path(a.out)

    class SaveAdapter(TrainerCallback):
        """Adaptateur courant copié dans le bucket tous les --save-steps : un run interrompu laisse un résultat."""

        prev: Path | None = None

        def on_step_end(self, args_, state, control, **kw):
            if state.global_step % a.save_steps:
                return
            # nouveau dossier à chaque fois (pas de réécriture de fichiers existants sur le montage FUSE)
            name = f"adapter-step{state.global_step}"
            d = local / name
            model.save_pretrained(d)
            tok.save_pretrained(d)
            push(d, dst / name, log)
            shutil.rmtree(d, ignore_errors=True)
            if self.prev is not None:
                shutil.rmtree(self.prev, ignore_errors=True)  # garde seulement le dernier dans le bucket
            self.prev = dst / name

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=ev,
        data_collator=collate,
        callbacks=[SaveAdapter()],
    )
    trainer.train()

    # adaptateur final dans le bucket AVANT l'éval et les générations (qui peuvent échouer)
    model.save_pretrained(local / "adapter")
    tok.save_pretrained(local / "adapter")
    push(local / "adapter", dst / "adapter", log)
    metrics = trainer.evaluate()
    log("eval", metrics)

    # ---- échantillons de génération (identité, mail, finance, slides), avec et sans raisonnement
    model.eval()
    model.config.use_cache = True
    samples = []
    for lang, thinking, q in EVAL_PROMPTS:
        try:
            text = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                           enable_thinking=thinking, tokenize=False)
            x = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
            with torch.no_grad():  # échantillonnage recommandé par Qwen (le greedy boucle en mode raisonnement)
                y = model.generate(**x, max_new_tokens=a.gen_tokens * (2 if thinking else 1), do_sample=True,
                                   temperature=0.6 if thinking else 0.7, top_p=0.95 if thinking else 0.8, top_k=20)
            out = tok.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=False)
        except Exception as e:  # noqa: BLE001 — l'adaptateur est déjà sauvegardé
            out = f"ERREUR: {e!r}"
        samples.append({"lang": lang, "thinking": thinking, "prompt": q, "output": out})
        log(f"--- [{lang} think={thinking}] {q}\n{out}\n")

    report = local / "report"
    report.mkdir(exist_ok=True)
    (report / "eval.json").write_text(json.dumps({"metrics": metrics, "samples": samples, "args": vars(a)},
                                                 ensure_ascii=False, indent=1))
    (report / "log_history.json").write_text(json.dumps(trainer.state.log_history))
    for h in trainer.state.log_history:
        print(json.dumps(h), flush=True)
    push(report, dst, log)
    os.sync()
    log(f"attente de {a.flush_wait}s pour la synchronisation du bucket")
    time.sleep(a.flush_wait)
    log("terminé")


if __name__ == "__main__":
    main()
