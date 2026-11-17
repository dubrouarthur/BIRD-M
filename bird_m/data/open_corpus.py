"""Open-source SFT corpus for BIRD-M: license- and provenance-checked Hugging Face datasets.

Every source below was checked on its dataset card (license + who wrote the answers). Only human answers
or answers from open-weight models with permissive licenses are used as TARGETS (see docs/DATASETS.md).
Records are converted to the plain record format used by bird_m.train.data:
    {id, task_id, lang, source, license, system?, messages: [user/assistant..., user], target}

Excluded on purpose (documented): tool-calling sets in the legacy JSON <tool_call> format (xlam, When2Call)
because Qwen3.8 uses an XML tool format; Llama-generated sets (license naming terms); GPT-generated sets.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pyarrow.parquet as pq

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
CONTROL_RE = re.compile(r"<\|[A-Za-z_0-9]+\|>")
IDENTITY_RE = re.compile(
    r"\b(ChatGPT|OpenAI|GPT-4|GPT-3|Anthropic|Claude|Gemini|Bard|Llama|Meta AI|SmolLM|Qwen|Alibaba|Cohere|Aya)\b",
    re.I,
)
FR_WORDS = {
    "le",
    "la",
    "les",
    "des",
    "est",
    "une",
    "pour",
    "que",
    "vous",
    "dans",
    "et",
    "du",
    "au",
    "avec",
    "sur",
    "nous",
    "pas",
    "qui",
    "ce",
    "sont",
}
EN_WORDS = {
    "the",
    "and",
    "is",
    "to",
    "of",
    "in",
    "for",
    "you",
    "that",
    "with",
    "are",
    "this",
    "on",
    "it",
    "be",
    "can",
    "your",
    "as",
    "an",
    "or",
}


def detect_lang(text: str) -> str | None:
    """Cheap FR/EN detector (stopword ratio); returns None for other languages / undecidable."""
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœæ']+", text.lower())[:400]
    if len(words) < 5:
        return None
    fr = sum(w in FR_WORDS for w in words) / len(words)
    en = sum(w in EN_WORDS for w in words) / len(words)
    if max(fr, en) < 0.08:
        return None
    return "fr" if fr > en else "en"


def clean_answer(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def acceptable(
    messages: list[dict], target: str, min_chars: int = 20, max_chars: int = 7000
) -> bool:
    if not (min_chars <= len(target) <= max_chars):
        return False
    blob = target + " ".join(m["content"] for m in messages)
    if CONTROL_RE.search(blob) or "<think>" in target or "<tool_call>" in blob:
        return False
    if IDENTITY_RE.search(
        target
    ):  # identity contamination ("As ChatGPT..."), off-brand
        return False
    return True


@dataclass(frozen=True)
class Source:
    key: str
    hf_id: str
    files: tuple[str, ...]  # repo-relative paths
    license: str
    provenance: str
    task_id: str
    reader: Callable[
        [list[Path]], Iterator[dict]
    ]  # yields {system?, messages, target, lang_hint?}


# ---------------------------------------------------------------- readers
def _parquet_rows(paths):
    for p in paths:
        yield from pq.read_table(p).to_pylist()


def read_smoltalk(paths):
    """smoltalk2 SFT subsets: 'messages' (system/user/assistant), Qwen3-32B answers with <think> to strip."""
    for r in _parquet_rows(paths):
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
        system = None
        if msgs and msgs[0]["role"] == "system":
            system = msgs.pop(0)["content"].strip() or None
        if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
            continue
        out, ok = [], True
        for i, m in enumerate(msgs):
            want = "user" if i % 2 == 0 else "assistant"
            if m["role"] != want:
                ok = False
                break
            out.append(
                {
                    "role": m["role"],
                    "content": clean_answer(m["content"])
                    if want == "assistant"
                    else m["content"].strip(),
                }
            )
        if ok:
            yield {"system": system, "messages": out[:-1], "target": out[-1]["content"]}


def read_aya(paths):
    for r in _parquet_rows(paths):
        code = r["language_code"]
        if code not in ("fra", "eng"):
            continue
        yield {
            "messages": [{"role": "user", "content": r["inputs"].strip()}],
            "target": r["targets"].strip(),
            "lang_hint": "fr" if code == "fra" else "en",
        }


def read_oasst2(paths):
    """Best-ranked assistant reply to each root prompt (FR/EN), human-written, reviewed, not deleted."""
    rows = list(_parquet_rows(paths))
    by_id = {r["message_id"]: r for r in rows}
    best: dict[str, dict] = {}
    for r in rows:
        if (
            r["role"] != "assistant"
            or r["deleted"]
            or r["rank"] is None
            or not r["review_result"]
        ):
            continue
        parent = by_id.get(r["parent_id"])
        if (
            not parent
            or parent["parent_id"] is not None
            or parent["lang"] not in ("fr", "en")
        ):
            continue
        if r["parent_id"] not in best or r["rank"] < best[r["parent_id"]]["rank"]:
            best[r["parent_id"]] = r
    for pid, r in best.items():
        p = by_id[pid]
        yield {
            "messages": [{"role": "user", "content": p["text"].strip()}],
            "target": r["text"].strip(),
            "lang_hint": p["lang"],
        }


def read_helpsteer_edit(paths):
    """HelpSteer3 'edit': responses rewritten by human annotators -> human-quality targets (non-code domains)."""
    for p in paths:
        for line in gzip.open(p, "rt", encoding="utf-8"):
            r = json.loads(line)
            if r["domain"] not in ("general", "multilingual") or r["language"] not in (
                "english",
                "french",
            ):
                continue
            ctx = [{"role": m["role"], "content": m["content"]} for m in r["context"]]
            if (
                not ctx
                or ctx[-1]["role"] != "user"
                or any(m["role"] not in ("user", "assistant") for m in ctx)
            ):
                continue
            if any(
                ctx[i]["role"] != ("user" if i % 2 == 0 else "assistant")
                for i in range(len(ctx))
            ):
                continue
            yield {
                "messages": ctx,
                "target": r["edited_response"].strip(),
                "lang_hint": "fr" if r["language"] == "french" else "en",
            }


def read_structx(paths):
    """Document + JSON schema -> extracted JSON (fictional documents)."""
    for r in _parquet_rows(paths):
        if r["language"] not in ("fr", "en"):
            continue
        fr = r["language"] == "fr"
        ask = (
            (
                "Extrais les informations de ce document au format JSON, en respectant strictement le schéma. "
                "Mets null pour toute information absente."
            )
            if fr
            else (
                "Extract the information from this document as JSON, strictly following the schema. Use null for missing fields."
            )
        )
        user = (
            f"{ask}\n\nSchéma :\n{r['schema']}\n\nDocument :\n{r['document']}"
            if fr
            else f"{ask}\n\nSchema:\n{r['schema']}\n\nDocument:\n{r['document']}"
        )
        yield {
            "messages": [{"role": "user", "content": user}],
            "target": r["extraction"].strip(),
            "lang_hint": r["language"],
        }


_FOOTER_RE = re.compile(r"^(Ce courriel|Ce message|Mode vert|This e-?mail|This message|Please consider|Avertissement|Disclaimer)", re.I | re.M)


def _strip_footer(text: str) -> str:
    m = _FOOTER_RE.search(text)
    return (text[: m.start()] if m else text).strip()


def read_sinabis_emails(paths):
    """Fictional business email threads (CC-BY-4.0, Qwen3.6 / DeepSeek): received email -> written reply."""
    cols = ["lang", "message_id", "in_reply_to", "model", "duplicate_of", "subject", "body", "from", "to"]
    for p in paths:
        rows = pq.read_table(p, columns=cols).to_pylist()
        by_id = {r["message_id"]: r for r in rows}
        for r in rows:
            if r["lang"] not in ("fr", "en") or not r["model"] or r["duplicate_of"]:
                continue  # skip rule-based calendar mails and duplicates
            parent = by_id.get(r["in_reply_to"])
            if not parent or not parent["body"].strip():
                continue
            fr = r["lang"] == "fr"
            sender = parent["from"]["name"] if parent.get("from") else ""
            ask = (f"Rédige la réponse à cet e-mail reçu de {sender}. Objet : {parent['subject']}" if fr
                   else f"Write the reply to this email from {sender}. Subject: {parent['subject']}")
            yield {"messages": [{"role": "user", "content": f"{ask}\n\n---\n{_strip_footer(parent['body'])}"}],
                   "target": _strip_footer(r["body"]), "lang_hint": r["lang"]}


_FAMILY = {"letter": ("une lettre", "a letter"), "contract": ("un contrat", "a contract"), "corporate": ("un document interne", "an internal document"),
           "report": ("un rapport", "a report"), "commercial": ("un document commercial", "a commercial document")}


def read_sinabis_docx(paths):
    """Fictional business documents (CC-BY-4.0, Gemma 4 / Apache-2.0): brief -> full document text. One per blueprint."""
    cols = ["lang", "family", "title", "issuer", "recipient", "theme", "blueprint_id", "text", "words"]
    seen = set()
    for p in paths:
        for r in pq.read_table(p, columns=cols).to_pylist():
            if r["lang"] not in ("fr", "en") or r["blueprint_id"] in seen or not r["text"] or not (120 <= (r["words"] or 0) <= 1100):
                continue
            seen.add(r["blueprint_id"])
            fr = r["lang"] == "fr"
            kind = _FAMILY.get(r["family"], ("un document", "a document"))[0 if fr else 1]
            theme = r["theme"].replace("_", " ")
            ask = (f"Rédige {kind} ({theme}) intitulé « {r['title']} », émis par {r['issuer']} à l'attention de {r['recipient']}."
                   if fr else f"Write {kind} ({theme}) titled \"{r['title']}\", issued by {r['issuer']} to {r['recipient']}.")
            yield {"messages": [{"role": "user", "content": ask}], "target": r["text"].strip(), "lang_hint": r["lang"]}


ST = "HuggingFaceTB/smoltalk2"
SOURCES: tuple[Source, ...] = (
    Source(
        "smoltalk_ml8_fr",
        ST,
        ("SFT/smoltalk_multilingual8_Qwen3_32B_think-00000-of-00004.parquet",),
        "apache-2.0",
        "prompts translated to 8 languages, answers by Qwen3-32B",
        "open.chat",
        read_smoltalk,
    ),
    Source(
        "smoltalk_aya_qwen",
        ST,
        ("SFT/aya_dataset_Qwen3_32B_think-00000-of-00001.parquet",),
        "apache-2.0",
        "Aya prompts, answers by Qwen3-32B",
        "open.chat",
        read_smoltalk,
    ),
    Source(
        "smoltalk_systemchats",
        ST,
        ("SFT/smoltalk_systemchats_Qwen3_32B_think-00000-of-00001.parquet",),
        "apache-2.0",
        "system-prompt following, answers by Qwen3-32B",
        "open.system",
        read_smoltalk,
    ),
    Source(
        "smoltalk_tables",
        ST,
        ("SFT/table_gpt_Qwen3_32B_think-00000-of-00001.parquet",),
        "apache-2.0",
        "table tasks, answers by Qwen3-32B",
        "open.tables",
        read_smoltalk,
    ),
    Source(
        "smoltalk_everyday",
        ST,
        (
            "SFT/smoltalk_everyday_convs_reasoning_Qwen3_32B_think-00000-of-00001.parquet",
        ),
        "apache-2.0",
        "everyday conversations, answers by Qwen3-32B",
        "open.chat",
        read_smoltalk,
    ),
    Source(
        "aya_human",
        "CohereLabs/aya_dataset",
        ("data/train-00000-of-00001.parquet",),
        "apache-2.0",
        "human-written prompts and answers (Aya annotators)",
        "open.human",
        read_aya,
    ),
    Source(
        "oasst2",
        "OpenAssistant/oasst2",
        ("data/train-00000-of-00001-88ba0162028a73fc.parquet",),
        "apache-2.0",
        "human conversations, best-ranked human reply",
        "open.human",
        read_oasst2,
    ),
    Source(
        "helpsteer3_edit",
        "nvidia/HelpSteer3",
        ("edit/train.jsonl.gz",),
        "cc-by-4.0",
        "responses edited by human annotators",
        "open.human",
        read_helpsteer_edit,
    ),
    Source(
        "structured_extraction",
        "LiquidAI/structured-extraction",
        ("train.parquet",),
        "apache-2.0",
        "fictional documents, schema-checked extractions",
        "open.extraction",
        read_structx,
    ),
    Source("sinabis_emails", "sinabis-group/sinabis-1m-synthetic-emails",
           ("data/train-00000-of-00013.parquet", "data/train-00001-of-00013.parquet"), "cc-by-4.0",
           "fictional business email threads; replies by Qwen3.6 / DeepSeek (open weights)", "open.email", read_sinabis_emails),
    Source("sinabis_docs", "sinabis-group/sinabis-100k-synthetic-docx", ("data/train-00000-of-00001.parquet",), "cc-by-4.0",
           "fictional business documents written by Gemma 4 (Apache-2.0)", "open.document", read_sinabis_docx),
)

# how many records to keep per source (FR is preferred where available)
DEFAULT_CAPS = {
    "smoltalk_ml8_fr": {"fr": 2500, "en": 0},
    "smoltalk_aya_qwen": {"fr": 800, "en": 300},
    "smoltalk_systemchats": {"fr": 300, "en": 1200},
    "smoltalk_tables": {"fr": 0, "en": 600},
    "smoltalk_everyday": {"fr": 200, "en": 300},
    "aya_human": {"fr": 1500, "en": 800},
    "oasst2": {"fr": 1500, "en": 1000},
    "helpsteer3_edit": {"fr": 400, "en": 1200},
    "structured_extraction": {"fr": 400, "en": 400},
    "sinabis_emails": {"fr": 3500, "en": 1500},
    "sinabis_docs": {"fr": 200, "en": 800},
}


def fetch(src: Source, root: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download

    return [
        Path(
            hf_hub_download(src.hf_id, f, repo_type="dataset", local_dir=root / src.key)
        )
        for f in src.files
    ]


def build(
    root: Path, caps: dict | None = None, seed: int = 0, download: bool = True
) -> tuple[list[dict], dict]:
    caps = caps or DEFAULT_CAPS
    rng = random.Random(seed)
    out, stats, seen = [], {}, set()
    for src in SOURCES:
        paths = (
            fetch(src, root) if download else [root / src.key / f for f in src.files]
        )
        pools: dict[str, list[dict]] = {"fr": [], "en": []}
        n_read = rejected = 0
        for ex in src.reader(paths):
            n_read += 1
            text = " ".join(m["content"] for m in ex["messages"]) + " " + ex["target"]
            lang = ex.get("lang_hint") or detect_lang(text)
            if lang not in pools or detect_lang(ex["target"]) not in (lang, None):
                rejected += 1
                continue
            if not acceptable(ex["messages"], ex["target"]):
                rejected += 1
                continue
            h = hashlib.sha1((ex["messages"][-1]["content"][:500]).encode()).hexdigest()
            if h in seen:
                rejected += 1
                continue
            seen.add(h)
            pools[lang].append(ex)
        kept = Counter()
        for lang, pool in pools.items():
            rng.shuffle(pool)
            for i, ex in enumerate(pool[: caps.get(src.key, {}).get(lang, 0)]):
                rec = {
                    "id": f"{src.key}-{lang}-{i:06d}",
                    "task_id": src.task_id,
                    "lang": lang,
                    "source": f"hf:{src.hf_id}",
                    "license": src.license,
                    "messages": ex["messages"],
                    "target": ex["target"],
                }
                if ex.get("system"):
                    rec["system"] = ex["system"]
                out.append(rec)
                kept[lang] += 1
        stats[src.key] = {
            "hf_id": src.hf_id,
            "license": src.license,
            "provenance": src.provenance,
            "read": n_read,
            "rejected": rejected,
            "available": {k: len(v) for k, v in pools.items()},
            "kept": dict(kept),
        }
    rng.shuffle(out)
    return out, stats
