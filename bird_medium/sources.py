"""Chargeurs de sources Hugging Face -> enregistrements d'entraînement.

Format commun d'un enregistrement :
    {"id", "source", "lang", "thinking": bool, "messages": [...]}
Le dernier message est la réponse de l'assistant (seule partie apprise). En mode raisonnement,
la trace est dans `reasoning_content` (rendu par le chat template Qwen3.8 dans <think>...</think>).

Lecture paresseuse des parquet (groupes de lignes tirés au hasard) pour ne pas tout télécharger.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

_fs = HfFileSystem()


def _row_groups(
    repo: str, glob: str, columns: list[str], rng: random.Random
) -> Iterator[dict]:
    """Itère sur les lignes des parquet d'un dataset, groupes de lignes dans un ordre aléatoire."""
    files = sorted(_fs.glob(f"datasets/{repo}@~parquet/{glob}"))
    if not files:
        raise FileNotFoundError(f"aucun parquet pour {repo}/{glob}")
    groups = []
    for f in files:
        n = pq.ParquetFile(_fs.open(f)).metadata.num_row_groups
        groups += [(f, g) for g in range(n)]
    rng.shuffle(groups)
    for f, g in groups:
        rows = (
            pq.ParquetFile(_fs.open(f)).read_row_group(g, columns=columns).to_pylist()
        )
        rng.shuffle(rows)
        yield from rows


def take(it: Iterator[dict], n: int) -> list[dict]:
    out = []
    for r in it:
        if len(out) >= n:
            break
        out.append(r)
    return out


# ------------------------------------------------------------------ détection de langue (légère)
_FR = set(
    "le la les des une est pour que qui dans pas avec sur vous nous être cette votre sont mais".split()
)
_EN = set(
    "the and is are for that with you this have not your from was will can".split()
)


_ES = set("el los las del por una con para como pero sus muy está también puede es y lo".split())
_IT = set("il di che non per una sono gli della come anche più questo nel essere ed".split())
_PT = set("o os um uma não com para por como mais seu sua são está também ao do da".split())
_NOT_FR_CHARS = re.compile(r"[¿¡ñãõìò]")


def lang_of(text: str) -> str:
    """Détection légère fr/en ; les autres langues romanes (es/it/pt) renvoient 'other'."""
    low = text.lower()
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœãõñìòá']+", low)[:300]
    c = {k: sum(w in v for w in words) for k, v in (("fr", _FR), ("en", _EN), ("es", _ES), ("it", _IT), ("pt", _PT))}
    best = max(c, key=c.get)
    if c[best] < 3:
        return "other"
    others = max(v for k, v in c.items() if k != best)
    if c[best] <= 1.5 * others:
        return "other"
    if best == "fr" and _NOT_FR_CHARS.search(low):
        return "other"
    return best if best in ("fr", "en") else "other"


_LEAK = re.compile(
    r"\b(qwen\w*|alibaba|tongyi|openai|chat ?gpt|gpt-?[345]\w*|anthropic|claude|smollm\w*|hugging ?face|mistral|gemini|llama|meta ai|deepseek)\b",
    re.I,
)


def leaks_identity(text: str) -> bool:
    return bool(_LEAK.search(text))


def split_think(content: str) -> tuple[str, str]:
    """'<think>r</think>a' -> (r, a). Sans balise : ('', content)."""
    m = re.match(r"\s*<think>(.*?)</think>\s*(.*)", content, re.S)
    return (m.group(1).strip(), m.group(2).strip()) if m else ("", content.strip())


# ------------------------------------------------------------------ général FR (avec raisonnement)
def general_fr(n: int, seed: int) -> list[dict]:
    """smoltalk2 / smoltalk_multilingual8_Qwen3_32B_think : réponses FR avec trace de raisonnement."""
    rng = random.Random(seed)
    out = []
    it = _row_groups(
        "HuggingFaceTB/smoltalk2",
        "SFT/smoltalk_multilingual8_Qwen3_32B_think/*.parquet",
        ["messages"],
        rng,
    )
    for r in it:
        if len(out) >= n:
            break
        msgs = r["messages"]
        if (
            len(msgs) != 2
            or msgs[0]["role"] != "user"
            or lang_of(msgs[0]["content"]) != "fr"
        ):
            continue
        reasoning, answer = split_think(msgs[1]["content"])
        if (
            not reasoning
            or not answer
            or lang_of(answer) != "fr"
            or leaks_identity(answer)
            or leaks_identity(reasoning)
            or lang_of(msgs[0]["content"]) != "fr"
            or len(reasoning) > 12000
        ):
            continue
        out.append(
            {
                "id": f"general_fr-{len(out)}",
                "source": "general_fr",
                "lang": "fr",
                "thinking": True,
                "messages": [
                    {"role": "user", "content": msgs[0]["content"].strip()},
                    {
                        "role": "assistant",
                        "content": answer,
                        "reasoning_content": reasoning,
                    },
                ],
            }
        )
    return out


# ------------------------------------------------------------------ e-mails
_FOOTER = re.compile(
    r"(\n\s*(Sinabis [^\n]*(GmbH|SAS|S\.A\.|Ltd|SpA|S\.p\.A|SE|AG)[^\n]*(Register|Registre|RCS|HRB|Handelsregister|Registro)"
    r"|This message is intended|Confidentiality notice|Ce message (est|et ses)|Avis de confidentialit|Le présent message"
    r"|Questo messaggio|Diese Nachricht)).*",
    re.S | re.I,
)
_HISTORY = re.compile(
    r"\n\s*-{5,}\s*(Forwarded message|Message transféré|Original Message|Message d'origine)[^\n]*-*.*",
    re.S | re.I,
)


_SIG_START = re.compile(
    r"\n(--\s*\n|[A-ZÀ-Ý][\w'\-]+(?: [A-ZÀ-Ý][\w'\-]+)?, [A-ZÀ-Ý][\w'\-]+(?: [A-ZÀ-Ý][\w'\-]+)?\s*\n)",
    re.S,
)
_DISCLAIMER = re.compile(
    r"\n[^\n]*(PLEASE CONSIDER THE ENVIRONMENT|Mode vert|Ce courriel et ses pi|n'imprimez|Pensez à l'environnement"
    r"|Think before you print|intended only for|confidenti)[^\n]*",
    re.I,
)


def clean_email(body: str) -> str:
    """Retire l'historique cité, le bloc de signature d'entreprise et les mentions légales."""
    body = _HISTORY.sub("", body or "")
    body = _FOOTER.sub("", body)
    m = _SIG_START.search(body)
    if m and m.start() > 80:
        body = body[: m.start()]
    body = _DISCLAIMER.sub("", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    lines = body.split("\n")
    # formule de politesse en double (préambule du bloc de signature supprimé)
    if len(lines) > 2 and _CLOSING.match(lines[-1]):
        prev = [l for l in lines[:-1] if l.strip()]
        # déjà une formule plus haut, ou mail déjà signé d'un nom court (« à tout à l'heure,\nmarek »)
        signed = prev and len(prev[-1].split()) <= 3 and not prev[-1].rstrip().endswith((".", "?", "!", ":"))
        if any(_CLOSING.match(l) for l in lines[:-1]) or signed:
            body = "\n".join(lines[:-1]).rstrip()
    return body


_CLOSING = re.compile(
    r"^\s*(bien |très )?(cordialement|sincères salutations|bonne journée|à bientôt|merci)[ ,.!]*$"
    r"|^\s*(best|kind|warm)? ?regards[ ,.!]*$|^\s*(cheers|thanks|best)[ ,.!]*$",
    re.I,
)


def sign(body: str, sender: str) -> str:
    """Ajoute le prénom du signataire si le mail se termine sur une formule de politesse sans nom."""
    last = body.rstrip().split("\n")[-1]
    first = sender.split(" ")[0] if sender else ""
    tail = " ".join(body.rstrip().split("\n")[-3:]).lower()
    if not first or not _CLOSING.match(last) or first.lower() in tail.replace(last.lower(), ""):
        return body
    return f"{body.rstrip()}\n{first}"


def _names(people) -> str:
    def one(p):
        n = (p or {}).get("name") or ""
        if "," in n:  # « Nom, Prénom » -> « Prénom Nom »
            last, first = [x.strip() for x in n.split(",", 1)]
            n = f"{first} {last}"
        return n

    return ", ".join(x for x in (one(p) for p in people or []) if x)


EMAIL_PROMPTS = {
    "fr": {
        "new": [
            "Rédige un e-mail professionnel à {to} avec pour objet « {subject} ». Signe au nom de {sender}.",
            "Écris un mail à {to} concernant : {subject}. Je suis {sender}.",
            "Peux-tu me préparer un e-mail pour {to} ? Objet : {subject}. Signataire : {sender}.",
        ],
        "reply": [
            "Voici le dernier message reçu :\n\n{prev}\n\nRédige la réponse de {sender}.",
            "Réponds à cet e-mail au nom de {sender} :\n\n{prev}",
            "Je suis {sender}. Aide-moi à répondre à ce mail :\n\n---\n{prev}\n---",
        ],
    },
    "en": {
        "new": [
            'Write a professional email to {to} with the subject "{subject}". Sign as {sender}.',
            "Draft an email to {to} about: {subject}. I'm {sender}.",
        ],
        "reply": [
            "Here is the last message I received:\n\n{prev}\n\nWrite {sender}'s reply.",
            "Reply to this email on behalf of {sender}:\n\n{prev}",
        ],
    },
}


def emails(n_by_lang: dict[str, int], seed: int) -> list[dict]:
    """sinabis-1m-synthetic-emails : rédaction (1er message d'un fil) et réponse (message suivant du fil)."""
    rng = random.Random(seed)
    want = dict(n_by_lang)
    out: list[dict] = []
    cols = [
        "case_id",
        "position",
        "lang",
        "subject",
        "from",
        "to",
        "body",
        "duplicate_of",
        "calendar_method",
    ]
    files = sorted(
        _fs.glob(
            "datasets/sinabis-group/sinabis-1m-synthetic-emails@~parquet/default/train/*.parquet"
        )
    )
    groups = [
        (f, g)
        for f in files
        for g in range(pq.ParquetFile(_fs.open(f)).metadata.num_row_groups)
    ]
    rng.shuffle(groups)
    for f, g in groups:
        if not any(v > 0 for v in want.values()):
            break
        rows = pq.ParquetFile(_fs.open(f)).read_row_group(g, columns=cols).to_pylist()
        by_key = {(r["case_id"], r["position"]): r for r in rows}
        cands = []
        for r in rows:
            lang = r["lang"]
            if want.get(lang, 0) <= 0 or r["duplicate_of"] or r["calendar_method"]:
                continue
            body = clean_email(r["body"])
            if not (200 <= len(body) <= 3500):
                continue
            sender = _names([r["from"]])
            body = sign(body, sender)
            if r["position"] == 1:
                subject = re.sub(
                    r"^(RE|TR|FW|AW|WG|R|I)\s*:\s*", "", r["subject"] or "", flags=re.I
                )
                to = _names(r["to"]) or ("l'équipe" if lang == "fr" else "the team")
                user = rng.choice(EMAIL_PROMPTS[lang]["new"]).format(
                    to=to, subject=subject, sender=sender
                )
                head = "Objet : " if lang == "fr" else "Subject: "
                cands.append((lang, "new", user, f"{head}{subject}\n\n{body}", ""))
                continue
            prev = by_key.get((r["case_id"], r["position"] - 1))
            if prev is None:
                continue
            pbody = clean_email(prev["body"])
            if not pbody:
                continue
            fr = lang == "fr"
            prev_txt = (
                (f"De : " if fr else "From: ")
                + _names([prev["from"]])
                + "\n"
                + (f"Objet : " if fr else "Subject: ")
                + (prev["subject"] or "")
                + "\n\n"
                + pbody
            )
            user = rng.choice(EMAIL_PROMPTS[lang]["reply"]).format(
                prev=prev_txt, sender=sender
            )
            cands.append((lang, "reply", user, body, prev_txt))
        rng.shuffle(cands)
        for lang, kind, user, answer, prev_txt in cands:
            if want[lang] <= 0:
                continue
            out.append(
                {
                    "id": f"emails-{lang}-{kind}-{len(out)}",
                    "source": "emails",
                    "lang": lang,
                    "thinking": False,
                    "thread_prev": prev_txt,  # message précédent brut (synthèses distillées)
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": answer},
                    ],
                }
            )
            want[lang] -= 1
    return out


# ------------------------------------------------------------------ slides
SLIDE_PROMPTS = {
    "fr": [
        "Prépare le contenu d'une présentation de {n} diapositives intitulée « {title} ».",
        "J'ai besoin d'un deck de {n} slides sur : {title}. Donne-moi le contenu diapositive par diapositive.",
        "Rédige le plan détaillé et le texte des slides pour une présentation « {title} » ({n} diapositives).",
    ],
    "en": [
        'Prepare the content of a {n}-slide presentation titled "{title}".',
        "I need a {n}-slide deck on: {title}. Give me the content slide by slide.",
    ],
}
_SLIDE_NOISE = re.compile(
    r"^(Diapositive|Slide|Folie|Diapositiva)\s+\d+\s*/\s*\d+$", re.I
)


def deck_to_markdown(text: str, lang: str, title: str = "") -> str | None:
    parts = re.split(r"\[Slide (\d+)\]\n", text or "")
    if len(parts) < 3:
        return None
    label = "Diapositive" if lang == "fr" else "Slide"
    blocks = []
    for i in range(1, len(parts), 2):
        lines = [l.strip() for l in parts[i + 1].strip().split("\n") if l.strip()]
        lines = [
            l.lstrip("◦•·✓- ").strip()
            for l in lines
            if not _SLIDE_NOISE.match(l) and not l.startswith("Sinabis ")
        ]
        lines = [l for l in lines if l and not l.isdigit()]
        if not lines:
            continue
        if (
            parts[i] == "1" and title
        ):  # diapositive de titre : le titre du deck en en-tête
            head, rest = title, [l for l in lines if l != title]
        else:
            head, rest = lines[0], lines[1:]
        body = "\n".join(f"- {l}" for l in rest)
        blocks.append(f"## {label} {parts[i]} — {head}\n{body}".rstrip())
    return "\n\n".join(blocks)


def slides(n_by_lang: dict[str, int], seed: int) -> list[dict]:
    """sinabis-100k-synthetic-pptx : brief -> contenu Markdown diapositive par diapositive."""
    rng = random.Random(seed)
    want = dict(n_by_lang)
    out = []
    path = "datasets/sinabis-group/sinabis-100k-synthetic-pptx@~parquet/default/train/0000.parquet"
    pf = pq.ParquetFile(_fs.open(path))
    groups = list(range(pf.metadata.num_row_groups))
    rng.shuffle(groups)
    for g in groups:
        if not any(v > 0 for v in want.values()):
            break
        langs = pf.read_row_group(g, columns=["lang", "slides"]).to_pylist()
        idx = [
            i
            for i, r in enumerate(langs)
            if want.get(r["lang"], 0) > 0 and 5 <= r["slides"] <= 12
        ]
        if not idx:
            continue
        rows = (
            pf.read_row_group(g, columns=["lang", "title", "slides", "text"])
            .take(idx)
            .to_pylist()
        )
        rng.shuffle(rows)
        for r in rows:
            lang = r["lang"]
            if want[lang] <= 0:
                continue
            md = deck_to_markdown(r["text"], lang, r["title"])
            if not md or len(md) > 9000:
                continue
            user = rng.choice(SLIDE_PROMPTS[lang]).format(
                n=r["slides"], title=r["title"]
            )
            out.append(
                {
                    "id": f"slides-{lang}-{len(out)}",
                    "source": "slides",
                    "lang": lang,
                    "thinking": False,
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": md},
                    ],
                }
            )
            want[lang] -= 1
    return out


# ------------------------------------------------------------------ finance EN
_CORP = re.compile(
    r"\b(budget|cash ?flow|ebitda|invoice|balance sheet|income statement|working capital|depreciation|amortization"
    r"|gross margin|net margin|break-?even|roi|npv|irr|payback|forecast|accounts (payable|receivable)|liquidity"
    r"|profit and loss|p&l|vat|pricing|cost of goods|cogs|revenue|expenses|financial ratio|debt|loan|interest)\b",
    re.I,
)


def finance_en(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    it = _row_groups(
        "Josephgflowers/Finance-Instruct-500k",
        "default/train/*.parquet",
        ["user", "assistant"],
        rng,
    )
    for r in it:
        if len(out) >= n:
            break
        u, a = (r["user"] or "").strip(), (r["assistant"] or "").strip()
        if (
            not (30 <= len(u) <= 3000 and 150 <= len(a) <= 5000)
            or not _CORP.search(u)
            or leaks_identity(a)
            or re.search(r"NAME_\d|\[REDACTED|<\|", u + a)
        ):
            continue
        out.append(
            {
                "id": f"finance_en-{len(out)}",
                "source": "finance_en",
                "lang": "en",
                "thinking": False,
                "messages": [
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ],
            }
        )
    return out


# ------------------------------------------------------------------ tableaux (sorties structurées)
def tables_en(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for r in _row_groups(
        "HuggingFaceTB/smoltalk2", "SFT/table_gpt_no_think/*.parquet", ["messages"], rng
    ):
        if len(out) >= n:
            break
        m = r["messages"]
        if len(m) != 2 or len(m[0]["content"]) > 6000:
            continue
        _, ans = split_think(m[1]["content"])
        out.append(
            {
                "id": f"tables_en-{len(out)}",
                "source": "tables_en",
                "lang": "en",
                "thinking": False,
                "messages": [
                    {"role": "user", "content": m[0]["content"].strip()},
                    {"role": "assistant", "content": ans},
                ],
            }
        )
    return out
