"""Finance tasks generated from BIRD-World companies.

Two kinds of records:
- "exact": the target answer is fully computed and rendered by templates (ratios, SIG, cash bridge).
  These are training-ready without any LLM.
- "teacher": the prompt + ground-truth facts are computed; the prose answer is written later by the
  Apache-licensed teacher (Qwen3.8-27B) and kept only if `numbers_check(answer, facts)` passes.

Record schema (JSONL): {id, task_id, lang, source, seed, kind, messages, target, facts, checks}
"""

from __future__ import annotations

import math
import random

from bird_m.deck.render import fmt_num
from bird_m.world.finance import (
    Company,
    FinancialYear,
    cash_bridge,
    generate_company,
    ratios,
)

SOURCE = "bird-world/finance@0.1"

# ---------------------------------------------------------------- labels
LINES = [  # (field, fr, en, section)
    ("chiffre_affaires", "Chiffre d'affaires", "Revenue", "cr"),
    ("achats_consommes", "Achats consommés", "Cost of goods sold", "cr"),
    (
        "charges_externes",
        "Autres achats et charges externes",
        "Other external expenses",
        "cr",
    ),
    ("impots_taxes", "Impôts et taxes", "Taxes other than income tax", "cr"),
    ("charges_personnel", "Charges de personnel", "Staff costs", "cr"),
    ("dotations_amortissements", "Dotations aux amortissements", "Depreciation", "cr"),
    ("resultat_financier", "Résultat financier", "Net financial result", "cr"),
    ("resultat_exceptionnel", "Résultat exceptionnel", "Exceptional items", "cr"),
    ("impot_societes", "Impôt sur les sociétés", "Corporate income tax", "cr"),
    ("resultat_net", "Résultat net", "Net income", "cr"),
    ("immobilisations_nettes", "Immobilisations nettes", "Net fixed assets", "actif"),
    ("stocks", "Stocks", "Inventories", "actif"),
    (
        "creances_clients",
        "Créances clients (TTC)",
        "Trade receivables (incl. VAT)",
        "actif",
    ),
    ("tresorerie", "Disponibilités", "Cash", "actif"),
    ("capitaux_propres", "Capitaux propres", "Shareholders' equity", "passif"),
    ("dettes_financieres", "Dettes financières", "Financial debt", "passif"),
    ("concours_bancaires", "Concours bancaires courants", "Bank overdrafts", "passif"),
    (
        "dettes_fournisseurs",
        "Dettes fournisseurs (TTC)",
        "Trade payables (incl. VAT)",
        "passif",
    ),
    (
        "dettes_fiscales_sociales",
        "Dettes fiscales et sociales",
        "Tax and social liabilities",
        "passif",
    ),
]
FLOWS = [
    ("investissements", "Investissements de l'exercice", "Capital expenditure"),
    ("dividendes", "Dividendes versés", "Dividends paid"),
]
RATIO_LABELS = {
    "taux_marge_brute": (
        "Taux de marge brute",
        "Gross margin",
        "%",
        "marge brute / CA",
    ),
    "taux_ebe": ("Taux d'EBE", "EBITDA margin", "%", "EBE / CA"),
    "taux_resultat_exploitation": (
        "Taux de résultat d'exploitation",
        "Operating margin",
        "%",
        "résultat d'exploitation / CA",
    ),
    "marge_nette": ("Marge nette", "Net margin", "%", "résultat net / CA"),
    "rentabilite_capitaux_propres": (
        "Rentabilité des capitaux propres",
        "Return on equity",
        "%",
        "résultat net / capitaux propres",
    ),
    "gearing": (
        "Gearing (endettement net / capitaux propres)",
        "Gearing (net debt / equity)",
        "%",
        "dette nette / capitaux propres",
    ),
    "bfr_jours_ca": (
        "BFR en jours de CA",
        "Working capital in days of revenue",
        "j",
        "BFR / CA × 365",
    ),
    "delai_clients_jours": (
        "Délai de paiement clients",
        "Days sales outstanding",
        "j",
        "créances clients TTC / (CA × 1,2) × 365",
    ),
    "delai_fournisseurs_jours": (
        "Délai de paiement fournisseurs",
        "Days payable outstanding",
        "j",
        "dettes fournisseurs TTC / ((achats + charges externes) × 1,2) × 365",
    ),
    "rotation_stocks_jours": (
        "Rotation des stocks",
        "Days inventory outstanding",
        "j",
        "stocks / achats consommés × 365",
    ),
    "capacite_remboursement_annees": (
        "Capacité de remboursement",
        "Debt payback",
        "ans",
        "dettes financières / CAF",
    ),
    "liquidite_generale": (
        "Liquidité générale",
        "Current ratio",
        "",
        "actif circulant / dettes à court terme",
    ),
}
PERSONAS = {
    "fr": [
        "Je suis DAF",
        "Je suis contrôleur de gestion",
        "Je dirige une PME",
        "Je suis chargé d'affaires dans une banque",
        "Je prépare un comité de direction",
        "Je suis expert-comptable",
        "Je suis analyste crédit",
    ],
    "en": [
        "I am the CFO",
        "I am a financial controller",
        "I run an SME",
        "I am a relationship manager at a bank",
        "I am preparing a board meeting",
        "I am an accountant",
        "I am a credit analyst",
    ],
}
NOTE = {
    "fr": "Montants en k€. Hypothèses simplificatrices : TVA 20 % sur clients et fournisseurs, IS 25 %.",
    "en": "Amounts in k€. Simplifying assumptions: 20% VAT on receivables and payables, 25% corporate tax.",
}


def k(v: float, lang: str) -> str:
    return fmt_num(v, lang)


def statements_table(c: Company, lang: str, years: list[FinancialYear]) -> str:
    col = 1 if lang == "fr" else 2
    head = "Poste" if lang == "fr" else "Line item"
    out = [
        f"| {head} | " + " | ".join(str(y.year) for y in years) + " |",
        "|---|" + "---:|" * len(years),
    ]
    sections = {
        "cr": ("**Compte de résultat**", "**Income statement**"),
        "actif": ("**Bilan – actif**", "**Balance sheet – assets**"),
        "passif": ("**Bilan – passif**", "**Balance sheet – liabilities**"),
    }
    current = None
    for line in LINES:
        if line[3] != current:
            current = line[3]
            out.append(f"| {sections[current][col - 1]} |" + " |" * len(years))
        out.append(
            f"| {line[col]} | "
            + " | ".join(k(getattr(y, line[0]), lang) for y in years)
            + " |"
        )
    for f, fr, en in FLOWS:
        out.append(
            f"| {fr if lang == 'fr' else en} | "
            + " | ".join(k(getattr(y, f), lang) for y in years)
            + " |"
        )
    return "\n".join(out)


def _intro(c: Company, lang: str, rng: random.Random) -> str:
    p = rng.choice(PERSONAS[lang])
    if lang == "fr":
        return f"{p}. Voici les comptes de {c.name}, entreprise de {c.sector.fr} basée à {c.city} ({c.employees} salariés)."
    return f"{p}. Here are the accounts of {c.name}, a {c.sector.en} company based in {c.city}, France ({c.employees} employees)."


# ---------------------------------------------------------------- exact tasks
def task_ratios(c: Company, lang: str, rng: random.Random) -> dict:
    cur, prev = c.years[-1], c.years[-2]
    r = ratios(cur, prev)
    keys = [
        x
        for x in rng.sample(list(RATIO_LABELS), rng.randint(2, 4))
        if not math.isnan(r[x])
    ]
    names = [RATIO_LABELS[x][0 if lang == "fr" else 1] for x in keys]
    if lang == "fr":
        ask = (
            f"Calcule pour {cur.year} : "
            + ", ".join(names)
            + ". Donne la formule et le résultat."
        )
    else:
        ask = (
            f"Compute for {cur.year}: "
            + ", ".join(names)
            + ". Give the formula and the result."
        )
    user = f"{_intro(c, lang, rng)}\n\n{statements_table(c, lang, [prev, cur])}\n\n{NOTE[lang]}\n\n{ask}"
    lines = []
    for x in keys:
        fr, en, unit, formula = RATIO_LABELS[x]
        label = fr if lang == "fr" else en
        u = {
            "j": " jours" if lang == "fr" else " days",
            "ans": (" an" if abs(r[x]) < 2 else " ans") if lang == "fr" else (" year" if r[x] == 1 else " years"),
            "%": "",
            "": "",
        }[unit]
        val = fmt_num(r[x], lang, "%" if unit == "%" else "") + u
        lines.append(f"- **{label}** = {formula} = **{val}**")
    head = f"Ratios {cur.year} :" if lang == "fr" else f"{cur.year} ratios:"
    target = head + "\n\n" + "\n".join(lines)
    return dict(
        task_id="fin.ratios",
        kind="exact",
        user=user,
        target=target,
        facts={x: r[x] for x in keys},
    )


def task_sig(c: Company, lang: str, rng: random.Random) -> dict:
    cur = c.years[-1]
    steps = [
        ("Chiffre d'affaires", "Revenue", cur.chiffre_affaires),
        ("Marge brute", "Gross margin", cur.marge_brute),
        ("Valeur ajoutée", "Value added", cur.valeur_ajoutee),
        ("Excédent brut d'exploitation (EBE)", "EBITDA", cur.ebe),
        ("Résultat d'exploitation", "Operating income", cur.resultat_exploitation),
        ("Résultat courant avant impôts", "Pre-tax current income", cur.rcai),
        ("Résultat net", "Net income", cur.resultat_net),
    ]
    ask = (
        f"Présente les soldes intermédiaires de gestion {cur.year} sous forme de tableau, avec le taux de chaque solde en % du CA."
        if lang == "fr"
        else f"Present the {cur.year} intermediate management balances (French SIG) as a table, with each as a % of revenue."
    )
    user = f"{_intro(c, lang, rng)}\n\n{statements_table(c, lang, [cur])}\n\n{NOTE[lang]}\n\n{ask}"
    hdr = (
        "| Solde | k€ | % du CA |"
        if lang == "fr"
        else "| Balance | k€ | % of revenue |"
    )
    rows = [hdr, "|---|---:|---:|"]
    facts = {}
    for fr, en, v in steps:
        pct = round(100 * v / cur.chiffre_affaires, 1)
        rows.append(
            f"| {fr if lang == 'fr' else en} | {k(v, lang)} | {fmt_num(pct, lang, '%')} |"
        )
        facts[en] = v
    return dict(
        task_id="fin.statements",
        kind="exact",
        user=user,
        target="\n".join(rows),
        facts=facts,
    )


def task_cash_bridge(c: Company, lang: str, rng: random.Random) -> dict:
    prev, cur = c.years[-2], c.years[-1]
    b = cash_bridge(prev, cur)
    ask = (
        f"Explique la variation de la trésorerie nette entre fin {prev.year} et fin {cur.year} (tableau de flux simplifié)."
        if lang == "fr"
        else f"Explain the change in net cash between end-{prev.year} and end-{cur.year} (simplified cash-flow bridge)."
    )
    user = f"{_intro(c, lang, rng)}\n\n{statements_table(c, lang, [prev, cur])}\n\n{NOTE[lang]}\n\n{ask}"
    sign = lambda v: ("+" if v > 0 else "") + k(v, lang)
    if lang == "fr":
        rows = [
            f"| Capacité d'autofinancement (résultat net + dotations) | {sign(b['caf'])} |",
            f"| Variation du BFR ({k(prev.bfr, lang)} → {k(cur.bfr, lang)}) | {sign(-b['variation_bfr'])} |",
            f"| Investissements | {sign(-b['investissements'])} |",
            f"| Variation des dettes financières | {sign(b['variation_dettes_financieres'])} |",
            f"| Dividendes versés | {sign(-b['dividendes'])} |",
            f"| **Variation de la trésorerie nette** | **{sign(b['variation_tresorerie_nette'])}** |",
        ]
        text = "| Flux (k€) | Montant |\n|---|---:|\n" + "\n".join(rows)
        text += f"\n\nLa trésorerie nette passe de {k(prev.tresorerie_nette, lang)} k€ à {k(cur.tresorerie_nette, lang)} k€."
    else:
        rows = [
            f"| Cash flow from operations before WC (net income + depreciation) | {sign(b['caf'])} |",
            f"| Change in working capital ({k(prev.bfr, lang)} → {k(cur.bfr, lang)}) | {sign(-b['variation_bfr'])} |",
            f"| Capital expenditure | {sign(-b['investissements'])} |",
            f"| Change in financial debt | {sign(b['variation_dettes_financieres'])} |",
            f"| Dividends paid | {sign(-b['dividendes'])} |",
            f"| **Change in net cash** | **{sign(b['variation_tresorerie_nette'])}** |",
        ]
        text = "| Flow (k€) | Amount |\n|---|---:|\n" + "\n".join(rows)
        text += f"\n\nNet cash moves from {k(prev.tresorerie_nette, lang)} k€ to {k(cur.tresorerie_nette, lang)} k€."
    # facts = the SIGNED contributions exactly as displayed (uses of cash are negative)
    facts = {
        "caf": b["caf"],
        "variation_bfr": -b["variation_bfr"],
        "investissements": -b["investissements"],
        "variation_dettes_financieres": b["variation_dettes_financieres"],
        "dividendes": -b["dividendes"],
        "variation_tresorerie_nette": b["variation_tresorerie_nette"],
    }
    return dict(task_id="fin.statements", kind="exact", user=user, target=text, facts=facts)


# ---------------------------------------------------------------- teacher tasks
def task_analysis(c: Company, lang: str, rng: random.Random) -> dict:
    """Full diagnosis: prose written by the teacher, checked against computed facts."""
    years = c.years
    r = ratios(years[-1], years[-2])
    ask = rng.choice(
        [
            "Fais-moi une analyse financière synthétique : points forts, points de vigilance et recommandations.",
            "Rédige une note d'analyse pour le comité de crédit, en une page maximum.",
            "Quelle est la santé financière de cette entreprise ? Justifie avec des chiffres.",
        ]
        if lang == "fr"
        else [
            "Give me a concise financial analysis: strengths, concerns and recommendations.",
            "Write a one-page note for the credit committee.",
            "How financially healthy is this company? Support your view with figures.",
        ]
    )
    user = f"{_intro(c, lang, rng)}\n\n{statements_table(c, lang, years)}\n\n{NOTE[lang]}\n\n{ask}"
    facts = {
        key: r[key]
        for key in ("taux_ebe", "marge_nette", "bfr_jours_ca", "croissance_ca")
        if not math.isnan(r[key])
    }
    return dict(
        task_id="fin.statements",
        kind="teacher",
        user=user,
        target=None,
        facts=facts,
        extra={"health": c.health, "ratios": r},
    )


def task_results_deck(c: Company, lang: str, rng: random.Random) -> dict:
    cur, prev = c.years[-1], c.years[-2]
    ask = (
        f"Prépare un deck de 6 à 8 slides présentant les résultats {cur.year} au comité de direction."
        if lang == "fr"
        else f"Prepare a 6 to 8 slide deck presenting the {cur.year} results to the executive committee."
    )
    user = f"{_intro(c, lang, rng)}\n\n{statements_table(c, lang, [prev, cur])}\n\n{NOTE[lang]}\n\n{ask}"
    r = ratios(cur, prev)
    return dict(
        task_id="fin.deck",
        kind="teacher",
        user=user,
        target=None,
        facts={"croissance_ca": r["croissance_ca"], "taux_ebe": r["taux_ebe"]},
        extra={"skills": ["bird-deck"], "output": "bird-deck"},
    )


GENERATORS = {
    "ratios": (task_ratios, 0.30),
    "sig": (task_sig, 0.15),
    "cash_bridge": (task_cash_bridge, 0.15),
    "analysis": (task_analysis, 0.25),
    "results_deck": (task_results_deck, 0.15),
}


def generate(
    n: int, seed: int = 0, fr_share: float = 0.6, company_offset: int = 0
) -> list[dict]:
    """n records; company seeds are company_offset + i so train/bench splits can use disjoint ranges."""
    rng = random.Random(seed)
    names, weights = zip(*[(k_, w) for k_, (_, w) in GENERATORS.items()])
    out = []
    for i in range(n):
        c = generate_company(company_offset + i)
        lang = "fr" if rng.random() < fr_share else "en"
        gname = rng.choices(names, weights)[0]
        rec = GENERATORS[gname][0](c, lang, rng)
        out.append(
            {
                "id": f"{c.id}-{gname}-{lang}",
                "task_id": rec["task_id"],
                "lang": lang,
                "source": SOURCE,
                "seed": company_offset + i,
                "kind": rec["kind"],
                "company": {
                    "id": c.id,
                    "name": c.name,
                    "sector": c.sector.key,
                    "health": c.health,
                },
                "messages": [{"role": "user", "content": rec["user"]}],
                "target": rec["target"],
                "facts": rec["facts"],
                "checks": ["numbers"]
                + (["bird_deck_valid"] if gname == "results_deck" else []),
                "extra": rec.get("extra", {}),
            }
        )
    return out
