"""Exercices de finance d'entreprise en français, avec résultat exact et trace de raisonnement.

Chaque exemple est en mode raisonnement : `reasoning_content` détaille le calcul, la réponse donne
le résultat mis en forme (format français : 1 234,56 €). Les montants sont tirés au hasard et le
calcul est fait en Python, donc la cible est toujours juste.
"""

from __future__ import annotations

import random


def eur(x: float, dec: int = 2) -> str:
    s = f"{x:,.{dec}f}".replace(",", " ").replace(".", ",")
    return f"{s} €"


def pct(x: float, dec: int = 2) -> str:
    s = f"{x:.{dec}f}".rstrip("0").rstrip(".") if dec else f"{x:.0f}"
    return s.replace(".", ",") + " %"


def _tva_ht_ttc(rng):
    ht = round(rng.uniform(80, 25000), 2)
    t = rng.choice([20, 20, 20, 10, 5.5])
    tva = round(ht * t / 100, 2)
    ttc = round(ht + tva, 2)
    q = rng.choice(
        [
            f"Une facture fait {eur(ht)} HT avec une TVA à {pct(t)}. Quel est le montant TTC ?",
            f"Calcule le TTC pour {eur(ht)} hors taxes (TVA {pct(t)}).",
            f"Mon devis est à {eur(ht)} HT. Combien ça fait TTC avec {pct(t)} de TVA ?",
        ]
    )
    r = (
        f"Montant HT = {eur(ht)}, taux de TVA = {pct(t)}.\nTVA = HT × taux = {eur(ht)} × {pct(t)} = {eur(tva)}.\n"
        f"TTC = HT + TVA = {eur(ht)} + {eur(tva)} = {eur(ttc)}.\nVérification : {eur(ttc)} / {1 + t / 100:.3f} ≈ {eur(ttc / (1 + t / 100))}. OK."
    )
    a = (
        f"**Montant TTC : {eur(ttc)}**\n\n- Montant HT : {eur(ht)}\n- TVA ({pct(t)}) : {eur(tva)}\n"
        f"- TTC = {eur(ht)} + {eur(tva)} = {eur(ttc)}"
    )
    return q, r, a


def _tva_ttc_ht(rng):
    ttc = round(rng.uniform(100, 30000), 2)
    t = rng.choice([20, 20, 10, 5.5])
    ht = round(ttc / (1 + t / 100), 2)
    tva = round(ttc - ht, 2)
    q = rng.choice(
        [
            f"J'ai payé {eur(ttc)} TTC (TVA {pct(t)}). Quel est le montant HT et la TVA récupérable ?",
            f"Retrouve le HT à partir de {eur(ttc)} TTC, TVA à {pct(t)}.",
        ]
    )
    r = (
        f"On divise le TTC par (1 + taux) : {eur(ttc)} / {1 + t / 100:.3f} = {eur(ht)} HT.\n"
        f"TVA = TTC − HT = {eur(ttc)} − {eur(ht)} = {eur(tva)}.\nAttention à ne pas faire TTC × (1 − taux), ce qui serait faux."
    )
    a = (
        f"- **Montant HT : {eur(ht)}**\n- **TVA récupérable : {eur(tva)}**\n\n"
        f"Calcul : HT = {eur(ttc)} / {1 + t / 100:.3f} = {eur(ht)} ; TVA = {eur(ttc)} − {eur(ht)} = {eur(tva)}."
    )
    return q, r, a


def _marge(rng):
    pa = round(rng.uniform(5, 2000), 2)
    pv = round(pa * rng.uniform(1.1, 2.6), 2)
    m = round(pv - pa, 2)
    tmarge, tmarque = m / pa * 100, m / pv * 100
    q = rng.choice(
        [
            f"J'achète un produit {eur(pa)} HT et je le revends {eur(pv)} HT. Quelle est ma marge, mon taux de marge et mon taux de marque ?",
            f"Prix d'achat HT : {eur(pa)}, prix de vente HT : {eur(pv)}. Calcule la marge commerciale et les taux.",
        ]
    )
    r = (
        f"Marge = PV HT − PA HT = {eur(pv)} − {eur(pa)} = {eur(m)}.\n"
        f"Taux de marge = marge / PA = {eur(m)} / {eur(pa)} = {pct(tmarge)}.\n"
        f"Taux de marque = marge / PV = {eur(m)} / {eur(pv)} = {pct(tmarque)}.\n"
        "Le taux de marque est toujours inférieur au taux de marge (même marge, base plus grande)."
    )
    a = (
        f"| Indicateur | Calcul | Résultat |\n|---|---|---|\n| Marge commerciale | {eur(pv)} − {eur(pa)} | **{eur(m)}** |\n"
        f"| Taux de marge | marge / prix d'achat | **{pct(tmarge)}** |\n| Taux de marque | marge / prix de vente | **{pct(tmarque)}** |"
    )
    return q, r, a


def _seuil(rng):
    ca = round(rng.uniform(150_000, 5_000_000), -3)
    cv = round(ca * rng.uniform(0.35, 0.75), -3)
    cf = round((ca - cv) * rng.uniform(0.4, 1.3), -3)
    mscv = ca - cv
    tmscv = mscv / ca
    sr = cf / tmscv
    jours = sr / ca * 360
    q = (
        f"Mon entreprise fait {eur(ca, 0)} de chiffre d'affaires, avec {eur(cv, 0)} de charges variables et "
        f"{eur(cf, 0)} de charges fixes. Quel est mon seuil de rentabilité et mon point mort ?"
    )
    r = (
        f"MSCV = CA − CV = {eur(ca, 0)} − {eur(cv, 0)} = {eur(mscv, 0)}.\nTaux de MSCV = {eur(mscv, 0)} / {eur(ca, 0)} = {pct(tmscv * 100)}.\n"
        f"SR = CF / taux de MSCV = {eur(cf, 0)} / {tmscv:.4f} = {eur(sr)}.\n"
        + (f"Point mort (année de 360 jours) = SR / CA × 360 = {jours:.1f} jours.\n" if sr < ca
           else "SR > CA : le point mort n'est pas atteint dans l'année ; ")
        + (
            "SR < CA : l'entreprise est bénéficiaire."
            if sr < ca
            else "l'entreprise est déficitaire sur ce niveau d'activité."
        )
    )
    resultat = mscv - cf
    a = (
        f"- Marge sur coûts variables : {eur(mscv, 0)} (taux : {pct(tmscv * 100)})\n- **Seuil de rentabilité : {eur(sr)}**\n"
        + (f"- **Point mort : environ {round(jours)} jours** d'activité (base 360 jours)\n" if sr < ca
           else "- **Point mort : non atteint dans l'année** au niveau d'activité actuel\n")
        + f"- Résultat actuel : {eur(resultat, 0)}\n\n"
        + (
            "Vous dépassez le seuil : chaque euro de CA supplémentaire rapporte "
            f"{tmscv:.2f} € de marge.".replace(".", ",", 1)
            if sr < ca
            else f"Vous êtes sous le seuil : il manque {eur(sr - ca)} de CA pour atteindre l'équilibre."
        )
    )
    return q, r, a


def _amortissement(rng):
    v = round(rng.uniform(3000, 250000), -2)
    n = rng.choice([3, 4, 5, 5, 8, 10])
    ann = v / n
    q = f"Nous achetons une machine {eur(v, 0)} HT, amortie en linéaire sur {n} ans (mise en service le 1er janvier). Donne le plan d'amortissement."
    r = f"Annuité = valeur / durée = {eur(v, 0)} / {n} = {eur(ann)}. Taux = 100 / {n} = {pct(100 / n)}. Mise en service au 1er janvier : pas de prorata."
    rows, cum = [], 0.0
    for k in range(1, n + 1):
        dot = round(ann, 2) if k < n else round(v - cum, 2)  # dernière annuité ajustée (arrondis)
        cum = round(cum + dot, 2)
        rows.append(f"| {k} | {eur(dot)} | {eur(cum)} | {eur(v - cum)} |")
    a = (
        f"Annuité constante : **{eur(ann)}** par an (taux {pct(100 / n)}).\n\n| Année | Dotation | Cumul | Valeur nette comptable |\n"
        "|---|---|---|---|\n" + "\n".join(rows)
    )
    return q, r, a


def _emprunt(rng):
    c = round(rng.uniform(10_000, 500_000), -3)
    t = round(rng.uniform(1.5, 6.5), 2)
    ans = rng.choice([3, 5, 7, 10, 15, 20])
    i, n = t / 100 / 12, ans * 12
    m = c * i / (1 - (1 + i) ** -n)
    m = round(m, 2)  # mensualité arrondie au centime, comme sur un tableau d'amortissement
    total = m * n - c
    q = rng.choice(
        [
            f"Quelle sera la mensualité d'un prêt de {eur(c, 0)} sur {ans} ans à {pct(t)} par an ?",
            f"On emprunte {eur(c, 0)} à {pct(t)} sur {ans} ans, remboursement mensuel constant. Mensualité et coût total des intérêts ?",
        ]
    )
    r = (
        f"Taux mensuel i = {pct(t)} / 12 = {i:.6f}. Nombre de mensualités n = {n}.\nM = C × i / (1 − (1 + i)^−n) = {c:.0f} × {i:.6f} / (1 − {1 + i:.6f}^−{n}) = {eur(m)}.\n"
        f"Coût des intérêts = M × n − C = {eur(m)} × {n} − {eur(c, 0)} = {eur(total)}. Hors assurance et frais de dossier."
    )
    a = (
        f"- **Mensualité : {eur(m)}** ({n} mensualités)\n- **Coût total des intérêts : {eur(total)}**\n- Total remboursé : {eur(m * n)}\n\n"
        "Formule : M = C × i / (1 − (1 + i)^−n), avec i le taux mensuel. Montants hors assurance emprunteur et frais."
    )
    return q, r, a


def _variation(rng):
    a0 = round(rng.uniform(50_000, 9_000_000), -3)
    a1 = round(a0 * rng.uniform(0.7, 1.45), -3)
    v = (a1 - a0) / a0 * 100
    y = rng.randint(2019, 2025)
    q = f"Notre CA est passé de {eur(a0, 0)} en {y} à {eur(a1, 0)} en {y + 1}. Quelle est l'évolution en pourcentage ?"
    r = f"Variation = (V1 − V0) / V0 × 100 = ({a1:.0f} − {a0:.0f}) / {a0:.0f} × 100 = {pct(v, 4)}. Écart absolu = {a1 - a0:.0f} €."
    sens = "hausse" if v >= 0 else "baisse"
    a = f"Le chiffre d'affaires est en **{sens} de {pct(abs(v))}** ({'+' if v >= 0 else '−'}{eur(abs(a1 - a0), 0)}) entre {y} et {y + 1}."
    return q, r, a


def _bfr(rng):
    ca = round(rng.uniform(500_000, 20_000_000), -3)
    st, cl, fo = (
        round(ca * rng.uniform(lo, hi), -3)
        for lo, hi in ((0.03, 0.2), (0.08, 0.25), (0.05, 0.2))
    )
    bfr = st + cl - fo
    j = bfr / ca * 360
    q = (
        f"Stocks : {eur(st, 0)}, créances clients : {eur(cl, 0)}, dettes fournisseurs : {eur(fo, 0)}. "
        f"CA annuel HT : {eur(ca, 0)}. Calcule le BFR et exprime-le en jours de CA."
    )
    r = f"BFR d'exploitation = stocks + créances clients − dettes fournisseurs = {st:.0f} + {cl:.0f} − {fo:.0f} = {bfr:.0f}.\nEn jours de CA = BFR / CA × 360 = {j:.1f} jours."
    a = (
        f"- **BFR : {eur(bfr, 0)}**\n- **Soit {round(j)} jours de chiffre d'affaires** (base 360 jours)\n\n"
        + (
            "Un BFR positif doit être financé (trésorerie, fonds de roulement, découvert)."
            if bfr > 0
            else "Un BFR négatif signifie que l'exploitation génère de la trésorerie (fournisseurs payés après encaissement des clients)."
        )
    )
    return q, r, a


TEMPLATES = [
    _tva_ht_ttc,
    _tva_ttc_ht,
    _marge,
    _seuil,
    _amortissement,
    _emprunt,
    _variation,
    _bfr,
]


def generate(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        q, r, a = TEMPLATES[i % len(TEMPLATES)](rng)
        out.append(
            {
                "id": f"finance_fr-{i}",
                "source": "finance_fr",
                "lang": "fr",
                "thinking": True,
                "reasoning_effort": "low",  # traces courtes : ne pas les associer à l'effort "xhigh" par défaut
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a, "reasoning_content": r},
                ],
            }
        )
    return out
