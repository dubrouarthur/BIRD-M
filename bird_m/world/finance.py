"""BIRD-World financial generator: fictional companies with COHERENT multi-year statements.

Everything is computed, never written by an LLM, so every number in a derived task
has an exact ground truth. Amounts are in k€ (thousands of euros), rounded to 1 k€.

Invariants (tested):
- balance sheet: total actif == total passif, every year
- SIG chain: marge -> valeur ajoutée -> EBE -> résultat d'exploitation -> RCAI -> résultat net
- cash reconciliation: Δtrésorerie nette == CAF - ΔBFR - investissements + Δdettes financières - dividendes
Simplifications (stated in the generated tasks): French GAAP-like layout, VAT 20 % on receivables
and payables, corporate tax 25 % flat on positive profit, no deferred items.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

VAT = 1.20
TAX_RATE = 0.25


@dataclass(frozen=True)
class SectorProfile:
    key: str
    fr: str
    en: str
    gross_margin: tuple[float, float]  # (CA - achats consommés) / CA
    external: tuple[float, float]  # autres achats et charges externes / CA
    personnel: tuple[float, float]  # charges de personnel / CA
    dso: tuple[int, int]  # délai clients (jours)
    dpo: tuple[int, int]  # délai fournisseurs (jours)
    dio: tuple[int, int]  # rotation des stocks (jours d'achats)
    capex: tuple[float, float]  # investissements / CA
    asset_intensity: tuple[float, float]  # immobilisations nettes initiales / CA


SECTORS: tuple[SectorProfile, ...] = (
    SectorProfile(
        "retail",
        "commerce de détail",
        "retail",
        (0.35, 0.50),
        (0.12, 0.18),
        (0.12, 0.18),
        (2, 10),
        (35, 60),
        (45, 90),
        (0.02, 0.04),
        (0.10, 0.25),
    ),
    SectorProfile(
        "industry",
        "industrie manufacturière",
        "manufacturing",
        (0.40, 0.55),
        (0.10, 0.16),
        (0.20, 0.30),
        (45, 75),
        (45, 70),
        (40, 90),
        (0.04, 0.08),
        (0.35, 0.60),
    ),
    SectorProfile(
        "construction",
        "BTP",
        "construction",
        (0.45, 0.60),
        (0.18, 0.28),
        (0.20, 0.28),
        (60, 90),
        (50, 75),
        (5, 20),
        (0.02, 0.05),
        (0.15, 0.30),
    ),
    SectorProfile(
        "services",
        "services aux entreprises",
        "business services",
        (0.85, 0.97),
        (0.15, 0.25),
        (0.45, 0.60),
        (45, 75),
        (30, 50),
        (0, 3),
        (0.01, 0.03),
        (0.05, 0.15),
    ),
    SectorProfile(
        "software",
        "édition de logiciels",
        "software",
        (0.85, 0.95),
        (0.15, 0.22),
        (0.45, 0.58),
        (40, 70),
        (30, 45),
        (0, 2),
        (0.03, 0.06),
        (0.10, 0.25),
    ),
    SectorProfile(
        "hospitality",
        "hôtellerie-restauration",
        "hospitality",
        (0.65, 0.75),
        (0.18, 0.25),
        (0.30, 0.38),
        (2, 10),
        (25, 45),
        (5, 15),
        (0.04, 0.08),
        (0.50, 0.90),
    ),
    SectorProfile(
        "wholesale",
        "commerce de gros",
        "wholesale",
        (0.18, 0.30),
        (0.05, 0.09),
        (0.06, 0.11),
        (40, 65),
        (40, 60),
        (30, 60),
        (0.01, 0.02),
        (0.05, 0.12),
    ),
    SectorProfile(
        "transport",
        "transport et logistique",
        "transport & logistics",
        (0.70, 0.82),
        (0.25, 0.35),
        (0.28, 0.36),
        (45, 70),
        (35, 55),
        (2, 8),
        (0.05, 0.09),
        (0.40, 0.70),
    ),
    SectorProfile(
        "health",
        "santé et médico-social",
        "healthcare",
        (0.75, 0.88),
        (0.15, 0.22),
        (0.45, 0.55),
        (20, 45),
        (30, 50),
        (5, 15),
        (0.03, 0.06),
        (0.30, 0.60),
    ),
    SectorProfile(
        "agrifood",
        "agroalimentaire",
        "food processing",
        (0.30, 0.42),
        (0.10, 0.15),
        (0.12, 0.18),
        (35, 60),
        (40, 60),
        (20, 45),
        (0.03, 0.06),
        (0.30, 0.50),
    ),
)

_SYL = (
    "lu",
    "mi",
    "ver",
    "ta",
    "no",
    "ra",
    "pel",
    "cor",
    "dia",
    "sen",
    "va",
    "bri",
    "lo",
    "tex",
    "ka",
    "ro",
    "fen",
    "zo",
    "pla",
    "qui",
)
_FORMS = ("SAS", "SARL", "SA")


def company_name(rng: random.Random) -> str:
    """Pronounceable invented name (no dictionary words) to avoid matching real companies."""
    n = rng.choice((2, 3))
    base = "".join(rng.choice(_SYL) for _ in range(n)).capitalize()
    return f"{base} {rng.choice(_FORMS)}"


@dataclass
class FinancialYear:
    year: int
    # compte de résultat (k€)
    chiffre_affaires: int
    achats_consommes: int
    charges_externes: int
    impots_taxes: int
    charges_personnel: int
    dotations_amortissements: int
    resultat_financier: int
    resultat_exceptionnel: int
    impot_societes: int
    # bilan actif
    immobilisations_nettes: int
    stocks: int
    creances_clients: int
    tresorerie: int
    # bilan passif
    capital: int
    reserves: int
    resultat_net: int
    dettes_financieres: int
    concours_bancaires: int
    dettes_fournisseurs: int
    dettes_fiscales_sociales: int
    # flux de l'exercice
    investissements: int
    dividendes: int
    nouveaux_emprunts: int
    remboursements: int

    # --- SIG -------------------------------------------------------------
    @property
    def marge_brute(self) -> int:
        return self.chiffre_affaires - self.achats_consommes

    @property
    def valeur_ajoutee(self) -> int:
        return self.marge_brute - self.charges_externes

    @property
    def ebe(self) -> int:
        return self.valeur_ajoutee - self.impots_taxes - self.charges_personnel

    @property
    def resultat_exploitation(self) -> int:
        return self.ebe - self.dotations_amortissements

    @property
    def rcai(self) -> int:
        return self.resultat_exploitation + self.resultat_financier

    @property
    def caf(self) -> int:
        """Capacité d'autofinancement (méthode additive simplifiée)."""
        return self.resultat_net + self.dotations_amortissements

    # --- bilan -----------------------------------------------------------
    @property
    def capitaux_propres(self) -> int:
        return self.capital + self.reserves + self.resultat_net

    @property
    def total_actif(self) -> int:
        return (
            self.immobilisations_nettes
            + self.stocks
            + self.creances_clients
            + self.tresorerie
        )

    @property
    def total_passif(self) -> int:
        return (
            self.capitaux_propres
            + self.dettes_financieres
            + self.concours_bancaires
            + self.dettes_fournisseurs
            + self.dettes_fiscales_sociales
        )

    @property
    def bfr(self) -> int:
        return (
            self.stocks
            + self.creances_clients
            - self.dettes_fournisseurs
            - self.dettes_fiscales_sociales
        )

    @property
    def tresorerie_nette(self) -> int:
        return self.tresorerie - self.concours_bancaires

    @property
    def dette_nette(self) -> int:
        return self.dettes_financieres - self.tresorerie_nette

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in (
            "marge_brute",
            "valeur_ajoutee",
            "ebe",
            "resultat_exploitation",
            "rcai",
            "caf",
            "capitaux_propres",
            "total_actif",
            "total_passif",
            "bfr",
            "tresorerie_nette",
            "dette_nette",
        ):
            d[k] = getattr(self, k)
        return d


@dataclass
class Company:
    id: str
    name: str
    sector: SectorProfile
    city: str
    employees: int
    years: list[FinancialYear] = field(default_factory=list)
    health: str = "saine"  # saine | fragile | difficulte (ground truth for diagnosis tasks)

    def year(self, y: int) -> FinancialYear:
        return next(fy for fy in self.years if fy.year == y)


CITIES = (
    "Lyon",
    "Nantes",
    "Lille",
    "Bordeaux",
    "Toulouse",
    "Rennes",
    "Strasbourg",
    "Grenoble",
    "Montpellier",
    "Dijon",
    "Rouen",
    "Clermont-Ferrand",
)


def _u(rng: random.Random, lo_hi: tuple[float, float]) -> float:
    return rng.uniform(*lo_hi)


def generate_company(seed: int, first_year: int = 2023, n_years: int = 3) -> Company:
    rng = random.Random(seed)
    sec = rng.choice(SECTORS)
    ca = rng.choice((800, 1_500, 3_000, 6_000, 12_000, 25_000, 50_000)) * rng.uniform(
        0.8, 1.25
    )
    salary_cost = rng.uniform(55, 75)  # k€ chargé par salarié
    pers_ratio = _u(rng, sec.personnel)
    # financial health drives the EBE margin; personnel absorbs the rest of the value added
    health = rng.choices(("saine", "fragile", "difficulte"), (0.7, 0.2, 0.1))[0]
    ebe_target = {"saine": (0.07, 0.20), "fragile": (0.02, 0.07), "difficulte": (-0.06, 0.02)}[health]
    ebe_rate = rng.uniform(*ebe_target)
    comp = Company(
        id=f"bw-{seed:06d}",
        name=company_name(rng),
        sector=sec,
        city=rng.choice(CITIES),
        employees=0,  # set from year-1 personnel cost below
    )

    # opening position (end of year first_year - 1)
    immo = ca * _u(rng, sec.asset_intensity)
    capital = round(max(50, ca * rng.uniform(0.03, 0.10)))
    reserves = immo * rng.uniform(0.4, 1.0) + ca * rng.uniform(0.03, 0.10)
    debt = immo * rng.uniform(0.2, 0.7)
    growth_trend = rng.uniform(-0.05, 0.15)
    gm, ext, dso, dpo, dio = (
        _u(rng, sec.gross_margin),
        _u(rng, sec.external),
        _u(rng, sec.dso),
        _u(rng, sec.dpo),
        _u(rng, sec.dio),
    )
    prev_rn = 0

    for i in range(n_years):
        y = first_year + i
        if i > 0:
            ca *= 1 + growth_trend + rng.uniform(-0.06, 0.06)
        gm_y = min(0.98, max(0.05, gm + rng.uniform(-0.03, 0.03)))
        achats = ca * (1 - gm_y)
        charges_ext = ca * (ext + rng.uniform(-0.015, 0.015))
        impots_taxes = ca * rng.uniform(0.005, 0.015)
        ebe_y = ca * (ebe_rate + rng.uniform(-0.015, 0.015))
        personnel = max(ca * 0.04, ca - achats - charges_ext - impots_taxes - ebe_y)
        dotations = immo * rng.uniform(0.08, 0.14)
        interest_rate = rng.uniform(0.03, 0.055)
        res_fin = -debt * interest_rate
        res_exc = ca * rng.choice((0, 0, 0, rng.uniform(-0.01, 0.01)))

        r = {
            k: round(v)
            for k, v in dict(
                ca=ca,
                achats=achats,
                ext=charges_ext,
                it=impots_taxes,
                pers=personnel,
                dot=dotations,
                rf=res_fin,
                rex=res_exc,
            ).items()
        }
        rcai = (
            r["ca"] - r["achats"] - r["ext"] - r["it"] - r["pers"] - r["dot"] + r["rf"]
        )
        is_ = round(max(0, rcai + r["rex"]) * TAX_RATE)
        rn = rcai + r["rex"] - is_

        capex = round(max(ca * _u(rng, sec.capex) * 0.5, dotations * rng.uniform(0.7, 1.4)))
        immo_end = round(immo + capex - r["dot"])
        new_loan = round(capex * rng.choice((0, 0, 0.5, 0.8)))
        repay = round(min(debt, debt * rng.uniform(0.10, 0.20)))
        debt_end = round(debt + new_loan - repay)
        # dividends paid in year N come from the N-1 profit
        dividends = round(max(0, prev_rn) * rng.choice((0, 0, 0.3, 0.5))) if i > 0 else 0

        stocks = round(r["achats"] * dio / 365)
        clients = round(r["ca"] * VAT * dso / 365)
        fournisseurs = round((r["achats"] + r["ext"]) * VAT * dpo / 365)
        fisc_soc = round(r["pers"] * rng.uniform(0.08, 0.12) + is_ * 0.3)

        reserves_y = round(reserves)
        passif_hors_tres = (
            (capital + reserves_y + rn) + debt_end + fournisseurs + fisc_soc
        )
        plug = passif_hors_tres - (immo_end + stocks + clients)
        tres, concours = (plug, 0) if plug >= 0 else (0, -plug)

        fy = FinancialYear(
            year=y,
            chiffre_affaires=r["ca"],
            achats_consommes=r["achats"],
            charges_externes=r["ext"],
            impots_taxes=r["it"],
            charges_personnel=r["pers"],
            dotations_amortissements=r["dot"],
            resultat_financier=r["rf"],
            resultat_exceptionnel=r["rex"],
            impot_societes=is_,
            immobilisations_nettes=immo_end,
            stocks=stocks,
            creances_clients=clients,
            tresorerie=tres,
            capital=capital,
            reserves=reserves_y,
            resultat_net=rn,
            dettes_financieres=debt_end,
            concours_bancaires=concours,
            dettes_fournisseurs=fournisseurs,
            dettes_fiscales_sociales=fisc_soc,
            investissements=capex,
            dividendes=dividends,
            nouveaux_emprunts=new_loan,
            remboursements=repay,
        )
        comp.years.append(fy)
        # roll forward
        immo, debt = immo_end, debt_end
        reserves = reserves_y + rn  # provisional; _apply_dividends rebuilds exact reserves
        prev_rn = rn
    _apply_dividends(comp)
    comp.employees = max(2, round(comp.years[0].charges_personnel / salary_cost))
    comp.health = health
    return comp


def _apply_dividends(comp: Company) -> None:
    """Dividends decided in year N are deducted from the reserves carried into year N (paid from N-1 profit).

    Rebuild reserves so that CP_N = CP_{N-1} + RN_N - Div_N exactly, then re-balance cash.
    """
    for i in range(1, len(comp.years)):
        prev, cur = comp.years[i - 1], comp.years[i]
        cur.reserves = prev.reserves + prev.resultat_net - cur.dividendes
        plug = (
            cur.total_passif
            - cur.concours_bancaires
            - (cur.immobilisations_nettes + cur.stocks + cur.creances_clients)
        )
        cur.tresorerie, cur.concours_bancaires = (plug, 0) if plug >= 0 else (0, -plug)


# ----------------------------------------------------------------- analysis (ground truth)
def ratios(fy: FinancialYear, prev: FinancialYear | None = None) -> dict[str, float]:
    """Standard French analysis ratios. Percentages as %, delays in days. Rounded to 1 decimal."""
    ca = fy.chiffre_affaires
    out = {
        "taux_marge_brute": 100 * fy.marge_brute / ca,
        "taux_valeur_ajoutee": 100 * fy.valeur_ajoutee / ca,
        "taux_ebe": 100 * fy.ebe / ca,
        "taux_resultat_exploitation": 100 * fy.resultat_exploitation / ca,
        "marge_nette": 100 * fy.resultat_net / ca,
        "rentabilite_capitaux_propres": 100 * fy.resultat_net / fy.capitaux_propres
        if fy.capitaux_propres > 0
        else float("nan"),
        "gearing": 100 * fy.dette_nette / fy.capitaux_propres
        if fy.capitaux_propres > 0
        else float("nan"),
        "bfr_jours_ca": 365 * fy.bfr / ca,
        "delai_clients_jours": 365 * fy.creances_clients / (ca * VAT),
        "delai_fournisseurs_jours": 365
        * fy.dettes_fournisseurs
        / ((fy.achats_consommes + fy.charges_externes) * VAT),
        "rotation_stocks_jours": 365 * fy.stocks / fy.achats_consommes
        if fy.achats_consommes
        else 0.0,
        "capacite_remboursement_annees": fy.dettes_financieres / fy.caf
        if fy.caf > 0
        else float("nan"),
        "liquidite_generale": (fy.stocks + fy.creances_clients + fy.tresorerie)
        / (
            fy.dettes_fournisseurs + fy.dettes_fiscales_sociales + fy.concours_bancaires
        ),
    }
    if prev is not None:
        out["croissance_ca"] = 100 * (ca / prev.chiffre_affaires - 1)
    return {k: round(v, 1) for k, v in out.items()}


def cash_bridge(prev: FinancialYear, cur: FinancialYear) -> dict[str, int]:
    """Explains the change in net cash between two year-ends (tableau de flux simplifié)."""
    return {
        "caf": cur.caf,
        "variation_bfr": cur.bfr - prev.bfr,
        "investissements": cur.investissements,
        "variation_dettes_financieres": cur.dettes_financieres
        - prev.dettes_financieres,
        "dividendes": cur.dividendes,
        "variation_tresorerie_nette": cur.tresorerie_nette - prev.tresorerie_nette,
    }
