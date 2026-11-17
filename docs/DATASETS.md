# Datasets BIRD-M

Dépôts (privés) : entraînement → [`adubrou/bird-m-sft`](https://huggingface.co/datasets/adubrou/bird-m-sft) (une _config_ par dataset ci-dessous) · évaluation → [`adubrou/bird-bench`](https://huggingface.co/datasets/adubrou/bird-bench) (jamais mélangé à l'entraînement).

## Principes

1. **Les réponses cibles viennent de sources sous licence propre.** On auto-distille Qwen3.8-27B en mode thinking xhigh, avec best-of-n et un juge. On n'utilise jamais de sorties d'API fermées, dont les conditions interdisent d'entraîner un modèle concurrent. Les datasets FR existants sur le Hub (Alpaca traduits, souvent générés par GPT) servent **au mieux de graines de prompts**, jamais de cibles.
   L'assistant de développement (Claude) écrit **les générateurs, les vérificateurs et les skills**, mais pas les réponses cibles : les conditions d'Anthropic interdisent d'utiliser ses sorties pour entraîner un modèle concurrent.
2. **Tout ce qui est calculable est calculé.** Finance, Excel, JSON, routage et correction ont une vérité terrain exacte produite par du code. Le juge LLM ne sert que pour le style et la pertinence.
3. **Un univers cohérent plutôt que des prompts isolés** : voir BIRD-World ci-dessous.
4. **Aucune donnée personnelle réelle.** Les personnes, entreprises et IBAN sont fictifs, et les formats sont validés pour ne pas tomber sur un vrai identifiant.
5. **Traçabilité** : chaque exemple porte `task_id`, `lang`, `source`, `generator`, `seed`, `judge_score` et `skills`. On peut ainsi rejouer, filtrer et retirer n'importe quel sous-ensemble.

## La pièce maîtresse : BIRD-World (entreprises fictives)

On ne génère pas « un e-mail de relance » dans le vide. On génère **≈ 500 entreprises fictives complètes**, puis tous les documents en découlent :

```
Entreprise fictive (seed)
├─ identité : nom, secteur (12), taille (TPE → ETI), ville, ton, valeurs
├─ organigramme : 15–60 personnes (rôles, managers, style d'écriture)
├─ produits / services, clients, fournisseurs
├─ finances : 3 exercices COHÉRENTS (bilan équilibré, CR, flux, budget) ← générés par code
├─ skills : 5–15 SKILL.md propres (charte, procédures, modèles)       ← alimente BSP
└─ documents : e-mails, fils, CR, contrats, factures, appels d'offres, decks
```

Pourquoi c'est plus performant :

- **Tâches multi-documents réalistes** : « Prépare le CR à partir de ce fil et du planning » ; « Fais le deck de résultats à partir du bilan ».
- **Skills crédibles** : chaque entreprise a sa propre charte et ses procédures. Elles se contredisent d'une entreprise à l'autre, ce qui force le modèle à lire la skill plutôt que mémoriser.
- **Vérité terrain** : les chiffres viennent du générateur. Une analyse financière fausse est donc détectée automatiquement.
- **Diversité contrôlée** : les distributions de secteur, taille, ton et langue sont pilotées, pas subies.

## Les datasets

| Config       | Contenu                                                                                                                                 | Taille v1                 | Génération                                                                                                           | Vérification                                                                       |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `world`      | 500 entreprises BIRD-World (JSON)                                                                                                       | 500                       | LLM pour les textes + **code** pour les finances et les invariants                                                   | Tests d'invariants (bilan équilibré, organigramme connexe)                         |
| `tasks`      | Les 81 tâches de la taxonomie, en FR (60 %) et EN (40 %)                                                                                | 110 000                   | Matrice de graines (entreprise × rôle × tâche × ton × longueur × difficulté) → teacher best-of-4                     | Juge avec grille par tâche, filtres de langue et de longueur, dédup MinHash        |
| `skills`     | Routage (catalogue de 3 à 40 skills, distracteurs proches, 30 % _none_), application, composition, conflits, politiques, `skill.author` | 30 000                    | Skills BIRD-World + Core Skills, liaison aléatoire (`render_conversation`)                                           | Routage : exact. Application : juge + checklist                                    |
| `deck`       | BIRD-Deck : doc → deck, édition par opérations, données → graphique, critique sur capture                                               | 15 000 SFT + 5 000 paires | Teacher avec la skill `bird-deck`, rendu headless                                                                    | Schéma JSON, densité, débordement mesuré au rendu, juge visuel                     |
| `finance`    | Analyse d'états financiers, ratios, SIG, BFR, budget, DCF, commentaire d'écarts, deck de résultats                                      | 15 000                    | États financiers BIRD-World (code), questions dont la réponse est calculée                                           | Exact sur les chiffres, juge « contrôleur de gestion » sur l'analyse               |
| `excel`      | Formules FR/EN, explication, débogage                                                                                                   | 8 000                     | Feuilles générées + formule cible                                                                                    | **Exécution** de la formule (moteur de calcul headless) contre le résultat attendu |
| `correction` | Correction FR/EN, réécriture de ton, raccourcissement, langage clair                                                                    | 10 000                    | Textes propres sous licence ouverte + **injection d'erreurs réalistes** (a/à, accords, participe passé, typographie) | Exact : diff contre l'original                                                     |
| `tools`      | Appels d'outils au format Qwen, sorties JSON strictes, extraction de documents                                                          | 15 000                    | Outils fictifs (agenda, CRM, ERP) + documents BIRD-World                                                             | Validation de schéma, arguments exacts                                             |
| `trust`      | Abstention (« je ne sais pas »), masquage de PII, injection dans les documents, cohérence de langue                                     | 10 000                    | Cas adverses construits                                                                                              | Exact ou juge                                                                      |
| `replay`     | Raisonnement, code et culture générale, anti-oubli                                                                                      | 30 000                    | Prompts ouverts sous licence permissive → réponses de la base elle-même                                              | Filtre de qualité                                                                  |
| `pref`       | Paires préférée / rejetée pour le DPO                                                                                                   | 30 000                    | Best-of-n jugé, plus défauts injectés (trop long, mauvais ton, anglicismes, chiffres faux)                           | Juge validé contre des annotateurs humains                                         |

**Total SFT ≈ 250 000 exemples**, soit environ 375 millions de tokens, plus 35 000 paires de préférence.

### Datasets réutilisables depuis le Hub (licences vérifiées sur les cartes le 17/11/2026)

Critères : licence commerciale (Apache 2.0, MIT, CC-BY-4.0, CC0, ODC-BY, Licence Ouverte Etalab) **et** provenance propre, c'est-à-dire des réponses produites par des humains ou par des modèles ouverts. Un dataset « Apache » dont les réponses viennent de GPT-4, Claude ou Gemini est **exclu** comme cible d'entraînement.

| Dataset | Licence | Taille | Provenance | Usage BIRD-M | Volume pris |
|---|---|---|---|---|---|
| `HuggingFaceTB/smoltalk2` (sous-ensembles Qwen3, dont multilingue FR) | Apache 2.0 (nouveaux sous-ensembles) | ~245 000 multilingues | Qwen3-32B ✅ | `replay` FR/EN, suivi d'instructions | ~20 000 |
| `nvidia/Nemotron-Post-Training-Dataset-v2` (accès à demander) | CC-BY-4.0 | millions, dont FR | DeepSeek-R1 et Qwen ✅ | `replay` FR : raisonnement, chat | ~10 000 |
| `nvidia/Nemotron-Post-Training-Dataset-v1` | CC-BY-4.0 (usage commercial) | 10 M+ | DeepSeek-R1, Qwen3 ✅ | `tools` et `replay` | ~10 000 |
| `CohereForAI/aya_dataset` | Apache 2.0 | ~200 000 | **humains**, dont FR ✅ | `tasks` FR (style naturel) | ~5 000 FR |
| `OpenAssistant/oasst2` | Apache 2.0 | ~100 000 | **humains**, dont FR ✅ | `tasks` et `pref` | ~5 000 |
| `nvidia/HelpSteer3` | CC-BY-4.0 | ~40 000 | annotations humaines, modèles ouverts, multilingue ✅ | `pref` (DPO), calibration du juge | totalité |
| `nvidia/HelpSteer2` | CC-BY-4.0 | ~20 000 | annotations humaines ✅ | `pref` | totalité |
| `Salesforce/xlam-function-calling-60k` (accès à demander) | CC-BY-4.0 | 60 000 | DeepSeek-V2, Mixtral, appels vérifiés ✅ | `tools` | ~15 000 |
| `nvidia/When2Call` | CC-BY-4.0 | petit | à confirmer | `tools` : quand appeler ou refuser | totalité |
| `ibm-research/finqa` | CC-BY-4.0 | ~8 000 | **humains** (rapports financiers) ✅ | `finance` : calculs sur états | totalité |
| `next-tat/TAT-QA` | CC-BY-4.0 | ~16 000 | **humains** (tableaux et texte) ✅ | `finance` : questions sur tableaux | totalité |
| `gretelai/synthetic_text_to_sql` | Apache 2.0 | ~100 000 | Gretel Navigator (mélange de modèles non publié) ⚠️ | `sql` : prompts et schémas, réponses à re-vérifier | ~8 000 |
| `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | CC-BY-4.0 | ~30 000 | Mistral et Mixtral ✅ | `trust` | ~3 000 |
| `Benitoow/OfficeSmith-PPTX-IR` | Apache 2.0 | 958 | Qwen3.5-397B ✅ | `deck` : graines brief → slides, converties en BIRD-Deck | totalité |
| `sinabis-group/sinabis-100k-synthetic-pptx` | CC-BY-4.0 | 100 000 .pptx | générateur non nommé ⚠️ | `deck` : extraire des **structures** de decks (plans, enchaînements), jamais le texte | ~5 000 |
| `nvidia/Nemotron-Personas-France` | CC-BY-4.0 | grand | synthétique ✅ | **Personas FR** pour BIRD-World (diversité des rôles et des styles) | ~2 000 |
| `AgentPublic/open_government` | Apache 2.0 | ~380 Md de tokens | documents publics ✅ | Documents d'entrée : synthèse, appels d'offres, finance publique | à la demande |
| `AgentPublic/legi` | Licence Ouverte (Etalab 2.0) | 1 M+ | Légifrance ✅ | Questions-réponses juridiques **avec citations** | à la demande |
| `LiquidAI/structured-extraction` | Apache 2.0 | 6 500 | fictif ✅ | `tools` : extraction vers JSON | totalité |
| `juancavallotti/multilingual-gec` | Apache 2.0 | 100 000+ | erreurs générées par règles | `correction` : graines FR (phrases simples) | ~5 000 FR |
| `HuggingFaceFW/fineweb-2` (FR) | ODC-BY | énorme | web | Textes propres pour l'injection d'erreurs (`correction`) | ~20 000 extraits |

**Bilan** : environ **150 000 exemples réutilisables**, dont environ 60 000 utilisés comme cibles directes (rejeu, outils, finance, préférences humaines) et le reste comme graines ou documents. Les ~250 000 exemples SFT restent **majoritairement générés par nous** (BIRD-World et Core Skills), car **rien sur le Hub ne couvre nos tâches cœur** : e-mails d'entreprise FR, comptes rendus, decks au format BIRD-Deck, routage de skills.

**À ne pas utiliser** : `openbmb/UltraFeedback` (réponses GPT-4), `allenai/tulu-3-*` et `WildChat` (GPT-4o / Claude, parties non commerciales), `moncefem/legi-instruct-fr` (Gemini), `HuggingFaceH4/no_robots` et `APIGen-MT` (non commercial), `Spider` / `BIRD-SQL` / Wikipédia (ShareAlike), les datasets FR de type Alpaca (traductions de sorties GPT).

**Manques** : aucun corpus parallèle FR↔EN permissif n'a pu être vérifié sur le Hub (Europarl, OPUS : licence « unknown »). On vérifiera les conditions à la source, ou on fera de la rétro-traduction avec Qwen. Aucun corpus FR de correction annoté par des humains n'existe : on le construira par injection d'erreurs.

## BIRD-Bench (évaluation, `adubrou/bird-bench`)

| Split               | Contenu                                                                                                             | Taille |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- | ------ |
| `tasks`             | 20 à 50 cas par tâche, **entreprises BIRD-World réservées** (jamais vues à l'entraînement)                          | ~2 500 |
| `golden`            | Cas écrits ou relus par des professionnels FR (assistante de direction, contrôleur de gestion, juriste, commercial) | ~1 000 |
| `skills_unseen`     | Routage sur des skills et des domaines **jamais vus**                                                               | ~1 000 |
| `ablation`          | Les mêmes cas en trois conditions : base sans skill / base avec skills dans le prompt / BIRD-M avec BSP             | ~600   |
| `judge_calibration` | Cas notés par des humains pour valider le juge                                                                      | 200    |

Décontamination : MinHash entre `bird-m-sft` et `bird-bench`. Le split est **fait au niveau de l'entreprise BIRD-World**, pas de l'exemple.

## Pipeline technique

```
graines (YAML) ──► vLLM (HF Jobs, Qwen3.8-27B) ──► filtres ──► juge ──► parquet ──► push Hub (config)
                         │ shardé : N jobs H200 en parallèle, chaque shard = 1 fichier, reprise idempotente
```

Chaque étape est un script `config → run → stats JSON` qu'on peut relancer shard par shard. Une carte de dataset est générée automatiquement (volumes, distributions, scores).

## État d'avancement

- ✅ `finance` v0.1 : générateur BIRD-World (`bird_m/world/finance.py`), 5 types de tâches (`bird_m/data/finance_tasks.py`) et vérificateur de chiffres (`bird_m/data/checks.py`). **2 000 exemples pilote** publiés dans `adubrou/bird-m-sft` (config `finance`) et **300 exemples d'évaluation** dans `adubrou/bird-bench`, issus d'entreprises disjointes. 60 % sont des réponses exactes calculées, 40 % sont à faire rédiger par le teacher.

## Ordre de construction

1. `world` (50 entreprises pour le pilote) + 40 Core Skills (relecture humaine).
2. Pilote : 2 000 exemples (`tasks` P0 + `skills`), puis BIRD-Bench v0.
3. Passage à l'échelle de toutes les configs. `deck` attend le schéma BIRD-Deck v0.
