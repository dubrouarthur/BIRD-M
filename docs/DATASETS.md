# Datasets BIRD-M

Dépôts (privés) : entraînement → [`adubrou/bird-m-sft`](https://huggingface.co/datasets/adubrou/bird-m-sft) (une _config_ par dataset ci-dessous) · évaluation → [`adubrou/bird-bench`](https://huggingface.co/datasets/adubrou/bird-bench) (jamais mélangé à l'entraînement).

## Principes

1. **Les réponses cibles viennent de sources sous licence propre.** On auto-distille Qwen3.8-27B en mode thinking xhigh, avec best-of-n et un juge. On n'utilise jamais de sorties d'API fermées, dont les conditions interdisent d'entraîner un modèle concurrent. Les datasets FR existants sur le Hub (Alpaca traduits, souvent générés par GPT) servent **au mieux de graines de prompts**, jamais de cibles.
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

### Sources de textes réels (documents d'entrée uniquement, licence à vérifier par sous-collection)

| Source                                                                                             | Licence                                | Usage                                                                                               |
| -------------------------------------------------------------------------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------- |
| [`AgentPublic/open_government`](https://huggingface.co/datasets/AgentPublic/open_government)       | Apache 2.0                             | Textes administratifs, juridiques et financiers FR/UE → synthèse, Q&A documentaire, appels d'offres |
| [`PleIAs/common_corpus`](https://huggingface.co/datasets/PleIAs/common_corpus)                     | Ouvertes (variables selon le document) | Textes FR propres → base de `correction`                                                            |
| [`LiquidAI/structured-extraction`](https://huggingface.co/datasets/LiquidAI/structured-extraction) | Apache 2.0, fictif                     | Graines d'extraction et contrôle croisé de `tools`                                                  |

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

## Ordre de construction

1. `world` (50 entreprises pour le pilote) + 40 Core Skills (relecture humaine).
2. Pilote : 2 000 exemples (`tasks` P0 + `skills`), puis BIRD-Bench v0.
3. Passage à l'échelle de toutes les configs. `deck` attend le schéma BIRD-Deck v0.
