# BIRD-M — Blueprint v0.1

## Objectif

Un modèle **Apache 2.0** dérivé de **Qwen3.8-27B**, meilleur que sa base sur les **tâches quotidiennes et peu coûteuses des entreprises françaises** (FR et EN), avec le **BIRD-M Skill Protocol** pour brancher les compétences propres à chaque entreprise sans ré-entraînement.

Critères de succès v1 (mesurés sur BIRD-Bench, §5) :

1. Taux de victoire **≥ 65 %** contre Qwen3.8-27B (non-thinking) sur les tâches P0 en français, et **≥ 55 %** en anglais.
2. Routage de skills : **≥ 95 %** d'accuracy sur skills vues, **≥ 90 %** sur skills **jamais vues**, faux positifs **≤ 3 %**.
3. Pas de régression de plus de 2 points sur les benchmarks généraux (IFEval, MMLU-Pro, versions FR).
4. **Moins de tokens par tâche** que la base : la plupart des tâches P0 se font sans raisonnement (`thinking: direct`).

## Base : ce qu'on sait de Qwen3.8-27B (vérifié sur `config.json` et `tokenizer.json`)

- 27 milliards de paramètres denses, 64 couches hybrides : 16 × (3 × Gated DeltaNet + 1 × Gated Attention). Encodeur vision intégré (`Qwen3_5ForConditionalGeneration`).
- Vocabulaire de 248 077 tokens utilisés pour **248 320 lignes allouées**, ce qui laisse 243 lignes libres pour les tokens BSP (voir `SKILL_PROTOCOL.md`).
- ChatML avec `<think>`, appels d'outils `<tool_call>`. Mode thinking activé par défaut, `reasoning_effort` (low, medium, xhigh). Contexte natif de 262 000 tokens.
- Conséquences techniques : il faut les noyaux _flash-linear-attention_ et _causal-conv1d_ pour entraîner les couches DeltaNet efficacement. On **gèle la tour vision** en v1. Le _packing_ de séquences doit respecter les frontières de documents dans l'état récurrent (à vérifier selon la version de transformers/TRL).

## Pipeline

```
 tasks.yaml ──► générateur de prompts ──► teacher (génération) ──► filtres ──► juge ──► SFT mix
 (81 tâches)    personas × secteurs ×      Qwen3.8-27B thinking     dédup,       grille     │
                langue × ton × contraintes  xhigh + best-of-n        langue, PII  par tâche  │
                                                                                             ▼
 skills synthétiques (≈ 3 000) ──► rendu BSP (liaison aléatoire) ───────────────────► SFT (LoRA + tokens)
                                                                                             │
 paires préférées/rejetées (juge + humains FR) ─────────────────────────────────────► DPO
                                                                                             │
 récompenses vérifiables (JSON, formules Excel, routage, longueur) ─────────────────► GRPO (option)
                                                                                             ▼
                                                                        BIRD-Bench ──► go / no-go
```

### Données (cible v1)

> Vue d'ensemble ci-dessous ; le détail, qui fait foi (BIRD-World, 11 configs, ≈ 250 000 exemples), est dans [`DATASETS.md`](DATASETS.md).

| Bloc                                                                            | Part | Volume   | Source                                                             |
| ------------------------------------------------------------------------------- | ---- | -------- | ------------------------------------------------------------------ |
| Tâches d'entreprise P0/P1 (FR 60 %, EN 40 %)                                    | 55 % | ~110 000 | Synthétiques à partir de la taxonomie, auto-distillées et filtrées |
| Protocole de skills (routage, application, composition, abstention, politiques) | 15 % | ~30 000  | ~3 000 skills synthétiques réalistes + distracteurs                |
| Outils et JSON                                                                  | 10 % | ~20 000  | Format natif Qwen `<tool_call>`                                    |
| Rejeu général (raisonnement, code, culture)                                     | 15 % | ~30 000  | Échantillonné depuis la base (anti-oubli)                          |
| Fiabilité (abstention, PII, injection, langue)                                  | 5 %  | ~10 000  | Synthétiques ciblés                                                |
| **Golden set humain** (éval uniquement, jamais entraîné)                        | —    | ~1 000   | Rédigé ou relu par des pros FR                                     |

Principes :

- **Auto-distillation d'abord** : le teacher est la base elle-même en mode thinking xhigh, avec best-of-n jugé. C'est sous licence Apache et sans restriction de conditions d'utilisation. Les sorties d'API fermées sont exclues du jeu d'entraînement, car leurs conditions interdisent souvent d'entraîner un modèle concurrent.
- **Le style français est la valeur ajoutée** : vouvoiement, formules de politesse, typographie (espaces insécables, guillemets « »), formules Excel en version FR. Le juge a une grille spécifique au français.
- **Diversité** : 12 secteurs (banque, BTP, industrie, santé, public, retail, conseil…), 4 tailles d'entreprise, 10 rôles. Contraintes de longueur et de ton tirées aléatoirement.
- **Pas de données personnelles réelles**. Décontamination contre BIRD-Bench par MinHash.

### Entraînement

| Étape            | Méthode                                                                                                                                                                                                                | Paramètres de départ                                                                        |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 0. Tokens        | Ajout des 211 tokens BSP. Init = moyenne des embeddings + bruit, slots initialisés depuis l'embedding de « skill »                                                                                                     | Sans resize                                                                                 |
| 1. SFT           | LoRA r=64, α=128, sur toutes les projections linéaires du LM (attention + DeltaNet + MLP) ; **lignes d'embedding et de lm_head des 211 tokens entraînées en plein** (`trainable_token_indices` de PEFT) ; vision gelée | lr 1e-4, 2 epochs, 8 000 tokens de contexte, perte sur les segments `train=True` uniquement |
| 2. DPO           | Paires : meilleur contre pire selon le juge, plus des préférences humaines FR                                                                                                                                          | β=0,1, 1 epoch                                                                              |
| 3. GRPO (option) | Récompenses vérifiables : schéma JSON, exécution de formules, routage, longueur                                                                                                                                        | Seulement si l'étape 2 plafonne                                                             |

Matériel (estimation à confirmer par un run pilote) : 8 × H100 80 Go, FSDP ou ZeRO-3, bf16.
≈ 250 000 exemples × ~1 500 tokens × 2 epochs ≈ 750 millions de tokens. Durées et coûts : voir la section Infrastructure.
**Ordre de grandeur du budget de calcul v1 : 3 000 à 6 000 € de GPU loués**, hors itérations et hors relecture humaine.

## Axe prioritaire 1 : des slides magnifiques (BIRD-Deck)

Un LLM qui écrit du HTML ou du .pptx « à la main » produit des slides moyennes : le design n'est pas son métier. On sépare donc les rôles :

```
modèle ──► BIRD-Deck (JSON sémantique) ──► moteur de design de l'éditeur web ──► rendu
            contenu, choix de layout,       thèmes, grilles, typo,               │
            message clé, données            charte de l'entreprise               ▼
     ▲                                                                capture PNG
     └──────────── récompense : schéma valide + densité + juge VISUEL ◄──┘ (Qwen3.8 est multimodal)
```

- **BIRD-Deck** : schéma JSON partagé entre le modèle et l'éditeur. Il comprend un catalogue fermé d'environ 15 layouts (titre, agenda, grand chiffre, deux colonnes, comparaison, graphique, chronologie, citation, image + texte, équipe, conclusion…). Chaque slide porte un *message clé* (titre-affirmation) et des limites de densité par layout, par exemple 6 puces de 12 mots au maximum. Les graphiques sont décrits par leurs données, et c'est l'éditeur qui les dessine.
- **La charte de l'entreprise est une skill BSP** (`type: style` ou `template`) : couleurs, polices, logo, ton. Le même modèle produit ainsi des decks « à la marque » de chaque client.
- **Entraînement** : SFT sur des decks de référence, puis **récompense vérifiable et visuelle** : schéma valide, densité respectée, rendu sans débordement (mesuré dans le navigateur), plus une note esthétique d'un juge vision sur la capture. C'est la tâche où le GRPO a le plus de valeur.
- **Tâches** : `slides.deck`, `slides.edit` (opérations sur un deck existant, pour l'éditeur), `slides.chart`, `slides.critique` (à partir d'une capture), `fin.deck`.
- **Dépendance** : le schéma BIRD-Deck v0 et un moteur de rendu minimal (headless) doivent exister **avant** la génération des données slides. C'est le seul point où le site et le modèle se rejoignent.

## Axe prioritaire 2 : analyse financière

- **Calculs vérifiables** : ratios, SIG, BFR, DCF. Les données d'entraînement sont générées à partir d'états financiers *synthétiques mais cohérents* (bilan équilibré, flux réconciliés), produits par un générateur Python. La bonne réponse est donc calculée, pas devinée, et sert de récompense exacte.
- **Interprétation** : commentaire d'écarts, points d'attention et recommandations, évalués par un juge avec une grille de contrôleur de gestion.
- **Règles de prudence apprises** : hypothèses toujours explicites, unités et périodes rappelées, aucun chiffre inventé (s'il manque une donnée, le modèle la demande).
- **Sortie** : tableaux, formules Excel FR/EN (`fin.budget`) et deck de résultats (`fin.deck`).

## Infrastructure

**Ce conteneur n'a pas de GPU** : il sert au code, aux tests et à l'orchestration. Le calcul se fait sur des GPU loués, à la minute, uniquement pendant les runs.

### Recommandation : Hugging Face Jobs, sur H200

- **Déjà intégré** : nos dépôts privés (`adubrou/BIRD-M`, `adubrou/bird-m-sft`, `adubrou/bird-bench`) et le token sont sur Hugging Face. Un job se lance en une commande (`hf jobs uv run --flavor h200 …`) depuis ce conteneur, sans serveur à administrer.
- **Facturé à la minute**, avec un `--timeout` obligatoire. La facturation s'arrête aussi si le job plante.
- **La H200 a 141 Go de mémoire** : les 27B en bf16 (54 Go) tiennent **sur un seul GPU** pour la LoRA et la génération vLLM. Pas besoin de quantifier (QLoRA) pour le pilote, ce qui simplifie et préserve la qualité, surtout pour les couches DeltaNet.
- **Prérequis** : ajouter du **crédit** dans *Settings → Billing* du compte Hugging Face. Les Jobs exigent un solde positif ; l'abonnement PRO n'est pas obligatoire mais ses crédits mensuels comptent.
- **Le stockage d'un job est éphémère** : checkpoints et shards de données sont poussés sur le Hub au fil de l'eau, avec reprise idempotente.

Prix relevés le 17/11/2026 (`hf jobs hardware`) : `h200` 5 $/h · `h200x8` 40 $/h · `a100-large` 2,5 $/h · `rtx-pro-6000` (96 Go) 2,75 $/h.

| Étape | Flavor | Durée estimée | Coût estimé |
|---|---|---|---|
| Pilote (2 000 exemples, LoRA bf16) + BIRD-Bench v0 | `h200` | ~5–10 h | **~25–50 $** |
| Génération des données (≈ 1 M générations best-of-4, plus le juge) | N × `h200` en parallèle | ~200–300 GPU·h | **~1 000–1 500 $** |
| SFT v1 (≈ 750 M tokens) + DPO | `h200x8` | ~15–25 h | **~600–1 000 $** |
| Évals et itérations | `h200` ou `rtx-pro-6000` | à la demande | **~300–600 $** |
| **Total v1** | | | **≈ 2 000–3 200 $** |

Ce sont des estimations. Le pilote mesure les vrais débits (tokens/s en génération et en entraînement) et on recalcule le budget avant d'engager la suite. Le premier poste de coût est la **génération**, pas l'entraînement : on le réduit avec `reasoning_effort=medium` sur les tâches faciles et en comparant le débit par dollar de `rtx-pro-6000` et `h200` pendant le pilote.

**Plus tard, pour l'argument souverain** : le run final peut être refait sur **Scaleway** ou **OVHcloud** (GPU en France) avec les mêmes scripts. C'est utile pour vendre « entraîné en France », mais inutile pour itérer.

**Stack** : PyTorch, Transformers, PEFT, TRL, Accelerate (FSDP), avec les noyaux *flash-linear-attention* et *causal-conv1d* pour DeltaNet, et vLLM pour la génération et le serving. Le pilote sert aussi à valider que cette stack gère bien l'architecture hybride de Qwen3.8 (packing, gradient checkpointing, LoRA sur les projections DeltaNet).
**Automatisation** : chaque étape est un script reproductible (`config YAML → run → métriques JSON`). On peut ainsi lancer un run sur n'importe quel fournisseur, en CI ou en headless.

## Évaluation : BIRD-Bench

Un jeu figé et versionné, avec un avant/après à chaque run :

- **Par tâche** (81) : 20 à 50 cas, méthode de mesure définie dans `tasks.yaml` (juge, schéma JSON, exécution de formules, grammaire…).
- **Comparaisons** contre la base et contre un modèle de référence ; le juge est pré-validé sur 200 cas notés par des humains (accord ≥ 0,7).
- **Skills** : routage sur skills vues et inédites, invariance aux permutations, primauté des politiques, injection.
- **Non-régression** : IFEval, MMLU-Pro, GSM8K ou équivalents FR.
- **Coût** : tokens générés par tâche.

## Phases et décisions go / no-go

| Phase                  | Livrable                                                                             | Go si                                                              |
| ---------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| **0. Fondations** ✅   | Taxonomie, spec BSP, implémentation de référence testée                              | Tests verts, ids vérifiés sur le vrai tokenizer                    |
| **1. Mini-pilote**     | 2 000 exemples, SFT LoRA sur **27B en QLoRA** (ou un plus petit Qwen de même architecture s'il existe), BIRD-Bench v0 | Le routage apprend (> 80 % sur skills inédites)                    |
| **1b. BIRD-Deck v0** | Schéma JSON des slides, moteur de rendu minimal (headless), 50 decks de référence | Rendu sans débordement de 100 % des decks de référence |
| **2. Données v1**      | BIRD-World, 40 Core Skills, générateur, filtres, juge, ≈ 250 000 exemples, golden set                              | Échantillon de 300 exemples relus avec ≥ 90 % jugés « publiables » |
| **3. Entraînement v1** | SFT puis DPO sur 27B                                                                 | Critères de succès 1 à 4                                           |
| **4. Release**         | Poids Hugging Face, carte modèle, recette vLLM avec runtime BSP                      | Revue juridique                                                    |
| **W. Site Birdmodels** | Vitrine + chat de démo + éditeur de skills                                           | En parallèle de la phase 2, sans bloquer le modèle                 |

## Risques

| Risque                                                               | Mitigation                                                                                                                                                                                                                                                                          |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Oubli catastrophique ou perte de raisonnement                        | LoRA, 15 % de rejeu, éval de non-régression à chaque checkpoint                                                                                                                                                                                                                     |
| Juge biaisé (préfère les réponses longues)                           | Grille par tâche, normalisation de longueur, accord humain mesuré                                                                                                                                                                                                                   |
| Synthétique trop uniforme                                            | Personas, contraintes aléatoires, dédup sémantique, relecture humaine                                                                                                                                                                                                               |
| Noyaux DeltaNet ou packing mal supportés par la stack d'entraînement | Pilote sur 2 000 exemples avant le gros run, versions figées                                                                                                                                                                                                                        |
| Routage qui ne généralise pas aux skills inédites                    | Liaison aléatoire, noms de test inédits, critère go de la phase 1                                                                                                                                                                                                                   |
| Juridique                                                            | Apache 2.0 : conserver LICENSE et NOTICE de Qwen et indiquer les modifications. AI Act : un fine-tune ne fait pas de nous le fournisseur du modèle de base, mais nous documentons nos modifications (données, calcul, évals). Pas de données personnelles dans les jeux de données. |

## Décisions à prendre

1. **Crédit Hugging Face** : alimenter le compte (≈ 50 $ pour le pilote, puis ≈ 3 000 $ pour la v1) afin de lancer les Jobs.
2. **Nom public** du modèle (BIRD-M ?) et organisation Hugging Face.
3. **Relecture humaine** : qui rédige ou valide le golden set FR (~1 000 cas) ?
4. **Vision** : la garder gelée en v1 (recommandé) ou entraîner l'analyse de documents scannés et de slides en v1.1 ?
