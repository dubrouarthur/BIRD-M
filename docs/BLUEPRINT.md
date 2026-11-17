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
 (69 tâches)    personas × secteurs ×      Qwen3.8-27B thinking     dédup,       grille     │
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
~200 000 exemples × ~1 500 tokens × 2 epochs ≈ 600 millions de tokens, soit **~1 jour pour le SFT**. Génération des données avec vLLM sur les mêmes GPU : ~1 à 2 jours.
**Ordre de grandeur du budget de calcul v1 : 3 000 à 6 000 € de GPU loués**, hors itérations et hors relecture humaine.

## Évaluation : BIRD-Bench

Un jeu figé et versionné, avec un avant/après à chaque run :

- **Par tâche** (69) : 20 à 50 cas, méthode de mesure définie dans `tasks.yaml` (juge, schéma JSON, exécution de formules, grammaire…).
- **Comparaisons** contre la base et contre un modèle de référence ; le juge est pré-validé sur 200 cas notés par des humains (accord ≥ 0,7).
- **Skills** : routage sur skills vues et inédites, invariance aux permutations, primauté des politiques, injection.
- **Non-régression** : IFEval, MMLU-Pro, GSM8K ou équivalents FR.
- **Coût** : tokens générés par tâche.

## Phases et décisions go / no-go

| Phase                  | Livrable                                                                             | Go si                                                              |
| ---------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| **0. Fondations** ✅   | Taxonomie, spec BSP, implémentation de référence testée                              | Tests verts, ids vérifiés sur le vrai tokenizer                    |
| **1. Mini-pilote**     | 2 000 exemples, SFT LoRA sur **27B en QLoRA** (ou un plus petit Qwen de même architecture s'il existe), BIRD-Bench v0 | Le routage apprend (> 80 % sur skills inédites)                    |
| **2. Données v1**      | Générateur, filtres, juge, 200 000 exemples, golden set                              | Échantillon de 300 exemples relus avec ≥ 90 % jugés « publiables » |
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

1. **Budget et accès GPU** (cloud loué ? fournisseur FR comme Scaleway ou OVH pour l'argument souverain ?).
2. **Nom public** du modèle (BIRD-M ?) et organisation Hugging Face.
3. **Relecture humaine** : qui rédige ou valide le golden set FR (~1 000 cas) ?
4. **Vision** : la garder gelée en v1 (recommandé) ou entraîner l'analyse de documents scannés et de slides en v1.1 ?
