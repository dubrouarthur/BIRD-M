# Pourquoi et comment les skills rendent BIRD-M plus performant

> Thèse : sur les tâches d'entreprise, l'écart de qualité d'un bon LLM vient rarement d'un manque d'intelligence.
> Il vient d'un **manque de contexte** (nos modèles de documents, notre ton, nos procédures, nos chiffres) et d'un **manque de rigueur** (oublier une mention obligatoire, se tromper dans un calcul, déborder d'une slide).
> Les skills corrigent ces deux défauts. BSP les rend **apprises, routées et mesurées**, alors qu'ailleurs elles ne sont que collées dans le prompt.

## 1. Pourquoi c'est plus performant : 6 mécanismes

| #   | Mécanisme                                                                                                                                                     | Effet attendu                                                                                                               | Comment on le mesure                                                                               |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 1   | **Contexte juste** : la procédure ou le modèle exact de l'entreprise est dans le contexte au bon moment                                                       | Réponses « à la maison » au lieu de réponses génériques                                                                     | _Skill lift_ = score avec skill − score sans, sur BIRD-Bench                                       |
| 2   | **Contexte court** : seules les cartes (≈ 60 tokens) restent en permanence, le contenu n'est chargé qu'à l'invocation                                         | Pas de dilution de l'attention sur un prompt système de 50 000 tokens. Moins de latence et de coût, cache de préfixe stable | Qualité à catalogue de 5 vs 200 skills ; tokens par requête                                        |
| 3   | **Expertise externalisée** : une skill écrite par un expert (ou par un modèle frontière, une seule fois) transmet ce savoir-faire au 27B **à chaque requête** | Un modèle moyen avec une excellente procédure bat un excellent modèle sans procédure sur la tâche visée                     | BIRD-M + skill contre modèle plus gros sans skill                                                  |
| 4   | **Mise à jour sans ré-entraînement** : changer une règle = éditer un fichier                                                                                  | Le modèle ne vieillit pas avec les procédures (réforme, nouvelle charte)                                                    | Délai entre changement de règle et comportement conforme : quelques minutes au lieu d'un fine-tune |
| 5   | **Décomposition** : _choisir_ (routage, 1 token) et _appliquer_ (génération) sont deux compétences entraînées et mesurées séparément                          | On sait d'où vient une erreur : mauvaise skill ou mauvaise application                                                      | Accuracy de routage et fidélité d'application, séparément                                          |
| 6   | **Fidélité procédurale entraînée** : des dizaines de milliers d'exemples « applique cette procédure à la lettre »                                             | Meilleur suivi d'instructions **en général**, même sans skill                                                               | IFEval / IFEval-FR avant et après                                                                  |

Point clé pour le pitch : **le gain ne vient pas des skills seules, mais du modèle entraîné à s'en servir.** On le prouve par trois mesures sur les mêmes cas :
(a) Qwen3.8 sans skill, (b) Qwen3.8 **avec les mêmes skills collées dans le prompt**, (c) BIRD-M avec BSP. La différence (c) − (b), c'est notre valeur ajoutée.

## 2. Comment on maximise le gain : 7 leviers

### Levier 1 — Les « BIRD Core Skills » : une bibliothèque officielle livrée avec le modèle

Environ 40 skills maison couvrent nos tâches P0. Elles servent à la fois de **données d'entraînement** et de **produit** :

| Skill               | Contenu                                                                                       |
| ------------------- | --------------------------------------------------------------------------------------------- |
| `bird-deck`         | Schéma BIRD-Deck, catalogue de layouts, règles de densité, storytelling (titre = affirmation) |
| `bird-excel-fr`     | Table des noms de fonctions FR↔EN, séparateurs, pièges (dates, RECHERCHEV vs RECHERCHEX)      |
| `bird-fin-analyse`  | Formules des SIG, BFR, ratios et seuils d'alerte, checklist d'analyse, règles de prudence     |
| `bird-email-fr`     | Formules d'appel et de politesse selon la relation, vouvoiement, structure, objet             |
| `bird-cr-reunion`   | Structure de compte rendu : décisions, actions, responsable, échéance                         |
| `bird-appel-offres` | Structure du mémoire technique, lecture du RC/CCTP, critères de notation                      |
| `bird-typo-fr`      | Règles typographiques françaises (espaces insécables, « », majuscules accentuées)             |

Avantage : quand le schéma BIRD-Deck passe en v2, **on met à jour la skill, pas les poids**. Les poids apprennent _à utiliser_ la skill ; la skill porte la _spécification_.

### Levier 2 — Distillation de skills : les skills améliorent aussi le modèle _sans_ skill

Pour chaque tâche couverte par une Core Skill, le teacher génère la réponse **avec la skill dans son contexte**, ce qui donne une réponse nettement meilleure. On entraîne ensuite deux versions du même exemple :

- **avec** la skill, pour apprendre à l'appliquer ;
- **sans** la skill, avec la même réponse cible, pour intégrer ce savoir-faire dans les poids (_context distillation_).

Résultat : même un utilisateur qui n'a configuré aucune skill profite de l'expertise. Ratio de départ : 50 / 50, à ajuster selon BIRD-Bench.

### Levier 3 — Skills exécutables : le calcul fait par du code, pas « de tête »

Une skill peut embarquer des **outils** (dossier `scripts/` du format Agent Skills, exposés en `<tool_call>` Qwen). Exemples : `bird-fin-analyse` fournit `compute_ratios()` et `bird-excel-fr` fournit `check_formula()`.
Les chiffres d'une analyse financière sont alors **calculés**, et le modèle apprend à déléguer le calcul puis à interpréter le résultat. C'est le plus gros levier de fiabilité pour la finance. → BSP v0.2 : `type: tool`.

### Levier 4 — Checklists de vérification

Chaque skill peut se terminer par une section `## Vérifications`. Le modèle est entraîné à **passer la checklist avant de répondre** (dans le raisonnement s'il est activé, sinon de façon implicite). Les violations sont mesurées automatiquement là où c'est possible : objet présent, mention légale, densité de slide.

### Levier 5 — Composition et priorités explicites

L'ordre de priorité est fixe : **politique (RGPD, sécurité) > charte du client > skill métier du client > BIRD Core Skill > comportement par défaut.**
On l'entraîne avec des exemples de **conflit** construits exprès, par exemple une Core Skill qui dit « signature : prénom » contre une charte client qui dit « signature : service ».

### Levier 6 — Le modèle rédige les skills (onboarding en minutes)

Nouvelle tâche `skill.author` : à partir de 3 à 5 documents exemplaires d'une entreprise (e-mails, deck, CR), le modèle **rédige le `SKILL.md`** qui capture le style et la structure. L'administrateur relit et valide.
C'est l'argument commercial : « Donnez-nous 5 documents, BIRD-M écrit comme vous en 10 minutes. »

### Levier 7 — Boucle d'amélioration continue en production

Le site et l'éditeur enregistrent, avec consentement, les **corrections que les utilisateurs apportent** aux réponses.
Quand une même correction revient (par exemple « remplace toujours _meeting_ par _réunion_ »), BIRD-M **propose une mise à jour de la skill**, que l'administrateur valide.
Le modèle s'améliore donc **par entreprise** sans ré-entraînement, et ces données alimentent BIRD-M v2.

### Bonus v1.1 — Skills compilées

Voir `SKILL_PROTOCOL.md` §3.4 : une Core Skill très utilisée est distillée en quelques vecteurs, ce qui permet un coût de contexte quasi nul.

## 3. Ce que ça change dans le plan

- Taxonomie : ajout de `skill.author`, `skill.conflict` et `skill.checklist`.
- Données : environ 40 Core Skills à écrire **avant** la génération (relues par un humain) ; les données sont générées _avec_ ces skills (levier 2).
- Éval : BIRD-Bench inclut l'ablation (a)/(b)/(c) et le _skill lift_ par domaine.
- Objectif v1 : _skill lift_ ≥ +15 points sur les cas « à la maison », et (c) − (b) ≥ +8 points.
