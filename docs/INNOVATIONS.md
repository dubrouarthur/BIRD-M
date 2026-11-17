# Innovations techniques BIRD-M : exploiter le périmètre restreint

> Idée directrice : un modèle généraliste paie pour tout savoir faire. BIRD-M ne fait que ~80 tâches d'entreprise, en FR/EN, souvent répétitives, souvent vérifiables.
> On transforme cette restriction en avantage sur **trois axes** : qualité (récompenses vérifiables), coût (calcul adapté à chaque tâche) et vitesse (décodage spécialisé).

Les chiffres cités sont **ceux annoncés par les papiers** sur leurs propres benchmarks (résumés lus le 17/11/2026), pas des mesures sur BIRD-M. Chaque levier est validé par un avant/après sur BIRD-Bench avant d'être adopté.

## Synthèse

| #   | Innovation                                                                                                    | Axe     | Gain attendu                                                                       | Priorité |
| --- | ------------------------------------------------------------------------------------------------------------- | ------- | ---------------------------------------------------------------------------------- | -------- |
| 1   | **Calcul piloté par la skill** : le routage BSP fixe aussi le budget de raisonnement                          | coût    | − 50 % de tokens ou plus sur les tâches simples                                    | **P0**   |
| 2   | **Ré-alignement de la tête MTP** sur les sorties BIRD-M (décodage spéculatif natif)                           | vitesse | conserve, voire augmente l'accélération ; sinon elle s'effondre après le fine-tune | **P0**   |
| 3   | **Décodage par recopie** (prompt lookup / suffix decoding) pour correction et réécriture                      | vitesse | fort sur les tâches où la sortie recopie l'entrée                                  | **P0**   |
| 4   | **Décodage contraint** : grammaire BIRD-Deck, JSON des outils, routage BSP                                    | qualité | 100 % de sorties valides par construction                                          | **P0**   |
| 5   | **RL à récompenses vérifiables** avec nos vérificateurs (chiffres, rendu des slides, routage, formules Excel) | qualité | gains sur la structure et l'exactitude là où le SFT plafonne                       | **P1**   |
| 6   | **BIRD-M Duo** : élève 9B distillé du 27B, **même tokenizer, mêmes IDs BSP**, cascade 9B → 27B                | coût    | environ 3 fois moins de paramètres actifs sur la majorité des requêtes             | **P1**   |
| 7   | **Tête de brouillon à vocabulaire réduit FR/EN** (style FR-Spec)                                              | vitesse | sans perte de qualité, moins de calcul dans le brouillon                           | **P1**   |
| 8   | **Mode diff** pour la correction (option)                                                                     | coût    | sortie jusqu'à environ 5 fois plus courte selon les papiers                        | **P1**   |
| 9   | **Juge visuel des slides** : le modèle est multimodal, il note ses propres rendus                             | qualité | récompense esthétique sans annotateur                                              | **P1**   |
| 10  | Élagage de couches, réduction du vocabulaire principal, skills compilées en vecteurs                          | coût    | incertain sur une architecture hybride                                             | P2       |

## 1. Calcul piloté par la skill (notre innovation, P0)

Qwen3.8 « pense » par défaut, ce qui coûte des milliers de tokens, même pour un e-mail. Or la difficulté d'une tâche d'entreprise est **prévisible par son type**, et ce type est justement ce que le routage BSP détermine en un token.

- Chaque carte de skill porte un champ `compute: direct | short | full`, qui reprend la colonne `thinking` de la taxonomie.
- Après `<|skill_none|>`, le runtime applique le budget de la (des) skill(s) invoquée(s). En mode `direct`, il pré-remplit un raisonnement vide. En mode `short` ou `full`, il fixe un plafond de tokens de raisonnement.
- Sans skill, le modèle choisit lui-même : on l'entraîne en SFT avec des traces sans raisonnement pour les tâches `direct` de la taxonomie, et avec un raisonnement pour la finance et les formules. C'est une version supervisée d'AdaptThink, qui annonce −53 % de longueur et +2,4 points en maths.
- **C'est original, car le routage de compétence et le routage de calcul se font en une seule décision**, déjà payée. Il n'y a ni classifieur séparé ni RL nécessaire en v1.
- Mesure : tokens par tâche et qualité, en comparant thinking toujours activé, toujours désactivé et piloté par la skill.

## 2. Ré-alignement MTP (P0)

La tête MTP de Qwen prédit les tokens suivants de **la base**. Après notre fine-tune, la distribution change et le taux d'acceptation baisse. On ré-entraîne donc la tête MTP par auto-distillation sur les sorties de BIRD-M, une fois le modèle fini.

- Appuis : les brouillons spécialisés distillés hors ligne battent les génériques de 11 à 25 % (2503.07807, SambaNova). Des données synthétiques atteignent 80 à 93 % du gain obtenu avec de vraies requêtes. La distillation on-policy du brouillon donne plus de 5 fois d'accélération (2605.29343).
- Les textes d'entreprise (formules de politesse, structures de CR, JSON de slides) sont très prévisibles : on attend une acceptation **supérieure** à celle de la base.
- Mesure : longueur moyenne acceptée et tokens par seconde, avant et après ré-alignement.

## 3. Décodage par recopie (P0)

En correction et réécriture, 70 à 95 % de la sortie recopie l'entrée. Le _prompt lookup_ et le _suffix decoding_ (2411.04975, jusqu'à 5,3 fois sur des charges répétitives) proposent ces tokens sans modèle de brouillon. C'est une simple option de serving (vLLM, SGLang), activée pour les skills `correction.*`.

## 4. Décodage contraint (P0)

- Routage : `RoutingSession.allowed_next()` est déjà implémenté et testé.
- Slides : grammaire dérivée de `bird_m/deck/schema.json` (xgrammar ou outlines dans vLLM). Un deck invalide devient impossible, et les limites de mots restent vérifiées par le validateur.
- Outils : schéma JSON de chaque fonction.

## 5. RL à récompenses vérifiables (P1)

Le GRPO (ou une variante) sur les tâches où **on a déjà construit le vérificateur** :

| Tâche       | Récompense                                            | Déjà implémenté                   |
| ----------- | ----------------------------------------------------- | --------------------------------- |
| Finance     | `numbers_check` (chiffres exacts)                     | ✅ `bird_m/data/checks.py`        |
| Slides      | deck valide + zéro débordement au rendu + juge visuel | ✅ validateur et `deck_check.mjs` |
| Routage BSP | décision exacte (skill ou aucune)                     | ✅ épisodes `skills`              |
| Excel       | formule exécutée = résultat attendu                   | à faire                           |
| Outils      | nom et arguments exacts                               | à faire                           |

Appuis : RL-Struct (2512.00319) annonce 92 % de JSON valides, au-dessus du SFT. IFBench (2507.02833) montre l'efficacité des vérificateurs de contraintes, mais aussi le **risque de sur-apprentissage aux vérificateurs** : on garde une éval tenue à l'écart et un juge indépendant.

## 6. BIRD-M Duo : élève 9B à tokenizer identique (P1)

**Vérifié le 17/11/2026** : `Qwen/Qwen3.5-9B` (Apache 2.0, même architecture hybride, 32 couches, largeur 4096, tête MTP) a un vocabulaire et des règles de découpage **identiques** à Qwen3.8-27B. Le 27B n'a que 7 tokens audio en plus (ids 248 070 à 248 076).

- On ajoute 7 tokens de réserve au 9B, et **les 211 tokens BSP ont alors les mêmes IDs dans les deux modèles**.
- On peut faire une distillation au niveau des logits (le meilleur type de distillation), impossible entre tokenizers différents.
- **Cascade** : le 9B traite tout, et passe la main au 27B quand la skill invoquée est en `compute: full` ou quand sa confiance de routage est faible. Avec le même protocole et le même format, l'application ne voit pas la différence.
- Appuis : les petits élèves apprennent mieux sur des raisonnements courts (2502.12143), et de petits modèles spécialisés battent de grands généralistes sur des tâches étroites (2305.02301).
- Coût : environ 3 fois moins de calcul par token pour les requêtes servies par le 9B. Il tient en FP8 sur une carte de 16 Go.

## 7. Tête de brouillon à vocabulaire réduit (P1)

Le lm_head (≈ 1,27 milliard de paramètres, 248 320 lignes) est lu entièrement à chaque pas. On réduit **uniquement le brouillon** (MTP ou EAGLE) aux tokens fréquents en FR/EN sur nos données, comme FR-Spec (2502.14856), qui annonce −75 % de calcul de tête et une sortie identique. Le modèle principal reste complet : il n'y a **aucun risque de perte de qualité**.

## 8. Mode diff pour la correction (P1)

Option « liste des corrections » : on renvoie des opérations (position, ancien, nouveau) plutôt que le texte entier. Selon 2305.11862, la sortie tombe jusqu'à 21 % de sa longueur, à qualité comparable. L'éditeur web applique les opérations et les affiche en suivi de modifications, ce qui est **mieux pour l'utilisateur** qu'un texte réécrit.

## 9. Juge visuel des slides (P1)

Qwen3.8 comprend les images. On rend le deck, on capture chaque slide, et une instance du modèle note la lisibilité, la hiérarchie et l'équilibre selon une grille. Cette note devient une récompense en plus des contrôles exacts. Le juge est calibré sur environ 200 slides notées par des humains.

## 10. Écarté pour la v1 (P2)

- **Élagage de couches** : aucun travail publié sur les hybrides Gated DeltaNet. Le gain est incertain et la « guérison » coûte cher. Le Duo 9B donne mieux pour moins de risque.
- **Réduction du vocabulaire du modèle principal** : au mieux environ 5 % de bande passante en moins, avec un risque pour l'anglais, le code, le JSON et les formules.
- **Skills compilées en vecteurs** : les études montrent des pertes sur le rappel exact, or nos skills sont des consignes précises. On préfère la distillation dans les poids (levier 2 de `SKILLS_STRATEGY.md`) et le cache de préfixe. ⚠️ Il faut vérifier que le moteur de serving sait mettre en cache l'état récurrent des couches DeltaNet.

## Références (arXiv)

2505.13417 AdaptThink · 2505.14631 Large Hybrid-Reasoning Models · 2505.05315 Elastic Reasoning · 2503.07807 domain draft models · 2605.29343 Draft-OPD · 2503.01840 EAGLE-3 · 2411.04975 SuffixDecoding · 2502.14856 FR-Spec · 2512.00319 RL-Struct · 2507.02833 IFBench · 2509.04518 GRPO tool calling · 2305.11862 edit spans · 2506.02780 EfficientEdit · 2502.12143 small-model CoT distillation · 2305.02301 distilling step-by-step · 2403.03853 ShortGPT · 2411.15558 layer pruning · 2504.11409 Minitron-SSM · 2412.17483 gist study.
