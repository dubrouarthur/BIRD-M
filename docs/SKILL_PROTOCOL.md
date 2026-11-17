# BIRD-M Skill Protocol (BSP) — spécification v0.1

> Objectif : qu'une entreprise puisse **ajouter ses propres compétences (« skills ») au modèle sans le ré-entraîner**,
> que le modèle sache **quand** les utiliser, et que la DSI garde **le contrôle et la traçabilité** de leur usage.

## 1. Le problème

Aujourd'hui, pour spécialiser un LLM sur « notre façon de faire », on a deux options :

| Option                             | Limite                                                                                                                                      |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Tout mettre dans le prompt système | Le contexte explose au-delà de quelques procédures, coûte cher à chaque requête, et rien ne garantit que la bonne procédure soit appliquée. |
| Fine-tuner par client              | Long, coûteux, à refaire à chaque changement de procédure ; impossible de savoir _quelle_ règle a été appliquée.                            |

BSP propose une troisième voie, **apprise dès le premier fine-tuning**.

## 2. L'idée en une phrase

**Des tokens de contrôle dédiés et 200 « slots » de skills sont des _pointeurs_ liés à la volée. Router vers une skill tient en un seul token, auditable et masquable. Le contenu de la skill n'est chargé qu'au moment où le modèle l'invoque.**

## 3. Les quatre innovations

### 3.1 Tokens natifs gratuits (zéro changement d'architecture)

Qwen3.8-27B réserve **248 320** lignes d'embedding, mais le tokenizer n'en utilise que **248 077** (ids 0 → 248 076).
BSP ajoute **211 tokens** (11 de contrôle + 200 slots) dans les ids **248 077 → 248 287**, déjà alloués et jusqu'ici inutilisés.
→ Les poids gardent exactement la même forme : compatibles vLLM, SGLang, llama.cpp/GGUF et quantisation, sans patch.
_Vérifié sur le vrai `tokenizer.json` (`tests/test_skills.py::test_real_tokenizer_assigns_expected_ids`)._

| Token                                                   | Rôle                                                                                    |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `<\|skill_catalog_start\|>` / `<\|skill_catalog_end\|>` | Encadrent le catalogue (prompt système)                                                 |
| `<\|skill_start\|>` / `<\|skill_end\|>`                 | Encadrent une carte de skill (métadonnées seulement)                                    |
| `<\|skill_000\|>` … `<\|skill_199\|>`                   | **Slots** : poignées liées à une skill, différentes à chaque requête                    |
| `<\|skill_invoke\|>`                                    | Émis par le modèle : « je charge cette skill »                                          |
| `<\|skill_none\|>`                                      | Émis par le modèle : « aucune skill ne s'applique »                                     |
| `<\|skill_body_start\|>` / `<\|skill_body_end\|>`       | Contenu injecté par le runtime (jamais généré)                                          |
| `<\|policy_start\|>` / `<\|policy_end\|>`               | Politiques d'entreprise toujours actives, prioritaires sur l'utilisateur                |
| `<\|skill_mem_pad\|>`                                   | Emplacement réservé aux _skills compilées_ (§3.4), même mécanique que `<\|image_pad\|>` |

### 3.2 Slots-pointeurs liés en contexte (le cœur de l'innovation)

Un slot **n'a pas de sens fixe**. Pendant l'entraînement, la correspondance skill → slot est **tirée au hasard à chaque exemple**, et l'ordre du catalogue est mélangé.
Le modèle ne peut donc pas mémoriser « `<|skill_017|>` = relance d'impayés » : il **apprend à lire le catalogue et à pointer**.
Conséquence : à l'inférence, **n'importe quelle skill nouvelle, jamais vue à l'entraînement**, est routable dès qu'on l'ajoute au catalogue.

En production, la liaison est **déterministe** (hachage du nom). Un même catalogue produit donc un préfixe identique, ce qui permet de réutiliser le cache KV (prefix caching) d'une requête à l'autre.

### 3.3 Routage en un token : mesurable, gouvernable, auditable

Quand un catalogue est présent, la **première décision** du modèle est toujours, au choix :
`<|skill_none|>` ou `<|skill_invoke|><|skill_k|>`.

Cela donne, en une seule passe :

- **une distribution de probabilité sur les skills** (les logits des slots) : seuils de confiance, métriques de routage, détection du « je ne suis pas sûr » ;
- **une gouvernance au niveau du token** : le runtime met un `logit_bias = -100` sur tous les slots non liés. Le modèle ne peut ni inventer une skill, ni invoquer une skill que l'utilisateur n'a pas le droit d'utiliser (RBAC via le champ `roles`) ;
- **un journal d'audit** : « telle réponse a appliqué _relance-impayes v1.2.0_ ». C'est utile pour la conformité et pour la traçabilité attendue par l'AI Act.

**Chargement progressif** : le catalogue ne contient que les cartes (environ 60 tokens par skill). Le contenu, qui peut faire des milliers de tokens, n'est injecté **que si la skill est invoquée**. Ainsi, 200 skills coûtent environ 12 000 tokens de catalogue au lieu de centaines de milliers.

```
<|im_start|>assistant
<|skill_invoke|><|skill_017|>                ← généré (arrêt sur slot)
<|skill_body_start|> …procédure… <|skill_body_end|>   ← injecté par le runtime
<think>…</think>                              ← raisonnement optionnel
Objet : Facture n° 2026-118 – Niveau 1 …<|im_end|>
```

Le runtime s'arrête sur n'importe quel slot (`stop_token_ids = SLOT_IDS`), injecte le contenu et reprend dans le **même tour** (`continue_final_message`). Il n'y a pas d'aller-retour d'outil et le cache est réutilisé. On peut enchaîner jusqu'à 3 skills (par exemple procédure + charte).

### 3.4 Skills compilées (v1.1, piste de recherche)

Pour une skill très utilisée et longue (charte de 5 000 tokens), on la **distille en _k_ vecteurs** (k = 8 à 32) placés aux positions `<|skill_mem_pad|>`. C'est le principe des _gist tokens_ : apprentissage de prompt dont la cible est le comportement du modèle avec la skill complète dans le contexte.

- Coût de stockage : k × 5 120 × 2 octets ≈ **80 à 330 Ko par skill**, propre à chaque client, interchangeable à chaud.
- Aucun poids du modèle n'est modifié. Un client = un dossier de vecteurs.
- Risque : qualité inférieure au texte complet. On ne le livre que si l'éval montre une perte ≤ 2 points pour un coût en contexte divisé par plus de 50.

## 4. Format des skills (compatible Agent Skills)

Un dossier par skill avec un fichier `SKILL.md` (front-matter YAML + Markdown). Les champs `name` et `description` suivent le format ouvert _Agent Skills_ ; BSP ajoute :

| Champ              | Exemple                                                     | Usage                                       |
| ------------------ | ----------------------------------------------------------- | ------------------------------------------- |
| `type`             | `procedure` · `style` · `knowledge` · `template` · `policy` | `policy` = toujours injectée, non invocable |
| `when_to_use`      | « Relance d'un client en retard »                           | Signal de routage principal                 |
| `roles`            | `[finance, commercial]`                                     | RBAC : filtre du catalogue et `logit_bias`  |
| `version`, `owner` | `1.2.0`, `direction-financiere`                             | Audit                                       |
| `priority`         | `100`                                                       | Ordre des politiques                        |
| `lang`             | `[fr, en]`                                                  | Filtrage                                    |

Exemples : `examples/skills/`.

## 5. Sécurité

- **Injection de tokens** : le tokenizer de base a `split_special_tokens=False`. Un e-mail contenant littéralement `<|skill_invoke|>` deviendrait donc un vrai token de contrôle. `sanitize()` neutralise tout motif `<|…|>` dans les contenus utilisateur ou documentaires (espace insécable de largeur nulle). Test : `test_user_injection_is_neutralised` et contre le vrai tokenizer.
- **Slot non lié** : il est bloqué par `logit_bias`, puis rejeté par `pending_injection()` en défense en profondeur.
- **Boucle d'invocation** : plafond de 3 invocations par tour.
- **Politiques contre utilisateur** : comportement entraîné (tâche `policy.obey`) et mesuré.

## 6. Ce que l'entraînement doit apprendre (tâches `agentique.*`)

| Compétence              | Construction des données                                                                                                 | Métrique                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| Router                  | Catalogue de 3 à 40 skills, dont 0 à 2 pertinentes et des distracteurs proches (« relance impayés » / « relance devis ») | Accuracy et F1, y compris la classe _none_     |
| S'abstenir              | 30 % d'exemples sans skill pertinente                                                                                    | Taux de faux positifs                          |
| Indépendance au slot    | Liaison aléatoire, ordre mélangé                                                                                         | Écart de précision entre permutations (< 1 pt) |
| Généraliser             | Skills de test **jamais vues** (noms et domaines inédits)                                                                | Accuracy de routage sur skills inédites        |
| Appliquer               | Réponse jugée contre le contenu de la skill (respect des contraintes vérifiables)                                        | Taux de règles respectées                      |
| Composer                | 2 ou 3 skills invoquées                                                                                                  | Juge                                           |
| Primauté des politiques | L'utilisateur demande explicitement de violer la politique                                                               | Taux de respect                                |

Seules les décisions de routage et la réponse portent la perte. Le contenu injecté est masqué (`Segment.train = False`).

## 7. Implémentation de référence

`bird_m/skills/` : `spec.py` (tokens et ids), `skill.py` (parse `SKILL.md`, RBAC), `protocol.py` (liaison, rendu masqué pour le SFT, `sanitize`, `logit_bias`, `pending_injection`).
Tests : `pytest` (21 tests ; mettre `BIRD_M_TOKENIZER=…/tokenizer.json` pour vérifier contre le vrai tokenizer).
