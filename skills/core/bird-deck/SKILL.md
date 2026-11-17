---
name: bird-deck
description: Produit une présentation au format BIRD-Deck (JSON) rendue par l'éditeur BIRD, avec storytelling, choix de layouts et graphiques.
when_to_use: L'utilisateur demande des slides, une présentation, un deck, un support de réunion ou de transformer un document en présentation.
type: template
lang: [fr, en]
version: 0.1.0
owner: bird-models
---

# BIRD-Deck : produire une présentation

Réponds par **un seul objet JSON** valide `bird-deck/0.1`, sans texte autour. Tu choisis le contenu et les layouts. Le design (couleurs, polices, grilles, graphiques) est géré par l'éditeur : ne donne jamais de HTML, de CSS ni de positions.

## Structure

`{"version": "bird-deck/0.1", "meta": {"title", "lang": "fr"|"en"}, "theme": {"preset", "brand"}, "slides": [...]}`
Presets disponibles : `bird`, `ardoise`, `foret`, `corail`. Si une charte client est chargée, utilise ses couleurs dans `theme.primary` et `theme.accent` (format `#RRGGBB`).

## Layouts (champ `layout`)

| Layout        | Champs                                                                        | Quand l'utiliser                                     |
| ------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------- |
| `title`       | title, subtitle?, kicker?                                                     | Première slide, toujours                             |
| `agenda`      | title, items (2–7)                                                            | Decks de plus de 8 slides                            |
| `section`     | title, number?, subtitle?                                                     | Changement de partie                                 |
| `bullets`     | title, bullets (2–6), kicker?                                                 | Liste d'idées, en dernier recours                    |
| `two_columns` | title, left/right {heading, bullets 1–5}                                      | Avant/après, forces/faiblesses                       |
| `big_number`  | title, value (≤ 9 caractères), label, context?, trend?                        | Un chiffre qui porte le message                      |
| `kpis`        | title, kpis 2–4 {value ≤ 8 caractères, label, delta?, trend?}                 | Tableau de bord                                      |
| `chart`       | title, chart {type, categories, series, unit?, highlight?}, takeaway, source? | Toute évolution, répartition ou comparaison chiffrée |
| `comparison`  | title, options 2–3 {heading, points, recommended?}, verdict?                  | Choix et décision                                    |
| `timeline`    | title, steps 3–6 {date, label, detail?, done?}                                | Planning, feuille de route                           |
| `table`       | title, header (2–5), rows (≤ 7), highlight_row?                               | Données précises à lire                              |
| `quote`       | quote, author, role?                                                          | Parole client ou dirigeant                           |
| `closing`     | title, bullets?, cta?, contact?                                               | Dernière slide : décision ou prochaine étape         |

Types de graphique : `bar` (comparer, évolution courte), `line` (tendance, 5 points ou plus), `donut` (répartition, 6 parts au maximum, une seule série), `stacked_bar` (composition dans le temps). `highlight` met en valeur la catégorie dont parle le titre.

## Règles d'écriture

1. **Le titre est une affirmation**, pas un thème : « Le B2B affiche la plus forte croissance » et non « Ventes par canal ». Au maximum 14 mots.
2. **Une idée par slide.** Le takeaway d'un graphique dit ce qu'il faut retenir, en 22 mots au maximum.
3. **Montrer plutôt que lister** : dès qu'il y a des chiffres, utilise `chart`, `kpis`, `big_number` ou `table`, pas `bullets`. Jamais 3 slides `bullets` d'affilée, et au plus la moitié du deck en `bullets`.
4. **Densité** : puces de 16 mots au maximum, phrases nominales ou verbales courtes, pas de point final.
5. **Chiffres exacts** : n'invente aucune donnée. Si un chiffre manque, mets un libellé explicite (`"à compléter"`) et signale-le dans `notes`.
6. Format français : `18,4 M€`, `+12 %`, `T3 2026`. En anglais : `€18.4M`, `+12%`, `Q3 2026`.
7. `notes` contient le discours de l'orateur (150 mots au maximum) quand c'est utile.
8. Longueur : 6 à 12 slides pour une réunion, 3 à 5 pour un point rapide, sauf demande contraire.

## Vérifications avant de répondre

- [ ] La première slide est `title`, la dernière `closing` (décision, prochaine étape ou contact).
- [ ] Chaque titre est une affirmation qui se suffit à elle-même : lus à la suite, les titres racontent l'histoire.
- [ ] Le nombre de valeurs de chaque série est égal au nombre de catégories.
- [ ] Aucune limite de mots ou d'éléments n'est dépassée.
- [ ] Les layouts sont variés.
