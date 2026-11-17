# Taxonomie des tâches BIRD-M

> Fichier généré par `scripts/render_taxonomy.py` depuis `data/taxonomy/tasks.yaml`. Ne pas éditer à la main.

**69 tâches** · 12 domaines · P0 : 36 · P1 : 27 · P2 : 6 · toutes en FR et EN.

Légende : **P0** cœur v1 · **P1** v1 si budget · **P2** v1.1 — *thinking* : `direct` sans raisonnement, `short` court, `full` complet — *éval* : méthode de mesure automatique.

## Rédaction & communication

| ID | Tâche | Prio | Thinking | Éval |
|---|---|---|---|---|
| `email.write` | Rédiger un e-mail professionnel à partir de consignes | P0 | direct | judge |
| `email.reply` | Répondre à un e-mail (fil fourni), ton adapté | P0 | direct | judge |
| `email.followup` | Relance (devis, réponse, paiement) | P0 | direct | judge |
| `email.triage` | Trier une boîte mail : priorité, catégorie, action | P1 | direct | json_schema |
| `minutes` | Compte rendu de réunion depuis notes ou transcription | P0 | short | judge |
| `memo.internal` | Note interne / note de service | P0 | direct | judge |
| `chat.message` | Message Teams/Slack court et clair | P1 | direct | judge |
| `letter.formal` | Courrier formel (réclamation, résiliation, attestation) | P1 | direct | judge |
| `post.linkedin` | Post LinkedIn corporate / marque employeur | P1 | direct | pairwise |
| `press.release` | Communiqué de presse | P2 | short | judge |
| `newsletter` | Newsletter interne | P2 | direct | pairwise |

## Correction, réécriture & traduction

| ID | Tâche | Prio | Thinking | Éval |
|---|---|---|---|---|
| `proofread.fr` | Corriger orthographe, grammaire, accords, typographie FR | P0 | direct | grammar |
| `proofread.en` | Corriger un texte en anglais | P0 | direct | grammar |
| `rewrite.tone` | Changer le ton (plus formel, plus cordial, plus ferme) | P0 | direct | pairwise |
| `rewrite.shorten` | Raccourcir / densifier sous contrainte de longueur | P0 | direct | exact |
| `rewrite.simplify` | Simplifier (langage clair, FALC) | P1 | direct | judge |
| `translate.fr_en` | Traduction professionnelle FR↔EN (terminologie métier) | P0 | direct | judge |
| `explain.changes` | Expliquer les corrections apportées | P1 | direct | judge |

## Présentations & documents structurés

| ID | Tâche | Prio | Thinking | Éval |
|---|---|---|---|---|
| `slides.outline` | Plan de présentation (storyline, nb de slides) | P0 | short | judge |
| `slides.json` | Slides détaillées en JSON (titre, puces, notes, visuel suggéré) | P0 | short | json_schema |
| `slides.from_doc` | Transformer un document en présentation | P0 | short | judge |
| `doc.structure` | Structurer un document (plan, titres, résumé exécutif) | P1 | short | judge |
| `spec.writing` | Cahier des charges / expression de besoin | P1 | full | judge |

## Synthèse, extraction & analyse documentaire

| ID | Tâche | Prio | Thinking | Éval |
|---|---|---|---|---|
| `summarize.doc` | Résumer un document long (niveaux : 3 lignes, 1 page) | P0 | short | judge |
| `summarize.thread` | Synthèse d'un fil d'e-mails : décisions, actions, responsables | P0 | direct | judge |
| `extract.invoice` | Extraire les champs d'une facture / d'un bon de commande → JSON | P0 | direct | json_schema |
| `extract.contract` | Extraire les clauses clés d'un contrat (durée, préavis, pénalités) | P1 | short | json_schema |
| `qa.grounded` | Répondre à partir de documents fournis, avec citations | P0 | short | judge |
| `compare.versions` | Comparer deux versions d'un document | P1 | short | judge |
| `feedback.analysis` | Analyser des verbatims clients / enquêtes (thèmes, sentiment) | P1 | short | judge |

## Données, tableurs & SQL

| ID | Tâche | Prio | Thinking | Éval |
|---|---|---|---|---|
| `excel.formula.fr` | Formules Excel en version française (RECHERCHEX, SOMME.SI.ENS, séparateur ;) | P0 | short | formula_exec |
| `excel.formula.en` | Formules Excel en version anglaise | P0 | short | formula_exec |
| `excel.explain` | Expliquer / déboguer une formule ou un tableau | P0 | short | judge |
| `sql.query` | Écrire une requête SQL à partir d'un schéma | P1 | short | exact |
| `table.analysis` | Analyser un tableau (tendances, anomalies, commentaire) | P1 | full | judge |
| `script.automation` | Petit script d'automatisation (Python, VBA, Apps Script) | P2 | full | exact |

## RH, juridique & administratif (France)

| ID | Tâche | Prio | Thinking | Éval |
|---|---|---|---|---|
| `hr.job_ad` | Fiche de poste / offre d'emploi (non discriminante) | P0 | direct | judge |
| `hr.review` | Préparer un entretien annuel / professionnel | P1 | short | judge |
| `hr.onboarding` | Parcours d'intégration, livret d'accueil | P2 | short | judge |
| `hr.policy_qa` | Répondre à une question RH à partir de l'accord / du règlement fourni | P1 | short | judge |
| `legal.explain` | Expliquer simplement une notion juridique FR (avec renvoi à un professionnel) | P1 | full | judge |
| `gdpr.notice` | Mentions d'information RGPD, fiche registre de traitement | P1 | short | judge |

## Commercial, marketing & appels d'offres

| ID | Tâche | Prio | Thinking | Éval |
|---|---|---|---|---|
| `sales.proposal` | Proposition commerciale | P0 | short | judge |
| `sales.prospecting` | E-mail de prospection personnalisé | P1 | direct | pairwise |
| `tender.memoire` | Mémoire technique pour appel d'offres (marchés publics) | P1 | full | judge |
| `tender.analysis` | Analyser un DCE / RFP : exigences, critères, go/no-go | P1 | full | json_schema |
| `product.sheet` | Fiche produit / argumentaire | P1 | direct | judge |
| `crm.note` | Compte rendu de rendez-vous client → fiche CRM | P1 | direct | json_schema |

## Finance & comptabilité

| ID | Tâche | Prio | Thinking | Éval |
|---|---|---|---|---|
| `fin.dunning` | Relances d'impayés graduées | P0 | direct | judge |
| `fin.commentary` | Commentaire de reporting / écarts budgétaires | P1 | full | judge |
| `fin.accounting_qa` | Questions de comptabilité courante (PCG, TVA) avec prudence | P2 | full | judge |

## Gestion de projet & organisation

| ID | Tâche | Prio | Thinking | Éval |
|---|---|---|---|---|
| `pm.agenda` | Ordre du jour de réunion | P0 | direct | judge |
| `pm.plan` | Plan projet : jalons, RACI, risques | P1 | full | json_schema |
| `pm.user_stories` | User stories et critères d'acceptation | P1 | short | judge |
| `pm.status` | Point d'avancement / reporting hebdo | P0 | direct | judge |
| `pm.okr` | Formuler des objectifs / OKR | P2 | short | judge |

## Relation client & support

| ID | Tâche | Prio | Thinking | Éval |
|---|---|---|---|---|
| `support.reply` | Réponse à un ticket / réclamation client | P0 | direct | judge |
| `support.classify` | Classer un ticket (catégorie, urgence, sentiment) | P0 | direct | exact |
| `support.faq` | Rédiger une FAQ à partir de tickets | P1 | short | judge |

## Outils, skills & sorties structurées

| ID | Tâche | Prio | Thinking | Éval |
|---|---|---|---|---|
| `skill.route` | Choisir la bonne skill d'entreprise (ou aucune) dans le catalogue | P0 | direct | routing |
| `skill.follow` | Appliquer fidèlement le contenu d'une skill chargée | P0 | direct | judge |
| `skill.compose` | Combiner plusieurs skills (procédure + charte) | P1 | short | judge |
| `policy.obey` | Respecter une politique d'entreprise contre la demande utilisateur | P0 | direct | judge |
| `tool.call` | Appel d'outils (agenda, CRM, recherche) au format Qwen | P0 | short | exact |
| `json.output` | Sortie strictement conforme à un schéma JSON | P0 | direct | json_schema |

## Fiabilité, sécurité & conformité

| ID | Tâche | Prio | Thinking | Éval |
|---|---|---|---|---|
| `trust.idk` | Dire « je ne sais pas » / demander une précision plutôt qu'inventer | P0 | direct | judge |
| `trust.pii` | Masquer les données personnelles sensibles | P0 | direct | exact |
| `trust.injection` | Ignorer les instructions cachées dans un document / e-mail | P0 | direct | judge |
| `trust.lang` | Répondre dans la langue de l'utilisateur, sans mélange | P0 | direct | exact |
