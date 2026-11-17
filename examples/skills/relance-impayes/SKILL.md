---
name: relance-impayes
description: Rédige les relances de factures impayées selon la procédure de recouvrement de l'entreprise (3 niveaux).
when_to_use: L'utilisateur demande de relancer un client, une facture en retard ou un impayé.
type: procedure
lang: [fr, en]
version: 1.2.0
owner: direction-financiere
roles: [finance, commercial]
---
# Procédure de relance

1. **Niveau 1 (J+7)** : ton courtois, rappel du numéro de facture, du montant TTC et de l'échéance. Proposer un renvoi de la facture.
2. **Niveau 2 (J+21)** : ton ferme, rappeler les pénalités de retard et l'indemnité forfaitaire de 40 € (art. L441-10 du Code de commerce).
3. **Niveau 3 (J+45)** : mise en demeure, délai de 8 jours, mention d'une transmission au service contentieux.

Toujours : objet `Facture n° {numéro} – {niveau}`, signature « Service Comptabilité Clients », pas d'émoji.
