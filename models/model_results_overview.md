# ChessInsight Dataset and Model Results Overview

This document summarizes the chess dataset used by ChessInsight and the current performance of the skill-tier classification and behavioral clustering models.

## Dataset summary

- **Source:** Lichess public database (PGN file `data_1m_games.pgn` containing 350,060 games).[cite:5]
- **Total games processed:** 350,060 classical and rapid games after cleaning.[cite:5]
- **Unique players with feature vectors:** 22,725 players aggregated at the player level.[cite:5]
- **Unique players in raw PGN:** 238,200, including many with only a few games.[cite:5]
- **Rating range covered:** Elo 600–3265, spanning beginners to titled players.[cite:5]

## Feature engineering

- **Game-level features:** 39 features capturing move counts, clock usage by phase (opening, middlegame, endgame), position complexity, accuracy, and blunder statistics.[cite:5][cite:7]
- **Player-level features:** 30 aggregated features per player (means, variances, and rates) derived from their games, used for clustering and skill prediction.[cite:5]
- **Key behavioral signals:**
  - Phase-wise average move times and time variance.
  - Frequency of playing under time pressure (low-time moves, time-trouble frequency).
  - Engine-based metrics such as position complexity, centipawn loss, and blunder/mistake rates.
  - Opening aggression, piece activity, and material-imbalance frequency.[cite:7]

These features allow the models to connect *how* players use their time and handle complex positions with both their Elo and their inferred skill tier.

## Skill-tier classification model

- **Goal:** Predict a player’s skill tier (Beginner, Intermediate, Advanced, Expert) from behavioral features rather than raw Elo, aligning with the proposal’s focus on interpretable skill prediction.[cite:1][cite:6]
- **Model:** Random forest classifier trained on 18 key behavioral features.[cite:6]
- **Data splits:**
  - Training: 230,272 samples.
  - Validation: 49,344 samples.
  - Test: 49,345 samples.[cite:6]
- **Class distribution (before rebalancing):**
  - Advanced: 124,261 players.
  - Intermediate: 95,896 players.
  - Expert: 66,666 players.
  - Beginner: 42,138 players.[cite:6]

### Performance

- **Validation accuracy:** 42.3%.[cite:6]
- **Test accuracy:** 42.5% (exact-tier accuracy).[cite:6]
- **Adjacent accuracy (±1 tier):** 55.8%, meaning over half of predictions land at most one tier away from the true label.[cite:6]
- **Macro F1-score:** 0.42, indicating roughly balanced performance across tiers despite class imbalance.[cite:6]

These numbers are consistent with the goal of capturing coarse-grained skill tiers from behavior rather than exact Elo prediction, and they establish a solid baseline for further model refinement.

### Most important features

Top random-forest feature importances reveal that the classifier is largely driven by time usage patterns and position complexity:[cite:7]

- `white_avg_time_middlegame` (0.16)
- `white_time_variance_opening` (0.12)
- `white_avg_time_opening` (0.12)
- `white_time_variance_middlegame` (0.11)
- `avg_position_complexity` (0.10)
- `white_avg_time_endgame` (0.09)
- `num_moves` (0.08)
- `white_time_variance_endgame` (0.07)
- `white_low_time_move_ratio` (0.05)
- `white_time_trouble_frequency` (0.03)

The remaining features—opening aggression, piece activity, centipawn loss, and blunder/mistake rates—contribute smaller but still meaningful signal.[cite:7]

### Error patterns (confusion matrix)

The confusion matrix highlights how errors concentrate between adjacent tiers:[cite:8]

- Many **Advanced** players are confused with **Intermediate** or **Expert**, reflecting the fuzzy boundary between these groups.
- **Beginner** predictions are more stable, with most true beginners correctly recognized and only a moderate spillover into Intermediate.
- **Expert** players are occasionally misclassified as Advanced, consistent with the fact that they share strong behavioral traits but differ in consistency and time management.

Overall, the model already captures meaningful structure in behavioral data, but there is room to improve separation between adjacent tiers through richer features or more expressive models.

## Behavioral clustering model

- **Goal:** Discover recurring behavioral archetypes from player-level feature vectors (time usage, errors, complexity, etc.) without using Elo directly as a label.[cite:5][cite:9]
- **Method:** K-means clustering on PCA-reduced, standardized player features.
- **Optimal cluster count:** 5 clusters, chosen using the silhouette score over k = 3–7.[cite:3][cite:9]
- **Dimensionality reduction:** PCA to 10 components explaining ~85% of variance, followed by t-SNE for 2D visualization in the dashboard.[cite:9]

### Clustering quality metrics

For k = 5 with k-means:[cite:9]

- **Silhouette score:** 0.16 (moderate separation; clusters overlap but are distinguishable).
- **Calinski–Harabasz index:** 3,283.
- **Davies–Bouldin index:** 1.49.

Given the noisy, real-world behavioral data and overlapping styles, these scores are reasonable and reflect that styles form soft clusters rather than cleanly separable groups.

### Archetypes from current run

Using the saved clustering statistics, the five clusters can be summarized as follows:[cite:9][cite:10]

- **Cluster 0 – Deliberate intermediates**  
  Roughly 5.9% of players, with average Elo ≈ 1485 and a mix of Intermediate and Advanced tiers. They think significantly longer than average in all phases, sacrifice speed for accuracy, and rarely play in severe time trouble.

- **Cluster 1 – Solid advanced players**  
  Roughly 34% of players, average Elo ≈ 1560. Time usage is moderate with lower variance, and the group is dominated by Advanced players with relatively few beginners or experts.

- **Cluster 2 – Fast high-level strivers**  
  About 37% of players, average Elo ≈ 1796. They play quickly, show higher time variance, and often reach time-pressure situations, but maintain relatively strong results, suggesting experienced fast players.

- **Cluster 3 – Deliberate high-level specialists**  
  A small but distinct cluster (~0.4% of players, average Elo ≈ 1780) who spend a lot of time especially early in the game and in complex positions, with a high share of Advanced and Expert players.

- **Cluster 4 – Fast advanced grinders**  
  Around 22% of players, average Elo ≈ 1803. They combine high Elo with fast play, especially in middlegames and endgames, and experience substantial time pressure but still perform well.

The updated `name_clusters` logic in `src/clustering.py` now assigns **unique archetype names** to clusters by combining behavioral traits (fast, deliberate, time-scrambler, accurate, tactical) with dominant skill tier and Elo bands (e.g., “Intermediate Developing Positional Grinder”, “Expert Elite Speed Demon”), removing prior name collisions such as multiple generic “Deliberate Player” or “Fast Player” clusters.[cite:3]

## How these results support the dashboard

- The **classification model** provides a behavioral skill tier that can be visualized per player and contrasted with raw Elo, enabling views such as “players who punch above their Elo” or “players whose time usage resembles higher tiers”.[cite:5][cite:6]
- The **clustering model** supports a 2D behavioral map and archetype-level statistics panels, where users can filter by cluster, inspect time-usage heatmaps, and compare blunder patterns across archetypes.[cite:3][cite:9][cite:10]

Together, these components realize the proposal’s goal of moving beyond single-game engine analysis toward **interpretable, population-level behavioral insights** for players, coaches, and researchers.[cite:1][cite:5]
