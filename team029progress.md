# ChessInsight Progress Report – Team 029

## 1. Introduction

ChessInsight aims to build an interactive visual analytics system that reveals how chess gameplay patterns—such as time usage, blunders, and position complexity—vary across skill levels, and to predict a player’s skill tier from their behavioral traces rather than raw Elo.[file:1] The motivation is that existing tools like online analysis boards explain *what* happened in a single game, but do not aggregate *how* thousands of players behave, nor why players at different skill levels make systematically different decisions under time pressure.[file:1]

Our system ingests a large-scale Lichess dataset, derives game- and player-level features, trains a skill-tier classifier, discovers player archetypes via clustering, and surfaces these results in a visualization-ready format for dashboards.[file:1] This report summarizes our progress halfway through the semester and outlines remaining work and risks, following the course progress-report guidelines.[file:2]

## 2. Problem definition

We address two tightly coupled problems:

1. **Behavior-aware skill inference.** Given a player’s moves, clock times, and engine evaluations across many games, infer a discrete skill tier (Beginner, Intermediate, Advanced, Expert) that reflects *behavioral quality* rather than just rating.[file:1]
2. **Discovery of behavioral archetypes.** Using aggregated behavioral features, cluster players into interpretable archetypes (e.g., time scramblers, positional grinders) that help coaches and players understand common patterns and failure modes.[file:1]

Formally, we model each player as a vector of aggregated statistics over their games (time usage per phase, position complexity, error rates, opening style), and we learn:

- A supervised mapping from features to skill tiers using labeled Elo buckets.
- An unsupervised mapping from features to behavioral clusters using variants of k-means and related algorithms.

The downstream visual interface will support tasks such as exploring where a player lives in the behavioral map, how their time usage compares to peers, and which archetypes correlate with faster improvement.[file:1]

## 3. Literature survey (current status)

We have completed an initial literature survey covering four themes, as documented in our proposal:[file:1]

- **Player modeling and skill prediction.** Maia Chess and follow-up work show that neural models can match human move choices at specific Elo ranges, while other studies use gradient boosting and CNN–LSTM models to estimate ratings and outcomes from moves and clock times.[file:1]
- **Time pressure and decision making.** Psychological and economic studies analyze how time pressure affects depth of search and risk-taking, informing our choice of time-usage features and time-trouble indicators.[file:1]
- **Clustering of playing styles.** Prior work clusters behavior in other games and quantifies opening complexity in chess, motivating our use of behavioral clustering and complexity features.[file:1]
- **Visual analytics for chess.** Existing systems focus on single-game visualizations or elite-player studies; none provide large-scale, population-level dashboards for mixed-strength online players.[file:1]

Before the final report, we plan to (1) expand the survey with a few more recent visual analytics and clustering papers, and (2) tighten the mapping from each paper’s contributions to specific design and modeling choices in ChessInsight.[file:2]

## 4. Proposed method and current implementation

Our proposed pipeline had four main components—data/feature processing, skill-tier classification, behavioral clustering, and interactive visualization.[file:1] As of this progress report, the first three components are implemented end-to-end, and the visualization layer has initial wireframes and static plots.

### 4.1 Data ingestion and preprocessing

- We ingest a 1-million-game Lichess PGN file and parse moves and clock times for both players.[cite:5]
- After cleaning and filtering, we retain **350,060** games suitable for analysis.[cite:5]
- We identify **238,200** unique player handles in the raw PGNs, and aggregate per-player feature vectors for **22,725** players who have sufficient game history.[cite:5]

### 4.2 Feature extraction

We extract **39 game-level features** and then aggregate to **30 player-level features** that are used by both the classifier and clustering modules.[cite:5] These include:

- Average move times and time variance in opening, middlegame, and endgame.
- Frequency of low-time moves and time-trouble episodes.
- Engine-derived position complexity, centipawn loss, blunder and mistake rates.
- Opening aggression, piece activity, and material imbalance frequencies.[cite:7]

The feature extractor writes Parquet files for efficient downstream processing and caching, enabling the full pipeline to run on a laptop in a few minutes.

### 4.3 Skill-tier classification

We implement a **random forest classifier** that predicts four discrete skill tiers (Beginner, Intermediate, Advanced, Expert) from behavioral features.[cite:6]

- We construct a labeled dataset with **328,961** samples and **18** selected features, balanced across tiers using SMOTE in the training set.[cite:6]
- The data is split into **230,272** training, **49,344** validation, and **49,345** test samples.[cite:6]
- Class distribution before resampling is skewed toward Advanced and Intermediate players, but all four tiers are well represented.[cite:6]

The training script saves the fitted model, feature importance table, confusion matrix, and a JSON summary of key metrics under `models/`.

### 4.4 Behavioral clustering

We aggregate player-level features and run behavioral clustering to discover archetypes:[cite:9][cite:10]

- Standardize features and apply PCA to 10 components, which explain ~85% of variance.[cite:9]
- Evaluate k in the range 3–7 with k-means using silhouette, Calinski–Harabasz, and Davies–Bouldin indices; k = 5 achieves the best silhouette score.[cite:3][cite:9]
- Run final k-means with **k = 5**, then compute t-SNE embeddings for 2D visualization.[cite:9]

We compute per-cluster statistics (size, Elo distribution, game counts, skill-tier mix, and key feature means) and store both CSV summaries and JSON metadata for the dashboard.[cite:9][cite:10]

### 4.5 Visualization assets

While the interactive dashboard is not yet implemented, we have generated several static visualizations that directly support the eventual UI:[cite:5]

- Overall skill-tier distribution plots.
- Time-usage heatmaps by game phase and tier.
- Confusion matrix and feature-importance bar charts for the classifier.
- 2D cluster embedding scatterplots and bar charts of cluster-level characteristics.

We also created a **dashboard wireframe** image that sketches the intended layout: a behavioral map, control panel for filters, and linked detail views for time usage and blunder statistics.[cite:5]

## 5. Evaluation (current results and planned work)

### 5.1 Skill-tier classification performance

Using the saved metrics and confusion matrix, the current random forest baseline achieves:[cite:6][cite:8]

- **Validation accuracy:** 42.3%.
- **Test accuracy:** 42.5% (exact tier).
- **Adjacent accuracy (±1 tier):** 55.8%.
- **Macro precision / recall / F1:** approximately 0.42–0.45, indicating balanced performance across tiers despite class imbalance.

Error analysis shows most mistakes occur between adjacent tiers (e.g., Intermediate vs. Advanced, Advanced vs. Expert), which is expected given fuzzy boundaries between skill levels.[cite:8] Beginners are recognized more reliably, with fewer predictions leaking into higher tiers.

**Planned evaluation work:**

- Compare random forests with gradient boosting (e.g., XGBoost or LightGBM) and possibly shallow neural networks using the same features.
- Add calibration curves and top-k tier accuracy (e.g., probability mass over the two most likely tiers) to better quantify uncertainty.
- Conduct ablation studies to understand the contribution of time-based features vs. engine-based accuracy metrics.

### 5.2 Clustering quality and interpretability

The current k-means model with k = 5 yields the following metrics:[cite:9]

- **Silhouette score:** 0.16.
- **Calinski–Harabasz index:** 3,283.
- **Davies–Bouldin index:** 1.49.

From the saved statistics, we observe the following archetypal patterns:[cite:9][cite:10]

- A **deliberate, intermediate** cluster (≈6% of players, Elo ≈ 1485) with long move times and moderate error rates.
- A large **solid advanced** cluster (≈34%, Elo ≈ 1560) with stable time usage and mostly Advanced players.
- Two **fast, high-Elo** clusters (≈37% and 22%, Elo ≈ 1796–1803) that frequently reach time trouble but still perform well.
- A small **deliberate expert** cluster (≈0.4%, Elo ≈ 1780) characterized by long think times and a high fraction of Advanced/Expert players.

We have updated the `name_clusters` function to produce **unique, semantically richer archetype names** (e.g., “Advanced Elite Speed Demon”, “Intermediate Developing Positional Grinder”) based on behavior, dominant skill tier, and Elo buckets, eliminating earlier name collisions.[cite:3][cite:9]

**Planned evaluation work:**

- Use the new `compare_clustering_methods` utility to benchmark k-means against Gaussian Mixture Models, hierarchical clustering, DBSCAN, and Birch with the same feature matrix.[cite:3]
- Investigate cluster stability under resampling and feature perturbations.
- Validate interpretability via qualitative inspection and, if time permits, informal user feedback from chess-playing classmates.

## 6. Conclusions, remaining work, and risks

### 6.1 Current status relative to plan

According to our original Gantt chart, by the midpoint we aimed to have a working classifier with ≥50% accuracy and initial clusters; by the final report we target an accuracy of ≥65% and a complete interactive dashboard.[file:1]

- We have fully implemented **data processing, feature extraction, classification, and clustering**, with all intermediate artifacts saved under `data/processed` and `models/`.[cite:3][cite:5]
- The classifier currently reaches ~42.5% exact-tier accuracy and 55.8% adjacent accuracy, which is slightly below the original 50% target but provides a realistic baseline.[cite:6]
- Clustering produces 5 interpretable archetypes with reasonable separation metrics and has been wired into the analysis pipeline and visualization assets.[cite:9]

Overall, we are on track in terms of pipeline completeness, but we must focus the second half of the semester on **improving model quality** and **building the interactive UI**.

### 6.2 Upcoming milestones

For the remainder of the semester, our priorities are:

1. **Model improvements.** Experiment with stronger classifiers and richer features, and refine clustering with alternative algorithms and hyperparameters.
2. **Dashboard implementation.** Implement a web-based dashboard (e.g., Flask/Plotly Dash or a React front end) that consumes the saved Parquet/JSON artifacts and reproduces the planned visual interactions.
3. **Evaluation and polish.** Perform systematic quantitative evaluation, add usage scenarios, and harden the codebase for reproducible runs.

### 6.3 Risks and mitigation

- **Model performance risk.** Achieving ≥65% skill-tier accuracy may be challenging given noisy behavioral labels. We will mitigate this by exploring more expressive models, feature engineering, and possibly redefining the task in terms of coarser tiers or regression to Elo buckets.[file:1]
- **Cluster interpretability risk.** Some clusters may remain hard to interpret or unstable under resampling. Our mitigation is to (a) compare multiple clustering methods, (b) tie names directly to transparent statistics (time usage, Elo, error rates), and (c) use the dashboard to surface explanations rather than only labels.[cite:3][cite:9]
- **Time and implementation risk.** Building a polished interactive dashboard can be time-consuming. We will scope the UI carefully: prioritize a small set of high-impact coordinated views (behavioral map, time-usage heatmap, archetype comparison) over many partial features.[file:2]

### 6.4 Effort statement

So far, all team members have contributed a similar amount of effort in literature review, data processing, modeling, and documentation. We expect effort to remain balanced as we move into model refinement and dashboard implementation.[file:2]
