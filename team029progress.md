# ChessInsight Progress Report – Team 029

**Team Members:** Nareshkumar Prakash Kumar Jamnani, Pratik Narayan Gokhale, Maitreyi Mhaiskar, Kartik Dutt, Shashankkumar Shailendrakumar Mittal

---

## 1. Introduction

ChessInsight aims to build an interactive visual analytics system that reveals how chess gameplay patterns—such as time usage, blunders, and position complexity—vary across skill levels, and to predict a player's skill tier from their behavioral traces rather than raw Elo. Existing tools such as Lichess analysis boards and Chess.com focus on analyzing single games move-by-move using chess engines; they lack aggregated pattern analysis across thousands of games and cannot explain why players at different skill levels behave differently [1, 2, 3].

Our system ingests a large Lichess PGN dataset (350,060 games), derives game- and player-level features, trains a skill-tier classifier, discovers player archetypes via clustering, and surfaces these results in a visualization-ready format for an interactive dashboard. This report summarizes our progress halfway through the semester and outlines remaining work and risks, following the course progress-report guidelines.

## 2. Problem Definition

We address two tightly coupled problems:

1. **Behavior-aware skill inference.** Given a player's moves, clock times, and engine evaluations across many games, infer a discrete skill tier (Beginner, Intermediate, Advanced, Expert) that reflects *behavioral quality* rather than just rating [1, 2, 3].
2. **Discovery of behavioral archetypes.** Using aggregated behavioral features, cluster players into interpretable archetypes (for example, "time scramblers," "positional grinders") that help coaches and players understand common patterns and failure modes [7, 8].

Formally, we model each player as a vector of aggregated statistics over their games (time usage per phase, position complexity, error rates, opening style), and we learn:

- A supervised mapping from features to skill tiers using Elo-derived labels.
- An unsupervised mapping from features to behavioral clusters using k-means and related algorithms.

The downstream visual interface will support tasks such as exploring where a player lives in the behavioral map, how their time usage compares to peers, and which archetypes correlate with faster improvement [14, 15].

## 3. Literature Survey (Current Status)

We have completed an initial literature survey covering five themes:

- **Player modeling and skill prediction.** McIlroy-Young et al. [1] introduced Maia Chess, training neural networks to predict moves at specific Elo ranges (1100–1900). Their follow-up Maia-2 work [4] extended this with skill-aware attention mechanisms. Kaushik et al. [2] used gradient boosting to classify game outcomes with 83% accuracy, while Sharma et al. [3] applied CNN–LSTM architectures achieving a mean absolute error of 182 rating points. These works provide strong modeling approaches but do not offer an interactive user interface for exploring patterns.
- **Time pressure and decision making.** Van Harreveld et al. [5] showed that time pressure slows deep thinking but keeps quick pattern recognition intact. Künn et al. [6] showed that less thinking time leads players toward safer moves. These studies inform our choice of phase-wise time-usage and time-trouble features.
- **Clustering and playing styles.** Drachen et al. [7] and Sifa et al. [8] use clustering algorithms to group players by playing style across online and mobile games; these approaches remain unexplored in chess. Challet and Maillet [9] quantify the complexity and similarity of chess openings using graph-based algorithms, motivating our complexity features.
- **Behavioral stylometry and knowledge graphs.** McIlroy-Young et al. [10] use transformers to identify individual players from game sequences, demonstrating that behavior is fingerprinted in moves. A 2025 study [11] constructed a behavior-based knowledge graph to predict next moves. Both works lack visual interpretability.
- **Blunder prediction and time management.** Rokach and Shapira [12] propose a blunder-prediction model that operates per game rather than across aggregated populations. Guga et al. [13] use remaining time and position complexity to predict optimal think time per move, directly influencing our phase-wise time features.
- **Visual analytics for chess.** Lu and Wang [14] proposed a visual interactive tool for single-game evolution with linked views. García-Díaz and Mariscal-Quintero [15] propose visual tools to study elite players' decisions. Neither provides large-scale, population-level dashboards for mixed-strength online players.

Before the final report, we plan to expand the survey with additional visual analytics and clustering papers, and tighten the mapping from each paper's contributions to specific design and modeling choices in ChessInsight.

## 4. Proposed Method and Current Implementation

Our proposed pipeline has four main components—data and feature processing, skill-tier classification, behavioral clustering, and interactive visualization. As of this progress report, the first three components are implemented end-to-end, and the visualization layer has initial wireframes and static plots.

### 4.1 Data Ingestion and Preprocessing

- We ingest a Lichess PGN file containing **350,060** games and parse moves and clock times for both players.
- We identify **238,200** unique player handles in the raw PGNs and aggregate per-player feature vectors for **22,725** players who have sufficient game history.
- The full pipeline runs on a laptop in under ten minutes using cached chunk-based parsing.

### 4.2 Feature Extraction

We extract **39 game-level features** and aggregate them to **30 player-level features** used by both the classifier and clustering modules. These include:

- Average move times and time variance in opening, middlegame, and endgame [13].
- Frequency of low-time moves and time-trouble episodes [5, 6].
- Engine-derived position complexity, centipawn loss, blunder and mistake rates [12].
- Opening aggression, piece activity, and material imbalance frequencies [9].

### 4.3 Skill-Tier Classification

We implement a **random forest classifier** that predicts four discrete skill tiers (Beginner, Intermediate, Advanced, Expert) from behavioral features [2, 3].

- We construct a labeled dataset with **328,961** samples and **18** selected features, balanced across tiers using SMOTE in the training set.
- The data is split into **230,272** training, **49,344** validation, and **49,345** test samples.
- Class distribution before resampling is skewed toward Advanced and Intermediate players, but all four tiers are well represented.
- The fitted model, feature importance table, confusion matrix, and a summary of key metrics are saved for evaluation and reproducibility.

### 4.4 Behavioral Clustering

We aggregate player-level features and run behavioral clustering to discover archetypes [7, 8]:

- Standardize features and apply PCA to 10 components, which explain about 88% of variance in the current run.
- Evaluate k in the range 3–7 with k-means using silhouette, Calinski–Harabasz, and Davies–Bouldin indices; k = 5 achieves the best silhouette score among the tested values.
- Run final k-means with **k = 5**, then compute t-SNE embeddings for 2D visualization.
- Use the updated `name_clusters` logic to assign unique, behavior- and skill-aware archetype names to each cluster (for example, "Intermediate Deliberate Player", "Advanced Fast Player #2"), removing earlier label collisions.
- Per-cluster statistics (size, Elo distribution, game counts, skill-tier mix, and key feature means) are stored as CSV summaries and JSON metadata for the dashboard.

### 4.5 Visualization Assets

While the interactive dashboard is not yet implemented, we have generated several static visualizations that directly support the eventual UI [14, 15]:

- Overall skill-tier distribution plots.
- Time-usage heatmaps by game phase and tier.
- Confusion matrix and feature-importance bar charts for the classifier.
- 2D cluster embedding scatterplots and bar charts of cluster-level characteristics.
- A **dashboard wireframe** image sketching the intended layout: a behavioral map, a control panel for filters, and linked detail views for time usage and blunder statistics.

## 5. Evaluation (Current Results and Planned Work)

### 5.1 Skill-Tier Classification Performance

The current random forest baseline achieves:

- **Validation accuracy:** approximately 42.3%.
- **Test accuracy:** **42.6%** (exact tier).
- **Adjacent accuracy (±1 tier):** **55.6%**.
- **Macro F1:** approximately **0.425**.

Error analysis shows most mistakes occur between adjacent tiers (e.g., Intermediate vs. Advanced, Advanced vs. Expert), which is expected given fuzzy boundaries between skill levels. Beginners are recognized more reliably, with fewer predictions leaking into higher tiers.

**Planned evaluation work:**

- Compare random forests with gradient boosting (XGBoost or LightGBM) and possibly shallow neural networks using the same features [2, 3].
- Add calibration curves and top-k tier accuracy to better quantify prediction uncertainty.
- Conduct ablation studies to understand the contribution of time-based features [5, 6, 13] versus engine-based accuracy metrics [12].

### 5.2 Clustering Quality and Interpretability

The current k-means model with k = 5 yields:

- **Silhouette score:** **0.21**.
- **Calinski–Harabasz index:** **3440.3**.
- **Davies–Bouldin index:** **1.35**.

We observe several archetypal patterns, such as an **Intermediate Deliberate Player** cluster and multiple **Advanced Fast Player** clusters with slightly different Elo bands and sizes, capturing the trade-off between speed and deliberation at higher skill levels [7, 8].

We have updated the `name_clusters` function to produce **unique, semantically richer archetype names** based on behavior and dominant skill tier, eliminating earlier name collisions.

We also integrated a `compare_clustering_methods` helper that benchmarks k-means against Gaussian Mixture Models, hierarchical clustering, DBSCAN, and Birch on the same feature matrix. From the latest comparison, **Birch** achieves the highest silhouette score (≈0.29) and the lowest Davies–Bouldin index (≈1.02), outperforming k-means on these internal metrics, while k-means attains the highest Calinski–Harabasz score. We keep k-means as the primary pipeline method for now because it is simple and well-understood, and treat Birch as a promising alternative to explore further in the final phase.

**Planned evaluation work:**

- Use the comparison results and additional hyperparameter sweeps to decide whether Birch or another alternative offers better separation or interpretability than k-means [7, 8].
- Investigate cluster stability under resampling and feature perturbations.
- Validate interpretability via qualitative inspection and, if time permits, informal user feedback from chess-playing classmates [14, 15].

## 6. Conclusions, Remaining Work, and Risks

### 6.1 Current Status Relative to Plan

According to our original Gantt chart, by the midpoint we aimed to have a working classifier with >50% accuracy plus initial clusters; by the final report we target at least 65% classification accuracy and a complete interactive dashboard.

- We have fully implemented **data processing, feature extraction, classification, and clustering**, with all intermediate artifacts saved for reproducibility.
- The classifier currently reaches about **42.6%** exact-tier accuracy and **55.6%** adjacent accuracy, which is slightly below the original 50% midterm target but provides a realistic and well-characterized baseline.
- Clustering produces 5 interpretable archetypes with reasonable separation metrics and has been wired into the analysis pipeline and visualization assets.

Overall, we are on track in terms of pipeline completeness, but we must focus the second half of the semester on **improving model quality** and **building the interactive UI**.

### 6.2 Upcoming Milestones

For the remainder of the semester, our priorities are:

1. **Model improvements.** Experiment with stronger classifiers and richer features, and refine clustering with alternative algorithms and hyperparameters.
2. **Dashboard implementation.** Implement a web-based interactive dashboard consuming the saved artifacts and reproducing the planned visual interactions described in the proposal.
3. **Evaluation and polish.** Perform systematic quantitative evaluation, add usage scenarios, and harden the codebase for reproducible runs.

### 6.3 Risks and Mitigation

- **Model performance risk.** Achieving ≥65% skill-tier accuracy may be challenging given noisy behavioral labels. We will mitigate this by exploring more expressive models, richer feature engineering, and possibly redefining the task as regression to Elo or a coarser tier structure [2, 3].
- **Cluster interpretability risk.** Some clusters may remain hard to interpret or unstable under resampling. Our mitigation is to (a) compare multiple clustering methods, (b) tie archetype names directly to transparent statistics (time usage, Elo, error rates), and (c) use the dashboard to surface explanations alongside labels [7, 8, 14].
- **Time and implementation risk.** Building a polished interactive dashboard is time-consuming. We will scope the UI carefully: prioritize a small set of high-impact coordinated views (behavioral map, time-usage heatmap, archetype comparison) over many partial features [14, 15].

### 6.4 Effort Statement

So far, all team members have contributed a similar amount of effort in literature review, data processing, modeling, and documentation. We expect effort to remain balanced as we move into model refinement and dashboard implementation.

---

## References

1. McIlroy-Young, R., et al. "Aligning Superhuman AI with Human Behavior: Chess as a Model System." *KDD*, 2020. <https://www.cs.toronto.edu/~ashton/pubs/maia-kdd2020.pdf>

2. Kaushik, A., et al. "Machine Learning Approaches for Classifying Chess Game Outcomes." *Electronics*, 2025. <https://www.mdpi.com/2079-9292/15/1/1>

3. Perez, J., et al. "Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM." *arXiv*, 2024. <https://arxiv.org/abs/2409.11506>

4. Tang, Z., et al. "Maia-2: A Unified Model for Human–AI Alignment in Chess." *NeurIPS*, 2024. <https://www.cs.toronto.edu/~ashton/pubs/maia2-neurips2024.pdf>

5. van Harreveld, F., et al. "The Effects of Time Pressure on Chess Skill." *Psychological Science*, 2007. <https://pubmed.ncbi.nlm.nih.gov/17186308/>

6. Künn, S., et al. "Time Pressure and Strategic Risk-Taking in Professional Chess." *Journal of Economic Behavior & Organization*, 2025. <https://www.sciencedirect.com/science/article/pii/S0167268125003373>

7. Drachen, A., et al. "Clustering of Player Behavior in Computer Games in the Wild." *CIG*, 2012. <https://andersdrachen.files.wordpress.com/2014/07/gunsswordsanddataclustering-of-player-behavior-in-computer-games-in-the-wild_cig2012.pdf>

8. Sifa, R., et al. "Clustering Mixed-Type Player Behavior Data for Churn Prediction." *Central European Journal of Operations Research*, 2022. <https://link.springer.com/article/10.1007/s10100-022-00802-8>

9. Challet, D., & Maillet, M. "Quantifying the Complexity and Similarity of Chess Openings." *Nature Scientific Reports*, 2023. <https://www.nature.com/articles/s41598-023-31658-w>

10. McIlroy-Young, R., et al. "Detecting Individual Decision-Making Style: Exploring Behavioral Stylometry in Chess." *NeurIPS*, 2021. <https://proceedings.neurips.cc/paper_files/paper/2021/hash/ccf8111910291ba472b385e9c5f59099-Abstract.html>

11. Skidanov, B., et al. "A Behavior-Based Knowledge Representation Improves Prediction in Chess." *arXiv*, 2025. <https://arxiv.org/html/2504.05425v1>

12. Rokach, Y., & Shapira, B. "Blunder Prediction in Chess." *Applied Intelligence*, 56(4), 2026. <https://doi.org/10.1007/s10489-026-07131-2>

13. Guga, J., et al. "Time Management in a Chess Game Through Machine Learning." *CIS, Temple University*. <https://cis.temple.edu/~wu/research/publications/Publication_files/Paper_Guga.pdf>

14. Lu, M., & Wang, G. "Chess Evolution Visualization." *IEEE VIS*, 2014. <https://pubmed.ncbi.nlm.nih.gov/26357293/>

15. García-Díaz, P. R., & Mariscal-Quintero, R. "Visual Analytics as a Cognitive Microscope in Elite Chess." *TechRxiv*, February 17, 2026. DOI: <https://doi.org/10.36227/techrxiv.177130687.71691368/v1>
