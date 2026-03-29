# Proposal to Code Mapping - Quick Reference
## Team 029 - ChessInsight

---

## Heilmeier Questions Mapping

### Q1: What are we trying to do?

| Proposal Statement | Code Implementation | File:Line |
|-------------------|---------------------|-----------|
| "Build an interactive visual analytics system" | Static visualizations only | `visualizations.py` |
| "Show how time usage varies across skill levels" | ✅ Time heatmap by tier | `visualizations.py:62-106` |
| "Show how blunders vary across skill levels" | ✅ Accuracy by tier chart | `visualizations.py:281-324` |
| "Predict player's skill from game behavior" | ✅ RF/XGBoost classifier | `classifier.py:61-207` |

---

### Q2: How is it done today and limitations?

| Current Limitation (Proposal) | Our Solution | Code Evidence |
|------------------------------|--------------|---------------|
| "Single-game engine analysis only" | Aggregate across thousands | `feature_extractor.py:409-485` |
| "No patterns across games" | Player-level aggregation | `aggregate_player_features()` |
| "Cannot explain WHY different levels behave differently" | Cluster archetypes + feature importance | `clustering.py:233-329` |

---

### Q3: What's new in your approach?

| Innovation (Proposal) | Implementation | Status |
|----------------------|----------------|--------|
| **Skill Tier Classification** | Random Forest + XGBoost | ✅ `classifier.py` |
| Features: time usage by phase | 8 time features | ✅ `config.py:56-60` |
| Features: blunder rate (complexity-adjusted) | 4 accuracy features | ✅ `config.py:62-64` |
| Features: opening aggression | 2 opening features | ✅ `config.py:70-72` |
| **Behavioral Pattern Clustering** | K-means with auto-naming | ✅ `clustering.py` |
| Archetypes: "time scrambler" | Auto-detected | ✅ `clustering.py:305` |
| Archetypes: "positional grinder" | Auto-detected | ✅ `clustering.py:303` |
| **2D Cluster Map (t-SNE)** | t-SNE embedding | ✅ `clustering.py:155-158` |
| **Time-Usage Heatmaps** | Seaborn heatmap | ✅ `visualizations.py:62-106` |
| **Opening Network Graph** | Wireframe mockup only | ❌ `visualizations.py:419-445` |
| **Interactive Dashboard** | Static matplotlib only | ❌ Not implemented |

---

### Q4: Who cares?

| Stakeholder (Proposal) | How Code Serves Them |
|------------------------|---------------------|
| Chess players | Skill classification + archetype identification |
| Chess coaches | Feature importance shows what to focus on |
| Analysts/researchers | Cluster statistics + pattern analysis |

---

### Q5: If successful, what difference will it make?

| Success Metric (Proposal) | Code Tracking | Status |
|--------------------------|---------------|--------|
| Classification accuracy >65% | `metrics['test_accuracy']` | ⚠️ Pending verification |
| Adjacent accuracy (±1 tier) | `metrics['adjacent_accuracy']` | ✅ Tracked |
| Interpretable clusters | `cluster_names` dict | ✅ Auto-generated |

---

### Q6: What are the risks?

| Risk (Proposal) | Mitigation in Code |
|-----------------|-------------------|
| "Features may not distinguish tiers" | Multiple feature categories + SMOTE | `classifier.py:102-111` |
| "Clusters hard to interpret" | Auto-naming based on characteristics | `clustering.py:233-329` |
| "Visualization overwhelms users" | Dashboard wireframe design | `visualizations.py:384-483` |

---

### Q9: Milestones to check for success?

| Milestone | Target | Code Evidence | Status |
|-----------|--------|---------------|--------|
| **Midterm** | | | |
| Working classifier | >50% accuracy | `train_classifier()` | ✅ |
| Initial clusters | Identified | `name_clusters()` | ✅ |
| **Final** | | | |
| Interactive dashboard | Complete | `create_dashboard_wireframe()` | ❌ Mockup only |
| Classification accuracy | ≥65% | `metrics['test_accuracy']` | ⚠️ Pending |

---

## Feature Engineering Traceability

### Time Features (8 total)

| Feature Name | Proposal Section | Code Location | Calculation |
|-------------|------------------|---------------|-------------|
| `avg_time_opening` | "time usage in different phases" | `feature_extractor.py:84` | Mean time for moves 1-10 |
| `avg_time_middlegame` | "time usage in different phases" | `feature_extractor.py:85` | Mean time for moves 11-25 |
| `avg_time_endgame` | "time usage in different phases" | `feature_extractor.py:86` | Mean time for moves 26+ |
| `time_variance_opening` | Implied by time analysis | `feature_extractor.py:87` | Variance in opening |
| `time_variance_middlegame` | Implied by time analysis | `feature_extractor.py:88` | Variance in middlegame |
| `time_variance_endgame` | Implied by time analysis | `feature_extractor.py:89` | Variance in endgame |
| `low_time_move_ratio` | "time trouble" behavior | `feature_extractor.py:90` | % moves under 30s |
| `time_trouble_frequency` | "time trouble" behavior | `feature_extractor.py:91` | % moves under 10% time |

### Accuracy Features (4 total)

| Feature Name | Proposal Section | Code Location | Calculation |
|-------------|------------------|---------------|-------------|
| `blunder_rate` | "number of blunders made" | `feature_extractor.py:142` | Blunders / total moves |
| `mistake_rate` | Implied by accuracy | `feature_extractor.py:143` | Mistakes / total moves |
| `avg_centipawn_loss` | Implied by accuracy | `feature_extractor.py:144` | Mean centipawn loss |
| `accuracy_percentage` | Implied by accuracy | `feature_extractor.py:145` | Overall accuracy % |

### Complexity Features (3 total)

| Feature Name | Proposal Section | Code Location | Calculation |
|-------------|------------------|---------------|-------------|
| `avg_position_complexity` | "move complexity" | `feature_extractor.py:206` | Tactical moves % |
| `material_imbalance_freq` | Implied by complexity | `feature_extractor.py:210` | Imbalance frequency |
| `piece_activity_score` | Implied by complexity | `feature_extractor.py:213` | Activity heuristic |

### Opening Features (2 total)

| Feature Name | Proposal Section | Code Location | Calculation |
|-------------|------------------|---------------|-------------|
| `opening_aggression_score` | "aggression of opening moves" | `feature_extractor.py:246-255` | ECO-based score |
| `book_deviation_move` | Implied by opening | `feature_extractor.py:261-262` | Move # of deviation |

---

## Visualization Traceability

| Proposal Visualization | Mockup (Slide 5) | Code Function | Output File | Status |
|----------------------|------------------|---------------|-------------|--------|
| 2D Cluster Map (t-SNE) | Top-left scatter | `plot_cluster_embedding()` | `cluster_embedding.png` | ✅ t-SNE |
| Time-Usage Heatmap | Middle-left grid | `plot_time_heatmap()` | `time_heatmap.png` | ✅ |
| Opening Network Graph | Bottom-left network | In wireframe only | `dashboard_wireframe.png` | ❌ |
| Chessboard Heatmap | Middle-right board | Not implemented | - | ❌ |
| Feature Importance | Not in mockup | `plot_feature_importance()` | `feature_importance.png` | ✅ |
| Confusion Matrix | Not in mockup | `plot_confusion_matrix()` | `confusion_matrix.png` | ✅ |

---

## Architecture Traceability

### Proposal System Architecture (Slide 4)

```
Lichess Data → Feature Eng. → Model (Classification/Clustering) → Insights → Dashboard → Users
```

### Code Implementation

```
data_loader.py → feature_extractor.py → classifier.py    → visualizations.py → [Dashboard TBD]
                                      → clustering.py   →
```

| Architecture Component | Proposal | Code Module | Status |
|-----------------------|----------|-------------|--------|
| Lichess Data | "4B+ games, PGN + Clock" | `data_loader.py` | ✅ |
| Feature Engineering | "Time, Blunders, Openings" | `feature_extractor.py` | ✅ |
| Classification | "Skill Tier" | `classifier.py` | ✅ |
| Clustering | "Archetypes" | `clustering.py` | ✅ |
| Visual Analysis | "Linked views, Interactive" | `visualizations.py` | ⚠️ Static |
| Dashboard | "D3.js/Streamlit" | Not implemented | ❌ |

---

## Gap Summary (Updated March 29, 2026)

| Gap | Proposal Reference | Required Action | Status |
|-----|-------------------|-----------------|--------|
| **Interactive Dashboard** | Slides 4-5 | Implement Streamlit app | ❌ Pending |
| **Opening Network Graph** | Slide 5 | Implement D3.js/Plotly graph | ❌ Pending |
| ~~**t-SNE Embedding**~~ | ~~Slide 5~~ | ~~Replace PCA with t-SNE~~ | ✅ FIXED |
| ~~**4 Skill Tiers**~~ | ~~Slide 4~~ | ~~Update config.py~~ | ✅ FIXED |
| **Chessboard Heatmap** | Slide 5 | Implement blunder location viz | ❌ Pending |

---

*Quick Reference Document - Team 029*
*Last Updated: March 29, 2026*
