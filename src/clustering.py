"""
Behavioral Clustering Module - Identify player archetypes
Team 029 - CSE6242 Spring 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, N_CLUSTERS_RANGE


def prepare_clustering_data(player_features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare player-level features for clustering.

    Args:
        player_features: DataFrame with player-level aggregated features

    Returns:
        Tuple of (feature matrix, list of feature names)
    """
    # Select numeric feature columns (excluding identifiers)
    exclude_cols = ['player', 'num_games', 'skill_tier', 'avg_elo']
    feature_cols = [c for c in player_features.columns
                    if c not in exclude_cols and
                    player_features[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    X = player_features[feature_cols].copy()

    # Handle missing values
    X = X.fillna(X.mean())

    # Remove constant columns
    constant_cols = X.columns[X.std() == 0]
    if len(constant_cols) > 0:
        print(f"Removing constant columns: {list(constant_cols)}")
        X = X.drop(columns=constant_cols)

    return X, list(X.columns)


def find_optimal_k(X: np.ndarray,
                   k_range: Tuple[int, int] = N_CLUSTERS_RANGE) -> Dict:
    """
    Find optimal number of clusters using multiple metrics.

    Args:
        X: Scaled feature matrix
        k_range: Range of k values to try

    Returns:
        Dictionary with evaluation metrics for each k
    """
    results = {
        'k_values': [],
        'inertia': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }

    print("Evaluating cluster counts...")
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X)

        results['k_values'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(X, labels))
        results['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
        results['davies_bouldin'].append(davies_bouldin_score(X, labels))

        print(f"  k={k}: Silhouette={results['silhouette'][-1]:.4f}, "
              f"CH={results['calinski_harabasz'][-1]:.1f}")

    # Find best k based on silhouette score
    best_idx = np.argmax(results['silhouette'])
    results['optimal_k'] = results['k_values'][best_idx]
    print(f"\nOptimal k based on silhouette score: {results['optimal_k']}")

    return results


def perform_clustering(X: pd.DataFrame,
                       n_clusters: int = 5,
                       method: str = 'kmeans',
                       compute_embedding: bool = True) -> Dict:
    """
    Perform clustering on player features.

    Args:
        X: Feature matrix (DataFrame)
        n_clusters: Number of clusters
        method: Clustering method ('kmeans', 'hierarchical', 'dbscan', 'gmm', 'birch')
        compute_embedding: Whether to compute a 2D t-SNE embedding (expensive)

    Returns:
        Dictionary with clustering results
    """
    print(f"Performing {method} clustering with k={n_clusters}...")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(10, X.shape[1]), random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"PCA: {pca.n_components_} components, {explained_variance:.1%} variance explained")

    # Perform clustering
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_pca)
        cluster_centers = model.cluster_centers_

    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_pca)
        cluster_centers = None

    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(X_pca)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_centers = None
        print(f"DBSCAN found {n_clusters} clusters")

    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_pca)
        cluster_centers = model.means_

    elif method == 'birch':
        model = Birch(n_clusters=n_clusters)
        labels = model.fit_predict(X_pca)
        cluster_centers = getattr(model, 'subcluster_centers_', None)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Calculate metrics
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X_pca, labels)
        calinski = calinski_harabasz_score(X_pca, labels)
        davies = davies_bouldin_score(X_pca, labels)
    else:
        silhouette = calinski = davies = 0

    print("Clustering metrics:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Calinski-Harabasz Index: {calinski:.1f}")
    print(f"  Davies-Bouldin Index: {davies:.4f}")

    # Get 2D embedding for visualization using t-SNE (per proposal)
    embedding_2d = None
    if compute_embedding:
        print("Computing t-SNE embedding for visualization...")
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, len(X_pca) - 1))
        embedding_2d = tsne.fit_transform(X_pca)
        print("t-SNE embedding complete")

    results = {
        'model': model,
        'scaler': scaler,
        'pca': pca,
        'method': method,
        'n_clusters': n_clusters,
        'labels': labels,
        'cluster_centers': cluster_centers,
        'feature_columns': list(X.columns),
        'metrics': {
            'silhouette_score': silhouette,
            'calinski_harabasz_index': calinski,
            'davies_bouldin_index': davies,
            'pca_explained_variance': explained_variance
        },
        'embedding_2d': embedding_2d,
        'embedding_method': 'tsne' if compute_embedding else None,
        'X_scaled': X_scaled,
        'X_pca': X_pca
    }

    return results


def compare_clustering_methods(X: pd.DataFrame,
                               n_clusters: int,
                               methods: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Run multiple clustering algorithms on the same feature matrix and compare metrics.

    This helper is intended for experimentation when exploring alternative
    behavioral clustering techniques beyond k-means.
    """
    if methods is None:
        methods = ['kmeans', 'hierarchical', 'dbscan', 'gmm', 'birch']

    rows: List[Dict] = []
    for method in methods:
        try:
            res = perform_clustering(X, n_clusters=n_clusters, method=method, compute_embedding=False)
            m = res['metrics']
            rows.append({
                'method': method,
                'n_clusters': res['n_clusters'],
                'silhouette_score': m['silhouette_score'],
                'calinski_harabasz_index': m['calinski_harabasz_index'],
                'davies_bouldin_index': m['davies_bouldin_index'],
                'pca_explained_variance': m['pca_explained_variance'],
            })
        except Exception as e:
            print(f"Skipping method {method} due to error: {e}")

    return pd.DataFrame(rows)


def analyze_clusters(player_features: pd.DataFrame,
                     labels: np.ndarray,
                     feature_columns: List[str]) -> pd.DataFrame:
    """
    Analyze cluster characteristics to identify archetypes.

    Args:
        player_features: Original player features DataFrame
        labels: Cluster labels
        feature_columns: List of feature column names used

    Returns:
        DataFrame with cluster statistics
    """
    df = player_features.copy()
    df['cluster'] = labels

    cluster_stats = []

    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1:  # DBSCAN noise points
            continue

        cluster_data = df[df['cluster'] == cluster_id]

        stats = {
            'cluster': cluster_id,
            'size': len(cluster_data),
            'pct_of_total': len(cluster_data) / len(df) * 100,
            'avg_elo': cluster_data['avg_elo'].mean() if 'avg_elo' in cluster_data.columns else np.nan,
            'avg_games': cluster_data['num_games'].mean() if 'num_games' in cluster_data.columns else np.nan,
        }

        # Skill tier distribution (4 tiers per proposal)
        if 'skill_tier' in cluster_data.columns:
            tier_dist = cluster_data['skill_tier'].value_counts(normalize=True)
            for tier in ['Beginner', 'Intermediate', 'Advanced', 'Expert']:
                stats[f'pct_{tier.lower()}'] = tier_dist.get(tier, 0) * 100

        # Key feature statistics
        for col in feature_columns[:10]:  # Top 10 features
            if col in cluster_data.columns:
                stats[f'{col}_mean'] = cluster_data[col].mean()

        cluster_stats.append(stats)

    return pd.DataFrame(cluster_stats)


def name_clusters(cluster_stats: pd.DataFrame,
                  player_features: pd.DataFrame,
                  labels: np.ndarray) -> Dict[int, Dict]:
    """
    Automatically name clusters based on their characteristics.

    Args:
        cluster_stats: DataFrame with cluster statistics
        player_features: Original player features
        labels: Cluster labels

    Returns:
        Dictionary mapping cluster ID to name and description. Names are
        guaranteed to be unique across clusters by incorporating skill tier
        and Elo-level qualifiers when needed.
    """
    df = player_features.copy()
    df['cluster'] = labels

    cluster_names: Dict[int, Dict] = {}

    # Pre-compute Elo buckets so we can add semantic qualifiers
    elo_q1 = df['avg_elo'].quantile(0.25) if 'avg_elo' in df.columns else None
    elo_q3 = df['avg_elo'].quantile(0.75) if 'avg_elo' in df.columns else None

    used_names = set()

    # Define naming criteria based on feature patterns
    for _, row in cluster_stats.iterrows():
        cluster_id = int(row['cluster'])
        cluster_data = df[df['cluster'] == cluster_id]

        # Analyze characteristics
        characteristics: List[str] = []
        base_name = f"Cluster {cluster_id}"
        description = ""

        # Check time management patterns
        time_cols = [c for c in cluster_data.columns if 'time' in c.lower() and 'mean' in c]
        if time_cols:
            avg_time = cluster_data[time_cols].mean().mean()
            overall_avg = df[time_cols].mean().mean()

            if avg_time < overall_avg * 0.7:
                characteristics.append("fast")
            elif avg_time > overall_avg * 1.3:
                characteristics.append("deliberate")

        # Check accuracy patterns
        blunder_cols = [c for c in cluster_data.columns if 'blunder' in c.lower()]
        if blunder_cols:
            avg_blunder = cluster_data[blunder_cols].mean().mean()
            overall_avg_blunder = df[blunder_cols].mean().mean()

            if avg_blunder < overall_avg_blunder * 0.7:
                characteristics.append("accurate")
            elif avg_blunder > overall_avg_blunder * 1.3:
                characteristics.append("tactical")

        # Check time trouble frequency
        trouble_cols = [c for c in cluster_data.columns if 'trouble' in c.lower() or 'low_time' in c.lower()]
        if trouble_cols:
            avg_trouble = cluster_data[trouble_cols].mean().mean()
            overall_avg_trouble = df[trouble_cols].mean().mean()

            if avg_trouble > overall_avg_trouble * 1.5:
                characteristics.append("time-scrambler")

        # Check skill distribution
        dominant_tier_label: Optional[str] = None
        if 'skill_tier' in cluster_data.columns:
            dominant_tier = cluster_data['skill_tier'].mode()
            if len(dominant_tier) > 0:
                dominant_tier_label = dominant_tier.iloc[0]
                characteristics.append(dominant_tier_label.lower())

        # Generate archetype name based on characteristics
        archetype_names = {
            ('fast', 'accurate'): ('Speed Demon', 'Fast, accurate players who maintain quality under time pressure'),
            ('fast', 'tactical'): ('Blitz Attacker', 'Aggressive players who play quickly but make tactical errors'),
            ('deliberate', 'accurate'): ('Positional Grinder', 'Careful, methodical players who take their time'),
            ('deliberate', 'tactical'): ('Deep Thinker', 'Players who think long but still make mistakes'),
            ('time-scrambler',): ('Time Scrambler', 'Players who frequently get into time trouble'),
            ('accurate',): ('Steady Hand', 'Consistent players with low error rates'),
            ('tactical',): ('Risk Taker', 'Players who make more mistakes but play aggressively'),
        }

        # Use up to two behavioral traits (ignoring skill tiers for archetype key)
        behavior_traits = [t for t in characteristics if t not in ['beginner', 'intermediate', 'advanced', 'expert']]
        char_tuple = tuple(behavior_traits[:2]) if len(behavior_traits) >= 1 else tuple()

        if char_tuple in archetype_names:
            base_name, description = archetype_names[char_tuple]
        elif len(behavior_traits) > 0:
            base_name = f"{behavior_traits[0].title()} Player"
            description = f"Players characterized by {', '.join(behavior_traits)} play style"
        else:
            base_name = f"Archetype {cluster_id + 1}"
            description = "Distinct player group with unique behavioral patterns"

        # Add Elo-based qualifier
        elo_label: Optional[str] = None
        if elo_q1 is not None and elo_q3 is not None and pd.notna(row['avg_elo']):
            if row['avg_elo'] >= elo_q3:
                elo_label = "Elite"
            elif row['avg_elo'] <= elo_q1:
                elo_label = "Developing"

        # Build final name with skill tier and Elo qualifiers
        qualifiers: List[str] = []
        if dominant_tier_label is not None:
            qualifiers.append(dominant_tier_label)
        if elo_label is not None:
            qualifiers.append(elo_label)

        full_name = f"{' '.join(qualifiers)} {base_name}".strip() if qualifiers else base_name

        # Ensure uniqueness across clusters
        if full_name in used_names:
            suffix = 2
            candidate = f"{full_name} #{suffix}"
            while candidate in used_names:
                suffix += 1
                candidate = f"{full_name} #{suffix}"
            full_name = candidate

        used_names.add(full_name)

        # Enrich description with qualifiers
        qualifier_phrases = []
        if dominant_tier_label is not None:
            qualifier_phrases.append(f"dominant skill tier {dominant_tier_label}")
        if elo_label is not None:
            qualifier_phrases.append(f"{elo_label.lower()} Elo range")
        if qualifier_phrases:
            description = f"{description} (typically {', '.join(qualifier_phrases)})."

        cluster_names[cluster_id] = {
            'name': full_name,
            'description': description,
            'characteristics': characteristics,
            'size': int(row['size']),
            'avg_elo': float(row['avg_elo']) if pd.notna(row['avg_elo']) else None
        }

    return cluster_names


def save_clustering_results(results: Dict,
                            cluster_stats: pd.DataFrame,
                            cluster_names: Dict,
                            model_name: str = 'player_clustering'):
    """
    Save clustering results to disk.

    Args:
        results: Dictionary from perform_clustering
        cluster_stats: DataFrame with cluster statistics
        cluster_names: Dictionary with cluster names
        model_name: Name for saved files
    """
    # Save model
    model_path = MODELS_DIR / f"{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': results['model'],
            'scaler': results['scaler'],
            'pca': results['pca'],
            'feature_columns': results['feature_columns']
        }, f)
    print(f"Saved clustering model to {model_path}")

    # Save metrics and names
    metrics_path = MODELS_DIR / f"{model_name}_results.json"
    results_to_save = {
        'method': results['method'],
        'n_clusters': results['n_clusters'],
        'metrics': results['metrics'],
        'cluster_names': cluster_names
    }
    with open(metrics_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Saved clustering results to {metrics_path}")

    # Save cluster statistics
    stats_path = MODELS_DIR / f"{model_name}_statistics.csv"
    cluster_stats.to_csv(stats_path, index=False)
    print(f"Saved cluster statistics to {stats_path}")

    # Save embeddings for visualization
    embedding_path = PROCESSED_DATA_DIR / f"{model_name}_embeddings.parquet"
    embedding_df = pd.DataFrame({
        'x': results['embedding_2d'][:, 0],
        'y': results['embedding_2d'][:, 1],
        'cluster': results['labels']
    })
    embedding_df.to_parquet(embedding_path)
    print(f"Saved embeddings to {embedding_path}")


def print_clustering_summary(cluster_stats: pd.DataFrame,
                             cluster_names: Dict,
                             metrics: Dict):
    """Print a formatted summary of clustering results."""
    print("\n" + "=" * 60)
    print("CLUSTERING RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nClustering Metrics:")
    print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.1f}")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")

    print(f"\nIdentified {len(cluster_names)} Player Archetypes:")
    print("-" * 60)

    for cluster_id, info in sorted(cluster_names.items()):
        print(f"\n  Cluster {cluster_id}: {info['name']}")
        print(f"    Size: {info['size']} players")
        if info['avg_elo']:
            print(f"    Avg Elo: {info['avg_elo']:.0f}")
        print(f"    Description: {info['description']}")
        if info['characteristics']:
            print(f"    Characteristics: {', '.join(info['characteristics'])}")


if __name__ == "__main__":
    print("ChessInsight Behavioral Clustering")
    print("=" * 50)

    # Load player features
    player_path = PROCESSED_DATA_DIR / "player_features.parquet"

    if player_path.exists():
        player_features = pd.read_parquet(player_path)
        print(f"Loaded features for {len(player_features)} players")

        # Prepare data
        X, feature_cols = prepare_clustering_data(player_features)
        print(f"Using {len(feature_cols)} features for clustering")

        # Find optimal k
        k_results = find_optimal_k(X.values, k_range=(3, 7))
        optimal_k = k_results['optimal_k']

        # Perform clustering with the primary method (k-means)
        results = perform_clustering(X, n_clusters=optimal_k, method='kmeans')

        # Analyze clusters
        cluster_stats = analyze_clusters(player_features, results['labels'], feature_cols)

        # Name clusters
        cluster_names = name_clusters(cluster_stats, player_features, results['labels'])

        # Print summary
        print_clustering_summary(cluster_stats, cluster_names, results['metrics'])

        # Save results
        save_clustering_results(results, cluster_stats, cluster_names)
    else:
        print(f"No player features found at {player_path}")
        print("Please run feature_extractor.py first")
