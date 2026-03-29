"""
Visualization Module - Create charts and dashboard designs
Team 029 - CSE6242 Spring 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, MODELS_DIR, VIZ_DIR, SKILL_TIERS


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_skill_distribution(features_df: pd.DataFrame,
                            save_path: Optional[Path] = None):
    """
    Plot distribution of skill tiers in the dataset.

    Args:
        features_df: DataFrame with game features
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # White player distribution
    tier_order = list(SKILL_TIERS.keys())
    white_counts = features_df['white_skill_tier'].value_counts().reindex(tier_order)
    axes[0].bar(tier_order, white_counts.values, color=sns.color_palette("Blues_d", len(tier_order)))
    axes[0].set_title('White Player Skill Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Skill Tier')
    axes[0].set_ylabel('Number of Games')
    axes[0].tick_params(axis='x', rotation=45)

    # Black player distribution
    black_counts = features_df['black_skill_tier'].value_counts().reindex(tier_order)
    axes[1].bar(tier_order, black_counts.values, color=sns.color_palette("Oranges_d", len(tier_order)))
    axes[1].set_title('Black Player Skill Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Skill Tier')
    axes[1].set_ylabel('Number of Games')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved skill distribution plot to {save_path}")

    plt.close()


def plot_time_heatmap(features_df: pd.DataFrame,
                      save_path: Optional[Path] = None):
    """
    Create heatmap of time usage across skill tiers and game phases.

    Args:
        features_df: DataFrame with game features
        save_path: Path to save the figure
    """
    # Calculate average time per phase for each skill tier
    time_cols = {
        'Opening': 'white_avg_time_opening',
        'Middlegame': 'white_avg_time_middlegame',
        'Endgame': 'white_avg_time_endgame'
    }

    tier_order = list(SKILL_TIERS.keys())
    heatmap_data = []

    for tier in tier_order:
        tier_data = features_df[features_df['white_skill_tier'] == tier]
        row = {}
        for phase, col in time_cols.items():
            if col in tier_data.columns:
                row[phase] = tier_data[col].mean()
            else:
                row[phase] = np.nan
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data, index=tier_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'Avg Time (seconds)'})
    ax.set_title('Average Time per Move by Skill Tier and Game Phase', fontsize=12, fontweight='bold')
    ax.set_xlabel('Game Phase')
    ax.set_ylabel('Skill Tier')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved time heatmap to {save_path}")

    plt.close()


def plot_confusion_matrix(conf_matrix: np.ndarray,
                          class_names: List[str],
                          save_path: Optional[Path] = None):
    """
    Plot confusion matrix for classification results.

    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_title('Skill Tier Classification - Normalized Confusion Matrix',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Tier')
    ax.set_ylabel('Actual Tier')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix plot to {save_path}")

    plt.close()


def plot_feature_importance(feature_importance: pd.DataFrame,
                            top_n: int = 15,
                            save_path: Optional[Path] = None):
    """
    Plot feature importance from classification model.

    Args:
        feature_importance: DataFrame with feature and importance columns
        top_n: Number of top features to show
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    top_features = feature_importance.head(top_n)

    colors = sns.color_palette("viridis", len(top_features))
    bars = ax.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()  # Highest importance at top

    ax.set_title(f'Top {top_n} Most Important Features for Skill Classification',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Importance')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")

    plt.close()


def plot_cluster_embedding(embedding_df: pd.DataFrame,
                           cluster_names: Dict,
                           player_features: Optional[pd.DataFrame] = None,
                           save_path: Optional[Path] = None):
    """
    Plot 2D embedding of player clusters.

    Args:
        embedding_df: DataFrame with x, y, cluster columns
        cluster_names: Dictionary mapping cluster IDs to names
        player_features: Optional DataFrame with player info for annotations
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create color map
    unique_clusters = sorted(embedding_df['cluster'].unique())
    colors = sns.color_palette("husl", len(unique_clusters))
    color_map = {c: colors[i] for i, c in enumerate(unique_clusters)}

    # Plot each cluster
    for cluster_id in unique_clusters:
        mask = embedding_df['cluster'] == cluster_id
        cluster_data = embedding_df[mask]

        label = cluster_names.get(cluster_id, {}).get('name', f'Cluster {cluster_id}')

        ax.scatter(cluster_data['x'], cluster_data['y'],
                   c=[color_map[cluster_id]], label=label,
                   alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

    ax.set_title('Player Behavioral Clusters (t-SNE Embedding)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(loc='best', title='Player Archetypes')

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved cluster embedding plot to {save_path}")

    plt.close()


def plot_cluster_characteristics(cluster_stats: pd.DataFrame,
                                 cluster_names: Dict,
                                 save_path: Optional[Path] = None):
    """
    Plot radar chart showing cluster characteristics.

    Args:
        cluster_stats: DataFrame with cluster statistics
        cluster_names: Dictionary mapping cluster IDs to names
        save_path: Path to save the figure
    """
    # Select key metrics for comparison
    metric_cols = [c for c in cluster_stats.columns
                   if c.endswith('_mean') or c.startswith('pct_')][:6]

    if len(metric_cols) < 3:
        print("Not enough metrics for radar chart")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    n_clusters = len(cluster_stats)
    x = np.arange(len(metric_cols))
    width = 0.8 / n_clusters

    colors = sns.color_palette("husl", n_clusters)

    for i, (_, row) in enumerate(cluster_stats.iterrows()):
        cluster_id = int(row['cluster'])
        label = cluster_names.get(cluster_id, {}).get('name', f'Cluster {cluster_id}')
        values = [row[col] if col in row and pd.notna(row[col]) else 0 for col in metric_cols]

        # Normalize values for comparison
        values = np.array(values)
        if values.max() > 0:
            values = values / values.max()

        ax.bar(x + i * width, values, width, label=label, color=colors[i], alpha=0.8)

    ax.set_xlabel('Characteristic')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Cluster Characteristics Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * (n_clusters - 1) / 2)
    ax.set_xticklabels([c.replace('_mean', '').replace('_', ' ').title()[:15]
                        for c in metric_cols], rotation=45, ha='right')
    ax.legend(title='Archetypes')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved cluster characteristics plot to {save_path}")

    plt.close()


def plot_accuracy_by_tier(features_df: pd.DataFrame,
                          save_path: Optional[Path] = None):
    """
    Plot accuracy metrics (blunder rate, etc.) by skill tier.

    Args:
        features_df: DataFrame with game features
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    tier_order = list(SKILL_TIERS.keys())

    # Blunder rate by tier
    if 'white_blunder_rate' in features_df.columns:
        blunder_by_tier = features_df.groupby('white_skill_tier')['white_blunder_rate'].mean()
        blunder_by_tier = blunder_by_tier.reindex(tier_order)

        axes[0].bar(tier_order, blunder_by_tier.values * 100,
                    color=sns.color_palette("Reds_d", len(tier_order)))
        axes[0].set_title('Average Blunder Rate by Skill Tier', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Skill Tier')
        axes[0].set_ylabel('Blunder Rate (%)')
        axes[0].tick_params(axis='x', rotation=45)

    # Accuracy by tier
    if 'white_accuracy_percentage' in features_df.columns:
        acc_by_tier = features_df.groupby('white_skill_tier')['white_accuracy_percentage'].mean()
        acc_by_tier = acc_by_tier.reindex(tier_order)

        axes[1].bar(tier_order, acc_by_tier.values,
                    color=sns.color_palette("Greens_d", len(tier_order)))
        axes[1].set_title('Average Accuracy by Skill Tier', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Skill Tier')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved accuracy by tier plot to {save_path}")

    plt.close()


def generate_all_visualizations(features_df: pd.DataFrame,
                                player_features: pd.DataFrame,
                                classification_results: Dict,
                                clustering_results: Dict,
                                cluster_stats: pd.DataFrame,
                                cluster_names: Dict):
    """
    Generate all visualizations for the progress report.

    Args:
        features_df: Game-level features
        player_features: Player-level features
        classification_results: Results from classifier
        clustering_results: Results from clustering
        cluster_stats: Cluster statistics
        cluster_names: Cluster names dictionary
    """
    print("\nGenerating visualizations...")

    # 1. Skill distribution
    plot_skill_distribution(features_df, VIZ_DIR / "skill_distribution.png")

    # 2. Time heatmap
    plot_time_heatmap(features_df, VIZ_DIR / "time_heatmap.png")

    # 3. Confusion matrix
    if 'confusion_matrix' in classification_results:
        plot_confusion_matrix(
            classification_results['confusion_matrix'],
            list(classification_results['label_encoder'].classes_),
            VIZ_DIR / "confusion_matrix.png"
        )

    # 4. Feature importance
    if classification_results.get('feature_importance') is not None:
        plot_feature_importance(
            classification_results['feature_importance'],
            save_path=VIZ_DIR / "feature_importance.png"
        )

    # 5. Cluster embedding
    embedding_df = pd.DataFrame({
        'x': clustering_results['embedding_2d'][:, 0],
        'y': clustering_results['embedding_2d'][:, 1],
        'cluster': clustering_results['labels']
    })
    plot_cluster_embedding(embedding_df, cluster_names, save_path=VIZ_DIR / "cluster_embedding.png")

    # 6. Cluster characteristics
    plot_cluster_characteristics(cluster_stats, cluster_names, VIZ_DIR / "cluster_characteristics.png")

    # 7. Accuracy by tier
    plot_accuracy_by_tier(features_df, VIZ_DIR / "accuracy_by_tier.png")

    print(f"All visualizations saved to {VIZ_DIR}")


def create_dashboard_wireframe():
    """
    Create a wireframe/mockup of the dashboard design.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # View 1: Player Embedding Map
    ax1 = axes[0, 0]
    ax1.set_title('View 1: Player Embedding Map', fontsize=14, fontweight='bold')
    # Simulate scatter plot
    np.random.seed(42)
    for i, color in enumerate(['#e74c3c', '#3498db', '#2ecc71', '#f39c12']):
        x = np.random.randn(50) + i * 2
        y = np.random.randn(50)
        ax1.scatter(x, y, c=color, alpha=0.6, s=30, label=f'Archetype {i+1}')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlabel('t-SNE Dim 1')
    ax1.set_ylabel('t-SNE Dim 2')
    ax1.text(0.02, 0.98, 'Interactive: Hover for details\nClick to filter other views',
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # View 2: Time Usage Heatmap (4 tiers per proposal)
    ax2 = axes[0, 1]
    ax2.set_title('View 2: Time Usage Heatmap', fontsize=14, fontweight='bold')
    data = np.random.rand(4, 3) * 20 + 5
    sns.heatmap(data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2,
                xticklabels=['Opening', 'Middlegame', 'Endgame'],
                yticklabels=['Beginner', 'Intermediate', 'Advanced', 'Expert'])
    ax2.set_xlabel('Game Phase')
    ax2.set_ylabel('Skill Tier')
    ax2.text(0.02, -0.15, 'Interactive: Filter by cluster/rating',
             transform=ax2.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # View 3: Opening Network Graph
    ax3 = axes[1, 0]
    ax3.set_title('View 3: Opening Network Graph', fontsize=14, fontweight='bold')
    # Simulate network graph
    from matplotlib.patches import Circle, FancyArrowPatch
    openings = ['e4', 'd4', 'c4', 'Nf3', 'e5', 'd5']
    positions = {
        'e4': (0.3, 0.7), 'd4': (0.7, 0.7), 'c4': (0.5, 0.5),
        'Nf3': (0.3, 0.3), 'e5': (0.7, 0.3), 'd5': (0.5, 0.1)
    }
    sizes = [0.12, 0.1, 0.06, 0.05, 0.08, 0.05]
    for i, (opening, pos) in enumerate(positions.items()):
        circle = Circle(pos, sizes[i], color=plt.cm.Blues(0.3 + sizes[i] * 3),
                        ec='darkblue', linewidth=2)
        ax3.add_patch(circle)
        ax3.text(pos[0], pos[1], opening, ha='center', va='center', fontsize=10, fontweight='bold')
    # Add edges
    edges = [('e4', 'e5'), ('d4', 'd5'), ('e4', 'c4'), ('d4', 'Nf3')]
    for start, end in edges:
        ax3.annotate('', xy=positions[end], xytext=positions[start],
                     arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.text(0.02, 0.02, 'Interactive: Filter by skill tier\nNode size = popularity, Color = win rate',
             transform=ax3.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # View 4: Controls & Filters
    ax4 = axes[1, 1]
    ax4.set_title('Dashboard Controls', fontsize=14, fontweight='bold')
    ax4.axis('off')
    controls_text = """
    FILTERS:
    ┌─────────────────────────────┐
    │ Skill Tier: [All ▼]         │
    │ Time Control: [Blitz ▼]     │
    │ Rating Range: [1200] - [1800]│
    │ Cluster: [All ▼]            │
    └─────────────────────────────┘

    DISPLAY OPTIONS:
    ☑ Show cluster labels
    ☑ Color by skill tier
    ☐ Show confidence intervals

    SELECTED PLAYER INFO:
    ┌─────────────────────────────┐
    │ Player: -                   │
    │ Rating: -                   │
    │ Games: -                    │
    │ Archetype: -                │
    └─────────────────────────────┘
    """
    ax4.text(0.1, 0.9, controls_text, transform=ax4.transAxes, fontsize=10,
             va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('ChessInsight Dashboard Wireframe', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    wireframe_path = VIZ_DIR / "dashboard_wireframe.png"
    plt.savefig(wireframe_path, dpi=150, bbox_inches='tight')
    print(f"Saved dashboard wireframe to {wireframe_path}")
    plt.close()


if __name__ == "__main__":
    print("ChessInsight Visualizations")
    print("=" * 50)

    # Create dashboard wireframe
    create_dashboard_wireframe()

    # Check for processed data
    features_path = PROCESSED_DATA_DIR / "game_features.parquet"

    if features_path.exists():
        features_df = pd.read_parquet(features_path)
        print(f"Loaded {len(features_df)} games")

        # Generate basic visualizations
        plot_skill_distribution(features_df, VIZ_DIR / "skill_distribution.png")
        plot_time_heatmap(features_df, VIZ_DIR / "time_heatmap.png")
        plot_accuracy_by_tier(features_df, VIZ_DIR / "accuracy_by_tier.png")
    else:
        print("No processed data found. Run the pipeline first.")
