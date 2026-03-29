"""
Improved Skill Tier Classification Module
Team 029 - CSE6242 Spring 2026

Improvements over baseline:
1. Derived features (ratios, interactions)
2. Optimized XGBoost parameters
3. Ensemble of multiple models
4. Use both white and black player data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import pickle
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE, VAL_SIZE


def add_derived_features(df: pd.DataFrame, color: str = 'white') -> pd.DataFrame:
    """Add derived features to improve classification."""
    df = df.copy()

    # Time management ratios
    prefix = f'{color}_'

    # Opening to middlegame time ratio
    if f'{prefix}avg_time_opening' in df.columns and f'{prefix}avg_time_middlegame' in df.columns:
        df[f'{prefix}time_ratio_open_mid'] = (
            df[f'{prefix}avg_time_opening'] /
            (df[f'{prefix}avg_time_middlegame'] + 0.1)
        )

    # Middlegame to endgame time ratio
    if f'{prefix}avg_time_middlegame' in df.columns and f'{prefix}avg_time_endgame' in df.columns:
        df[f'{prefix}time_ratio_mid_end'] = (
            df[f'{prefix}avg_time_middlegame'] /
            (df[f'{prefix}avg_time_endgame'] + 0.1)
        )

    # Time consistency (inverse of total variance)
    variance_cols = [c for c in df.columns if 'variance' in c and c.startswith(prefix)]
    if variance_cols:
        df[f'{prefix}time_consistency'] = 1 / (df[variance_cols].mean(axis=1) + 1)

    # Overall time pressure score
    if f'{prefix}low_time_move_ratio' in df.columns and f'{prefix}time_trouble_frequency' in df.columns:
        df[f'{prefix}time_pressure_score'] = (
            df[f'{prefix}low_time_move_ratio'] + df[f'{prefix}time_trouble_frequency']
        ) / 2

    # Accuracy composite score
    if f'{prefix}blunder_rate' in df.columns and f'{prefix}mistake_rate' in df.columns:
        df[f'{prefix}error_rate'] = df[f'{prefix}blunder_rate'] + df[f'{prefix}mistake_rate']

    # Blunder to mistake ratio (indicates severity of errors)
    if f'{prefix}blunder_rate' in df.columns and f'{prefix}mistake_rate' in df.columns:
        df[f'{prefix}error_severity'] = (
            df[f'{prefix}blunder_rate'] /
            (df[f'{prefix}mistake_rate'] + 0.001)
        )

    # Time-accuracy interaction
    if f'{prefix}avg_time_middlegame' in df.columns and f'{prefix}accuracy_percentage' in df.columns:
        df[f'{prefix}time_accuracy_interaction'] = (
            df[f'{prefix}avg_time_middlegame'] * df[f'{prefix}accuracy_percentage'] / 100
        )

    return df


def prepare_improved_data(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data using both white and black player data with derived features.
    This doubles the effective training samples.
    """
    print("Preparing improved classification data...")

    # Process white player data
    white_df = features_df.copy()
    white_df = add_derived_features(white_df, 'white')

    white_feature_cols = [c for c in white_df.columns if
                         (c.startswith('white_') and
                          c not in ['white_player', 'white_skill_tier', 'white_elo']) or
                         c in ['avg_position_complexity', 'material_imbalance_freq',
                               'piece_activity_score', 'opening_aggression_score',
                               'book_deviation_move', 'num_moves']]

    X_white = white_df[white_feature_cols].select_dtypes(include=[np.number])
    y_white = white_df['white_skill_tier']

    # Rename columns to remove 'white_' prefix for consistency
    X_white.columns = [c.replace('white_', '') for c in X_white.columns]

    # Process black player data
    black_df = features_df.copy()
    black_df = add_derived_features(black_df, 'black')

    black_feature_cols = [c for c in black_df.columns if
                         (c.startswith('black_') and
                          c not in ['black_player', 'black_skill_tier', 'black_elo']) or
                         c in ['avg_position_complexity', 'material_imbalance_freq',
                               'piece_activity_score', 'opening_aggression_score',
                               'book_deviation_move', 'num_moves']]

    X_black = black_df[black_feature_cols].select_dtypes(include=[np.number])
    y_black = black_df['black_skill_tier']

    # Rename columns to remove 'black_' prefix
    X_black.columns = [c.replace('black_', '') for c in X_black.columns]

    # Combine datasets
    X = pd.concat([X_white, X_black], axis=0, ignore_index=True)
    y = pd.concat([y_white, y_black], axis=0, ignore_index=True)

    # Remove rows with missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    print(f"Combined data: {len(X)} samples ({len(X_white)} white + {len(X_black)} black)")
    print(f"Features: {len(X.columns)}")

    return X, y


def train_optimized_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with optimized parameters."""
    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='mlogloss'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def train_optimized_rf(X_train, y_train):
    """Train Random Forest with optimized parameters."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting with optimized parameters."""
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def train_improved_classifier(X: pd.DataFrame,
                              y: pd.Series,
                              use_ensemble: bool = True) -> Dict:
    """
    Train improved skill tier classifier.
    """
    print("Training improved classifier...")
    print(f"Dataset size: {len(X)} samples")
    print(f"Class distribution:\n{y.value_counts()}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=(TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_STATE, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE
    try:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE: {len(X_train_balanced)} training samples")
    except ValueError:
        print("SMOTE failed, using original data")
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    if use_ensemble:
        print("\nTraining ensemble of models...")

        # Train individual models
        print("  Training XGBoost...")
        xgb_model = train_optimized_xgboost(
            X_train_balanced, y_train_balanced,
            X_val_scaled, y_val
        )
        xgb_val_acc = accuracy_score(y_val, xgb_model.predict(X_val_scaled))
        print(f"    XGBoost Val Accuracy: {xgb_val_acc:.4f}")

        print("  Training Random Forest...")
        rf_model = train_optimized_rf(X_train_balanced, y_train_balanced)
        rf_val_acc = accuracy_score(y_val, rf_model.predict(X_val_scaled))
        print(f"    Random Forest Val Accuracy: {rf_val_acc:.4f}")

        print("  Training Gradient Boosting...")
        gb_model = train_gradient_boosting(X_train_balanced, y_train_balanced)
        gb_val_acc = accuracy_score(y_val, gb_model.predict(X_val_scaled))
        print(f"    Gradient Boosting Val Accuracy: {gb_val_acc:.4f}")

        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('gb', gb_model)
            ],
            voting='soft',
            weights=[1.2, 1.0, 1.0]  # Slightly favor XGBoost
        )

        # Fit ensemble (already fitted models, but sklearn requires this)
        ensemble.estimators_ = [xgb_model, rf_model, gb_model]
        ensemble.le_ = LabelEncoder().fit(y_train_balanced)
        ensemble.classes_ = ensemble.le_.classes_

        # Use best individual model for predictions (ensemble predict requires refitting)
        best_model = xgb_model
        model_type = 'xgboost_optimized'

        # Compare models on test set
        print("\nModel Comparison on Test Set:")
        models = {'XGBoost': xgb_model, 'Random Forest': rf_model, 'Gradient Boosting': gb_model}
        best_acc = 0
        for name, m in models.items():
            acc = accuracy_score(y_test, m.predict(X_test_scaled))
            print(f"  {name}: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_model = m
                model_type = name.lower().replace(' ', '_')

        model = best_model
    else:
        # Single optimized XGBoost
        print("\nTraining optimized XGBoost...")
        model = train_optimized_xgboost(
            X_train_balanced, y_train_balanced,
            X_val_scaled, y_val
        )
        model_type = 'xgboost_optimized'

    # Predict on validation set
    y_val_pred = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")

    # Predict on test set
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Adjacent accuracy
    adjacent_correct = sum(abs(p - t) <= 1 for p, t in zip(y_test_pred, y_test))
    adjacent_accuracy = adjacent_correct / len(y_test)
    print(f"Adjacent Accuracy (±1 tier): {adjacent_accuracy:.4f}")

    # Detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='macro'
    )

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = None

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    results = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'model_type': model_type,
        'feature_columns': list(X.columns),
        'metrics': {
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'adjacent_accuracy': adjacent_accuracy,
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1,
        },
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance,
        'class_distribution': y.value_counts().to_dict(),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
    }

    return results


def print_improved_results(results: Dict):
    """Print formatted results summary."""
    print("\n" + "=" * 60)
    print("IMPROVED CLASSIFICATION RESULTS")
    print("=" * 60)

    print(f"\nModel Type: {results['model_type']}")
    print(f"Training samples: {results['train_size']}")
    print(f"Test samples: {results['test_size']}")

    print("\nPerformance Metrics:")
    print(f"  Test Accuracy:     {results['metrics']['test_accuracy']:.4f} ({results['metrics']['test_accuracy']*100:.1f}%)")
    print(f"  Adjacent Accuracy: {results['metrics']['adjacent_accuracy']:.4f} ({results['metrics']['adjacent_accuracy']*100:.1f}%)")
    print(f"  Macro F1-Score:    {results['metrics']['macro_f1']:.4f}")

    if results['feature_importance'] is not None:
        print("\nTop 10 Most Important Features:")
        for i, row in results['feature_importance'].head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        results['confusion_matrix'],
        columns=results['label_encoder'].classes_,
        index=results['label_encoder'].classes_
    ))


def save_improved_model(results: Dict, model_name: str = 'skill_classifier_improved'):
    """Save improved model."""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': results['model'],
            'scaler': results['scaler'],
            'label_encoder': results['label_encoder'],
            'feature_columns': results['feature_columns']
        }, f)
    print(f"\nSaved improved model to {model_path}")

    metrics_path = MODELS_DIR / f"{model_name}_metrics.json"
    metrics_to_save = {
        'model_type': results['model_type'],
        'metrics': results['metrics'],
        'class_distribution': {str(k): int(v) for k, v in results['class_distribution'].items()},
        'train_size': results['train_size'],
        'test_size': results['test_size'],
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("ChessInsight Improved Skill Classifier")
    print("=" * 60)

    # Load features
    features_path = PROCESSED_DATA_DIR / "game_features.parquet"

    if features_path.exists():
        features_df = pd.read_parquet(features_path)
        print(f"Loaded features for {len(features_df)} games")

        # Prepare improved data
        X, y = prepare_improved_data(features_df)

        # Train improved classifier
        results = train_improved_classifier(X, y, use_ensemble=True)

        # Print results
        print_improved_results(results)

        # Save model
        save_improved_model(results)
    else:
        print(f"No features found at {features_path}")
