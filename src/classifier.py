"""
Skill Tier Classification Module
Team 029 - CSE6242 Spring 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, MODELS_DIR, SKILL_TIERS, RANDOM_STATE, TEST_SIZE, VAL_SIZE


def prepare_classification_data(features_df: pd.DataFrame,
                                target_color: str = 'white') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for skill tier classification.

    Args:
        features_df: DataFrame with game features
        target_color: Which player's skill to predict ('white' or 'black')

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Select feature columns
    feature_cols = [c for c in features_df.columns if
                    (c.startswith(f'{target_color}_') and
                     c not in [f'{target_color}_player', f'{target_color}_skill_tier',
                               f'{target_color}_elo']) or
                    c in ['avg_position_complexity', 'material_imbalance_freq',
                          'piece_activity_score', 'opening_aggression_score',
                          'book_deviation_move', 'num_moves']]

    # Remove non-numeric columns
    X = features_df[feature_cols].select_dtypes(include=[np.number])

    # Target variable
    y = features_df[f'{target_color}_skill_tier']

    # Remove rows with missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    return X, y


def train_classifier(X: pd.DataFrame,
                     y: pd.Series,
                     model_type: str = 'random_forest',
                     use_smote: bool = True) -> Dict:
    """
    Train a skill tier classifier.

    Args:
        X: Feature matrix
        y: Target labels
        model_type: Type of classifier ('random_forest', 'xgboost', 'gradient_boosting')
        use_smote: Whether to use SMOTE for class balancing

    Returns:
        Dictionary with model, metrics, and other information
    """
    print(f"Training {model_type} classifier...")
    print(f"Dataset size: {len(X)} samples")
    print(f"Class distribution:\n{y.value_counts()}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=(TEST_SIZE + VAL_SIZE), random_state=RANDOM_STATE, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE for class balancing
    if use_smote:
        try:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE: {len(X_train_balanced)} training samples")
        except ValueError:
            print("SMOTE failed (likely too few samples), using original data")
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    # Initialize model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model
    model.fit(X_train_balanced, y_train_balanced)

    # Predict on validation set
    y_val_pred = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Predict on test set
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Calculate adjacent accuracy (prediction within ±1 tier)
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

    # Classification report
    class_report = classification_report(
        y_test, y_test_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

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
        'classification_report': class_report,
        'feature_importance': feature_importance,
        'class_distribution': y.value_counts().to_dict(),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
    }

    return results


def hyperparameter_tuning(X: pd.DataFrame,
                          y: pd.Series,
                          model_type: str = 'random_forest') -> Dict:
    """
    Perform hyperparameter tuning using GridSearchCV.

    Args:
        X: Feature matrix
        y: Target labels
        model_type: Type of classifier

    Returns:
        Dictionary with best model and parameters
    """
    print("Performing hyperparameter tuning...")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
        }
    elif model_type == 'xgboost':
        model = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.05, 0.1, 0.2],
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_scaled, y_encoded)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }


def save_model(results: Dict, model_name: str = 'skill_classifier'):
    """
    Save trained model and associated objects.

    Args:
        results: Dictionary from train_classifier
        model_name: Name for saved files
    """
    # Save model
    model_path = MODELS_DIR / f"{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': results['model'],
            'scaler': results['scaler'],
            'label_encoder': results['label_encoder'],
            'feature_columns': results['feature_columns']
        }, f)
    print(f"Saved model to {model_path}")

    # Save metrics
    metrics_path = MODELS_DIR / f"{model_name}_metrics.json"
    metrics_to_save = {
        'model_type': results['model_type'],
        'metrics': results['metrics'],
        'class_distribution': results['class_distribution'],
        'train_size': results['train_size'],
        'val_size': results['val_size'],
        'test_size': results['test_size'],
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Save feature importance
    if results['feature_importance'] is not None:
        importance_path = MODELS_DIR / f"{model_name}_feature_importance.csv"
        results['feature_importance'].to_csv(importance_path, index=False)
        print(f"Saved feature importance to {importance_path}")

    # Save confusion matrix
    conf_matrix_path = MODELS_DIR / f"{model_name}_confusion_matrix.csv"
    pd.DataFrame(
        results['confusion_matrix'],
        columns=results['label_encoder'].classes_,
        index=results['label_encoder'].classes_
    ).to_csv(conf_matrix_path)
    print(f"Saved confusion matrix to {conf_matrix_path}")


def load_model(model_name: str = 'skill_classifier') -> Dict:
    """
    Load trained model.

    Args:
        model_name: Name of saved model

    Returns:
        Dictionary with model, scaler, label_encoder, feature_columns
    """
    model_path = MODELS_DIR / f"{model_name}.pkl"
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_skill_tier(features: pd.DataFrame, model_dict: Dict) -> np.ndarray:
    """
    Predict skill tier for new games.

    Args:
        features: DataFrame with game features
        model_dict: Dictionary from load_model

    Returns:
        Array of predicted skill tiers
    """
    # Select and order features
    X = features[model_dict['feature_columns']]

    # Scale
    X_scaled = model_dict['scaler'].transform(X)

    # Predict
    y_pred = model_dict['model'].predict(X_scaled)

    # Decode labels
    return model_dict['label_encoder'].inverse_transform(y_pred)


def print_results_summary(results: Dict):
    """Print a formatted summary of classification results."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nModel Type: {results['model_type']}")
    print(f"Training samples: {results['train_size']}")
    print(f"Validation samples: {results['val_size']}")
    print(f"Test samples: {results['test_size']}")

    print("\nPerformance Metrics:")
    print(f"  Test Accuracy:     {results['metrics']['test_accuracy']:.4f} ({results['metrics']['test_accuracy']*100:.1f}%)")
    print(f"  Adjacent Accuracy: {results['metrics']['adjacent_accuracy']:.4f} ({results['metrics']['adjacent_accuracy']*100:.1f}%)")
    print(f"  Macro F1-Score:    {results['metrics']['macro_f1']:.4f}")

    print("\nClass Distribution:")
    for tier, count in results['class_distribution'].items():
        print(f"  {tier}: {count}")

    if results['feature_importance'] is not None:
        print("\nTop 10 Most Important Features:")
        for i, row in results['feature_importance'].head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        results['confusion_matrix'],
        columns=results['label_encoder'].classes_,
        index=results['label_encoder'].classes_
    ))


if __name__ == "__main__":
    print("ChessInsight Skill Classifier")
    print("=" * 50)

    # Load features
    features_path = PROCESSED_DATA_DIR / "game_features.parquet"

    if features_path.exists():
        features_df = pd.read_parquet(features_path)
        print(f"Loaded features for {len(features_df)} games")

        # Prepare data
        X, y = prepare_classification_data(features_df, target_color='white')
        print(f"Prepared {len(X)} samples with {len(X.columns)} features")

        # Train classifier
        results = train_classifier(X, y, model_type='random_forest')

        # Print summary
        print_results_summary(results)

        # Save model
        save_model(results)
    else:
        print(f"No features found at {features_path}")
        print("Please run feature_extractor.py first")
