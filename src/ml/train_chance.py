"""Training script for chance model - predicts probability that a ride starts at given location."""

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available, using RandomForest")

from ..core.config import settings


def load_and_prepare_chance_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset and prepare for chance model training.
    
    Returns:
        DataFrame with lat, lng, and target (1 if status='A', 0 otherwise)
    """
    # Load data
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records from dataset")
    
    # Extract coordinates and create target variable
    # Target = 1 if status='A' (ride pickup), 0 otherwise
    features_df = df[['lat', 'lng', 'status']].copy()
    features_df['target'] = (features_df['status'] == 'A').astype(int)
    
    # Remove status column as it's not needed for features
    features_df = features_df.drop('status', axis=1)
    
    print(f"Created target variable:")
    print(f"  - Ride starts (A): {features_df['target'].sum()} ({features_df['target'].mean()*100:.2f}%)")
    print(f"  - Other points: {len(features_df) - features_df['target'].sum()} ({(1-features_df['target'].mean())*100:.2f}%)")
    
    return features_df


def create_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional spatial features to improve model performance.
    
    Args:
        df: DataFrame with lat, lng, target columns
        
    Returns:
        DataFrame with additional spatial features
    """
    df_enhanced = df.copy()
    
    # Basic coordinate features
    df_enhanced['lat_lng_interaction'] = df_enhanced['lat'] * df_enhanced['lng']
    
    # Distance from city center (assuming center is around mean coordinates of A points)
    a_points = df[df['target'] == 1]
    if len(a_points) > 0:
        center_lat = a_points['lat'].mean()
        center_lng = a_points['lng'].mean()
        
        df_enhanced['dist_from_center'] = np.sqrt(
            (df_enhanced['lat'] - center_lat)**2 + 
            (df_enhanced['lng'] - center_lng)**2
        )
    else:
        df_enhanced['dist_from_center'] = 0
    
    # Grid-based features (round coordinates to create grid cells)
    grid_precision = 0.001  # ~100m precision
    df_enhanced['lat_grid'] = np.round(df_enhanced['lat'] / grid_precision) * grid_precision
    df_enhanced['lng_grid'] = np.round(df_enhanced['lng'] / grid_precision) * grid_precision
    
    print(f"Created {len(df_enhanced.columns) - len(df.columns)} additional spatial features")
    
    return df_enhanced


def train_chance_model(features_df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """
    Train the chance model to predict probability of ride start at given location.
    
    Returns:
        Tuple of (trained_model, metadata)
    """
    # Prepare features and target
    feature_columns = ['lat', 'lng', 'lat_lng_interaction', 'dist_from_center', 'lat_grid', 'lng_grid']
    X = features_df[feature_columns].values
    y = features_df['target'].values
    
    print(f"Training on {len(X)} samples with {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.RANDOM_STATE, stratify=y
    )
    
    # Train model - use RandomForest as it works well for probability estimation
    if CATBOOST_AVAILABLE:
        model = CatBoostClassifier(
            iterations=100,
            depth=4,
            learning_rate=0.1,
            random_seed=settings.RANDOM_STATE,
            verbose=False,
            class_weights={0: 1, 1: 3}  # Give more weight to positive class due to imbalance
        )
        model.fit(X_train, y_train)
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=settings.RANDOM_STATE,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1
        )
        model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    
    print(f"Model: {type(model).__name__}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        print("Feature importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.4f}")
    
    metadata = {
        'model_type': type(model).__name__,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_columns': feature_columns,
        'positive_class_ratio': y.mean(),
        'feature_importance': feature_importance if hasattr(model, 'feature_importances_') else None
    }
    
    return model, metadata


def save_chance_model(model: Any, metadata: Dict[str, Any], file_path: str) -> None:
    """Save trained chance model and metadata to pickle file."""
    model_data = {
        'model': model,
        'metadata': metadata
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Chance model saved to: {file_path}")


def main():
    """Main training pipeline for chance model."""
    print("Starting chance model training pipeline...")
    print("Goal: Predict probability that a ride starts at given location")
    
    # Load and prepare data
    print("Loading and preparing data...")
    features_df = load_and_prepare_chance_data(settings.DATA_PATH)
    
    # Create additional spatial features
    print("Creating spatial features...")
    features_df = create_spatial_features(features_df)
    
    # Train model
    print("Training chance model...")
    model, metadata = train_chance_model(features_df)
    
    # Save model
    save_chance_model(model, metadata, settings.CHANCE_MODEL_PATH)
    
    print("Chance model training completed successfully!")
    print(f"Model can now predict ride start probability for any lat/lng coordinate")


if __name__ == "__main__":
    main()