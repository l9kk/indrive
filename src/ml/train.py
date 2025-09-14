"""Training script for destination prediction model."""

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any, Tuple

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available, falling back to LogisticRegression")

from ..core.config import settings
from ..utils.preprocessing import (
    load_data, extract_trips, cluster_destinations, build_features
)


def train_destination_model(features_df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """
    Train the destination prediction model.
    
    Returns:
        Tuple of (trained_model, metadata)
    """
    # Prepare features and target
    X = features_df[settings.FEATURE_COLUMNS].values
    y = features_df['cluster_id'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.RANDOM_STATE, stratify=y
    )
    
    # Train model
    if settings.MODEL_TYPE == "catboost" and CATBOOST_AVAILABLE:
        model = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_seed=settings.RANDOM_STATE,
            verbose=False
        )
        model.fit(X_train, y_train)
    else:
        model = LogisticRegression(
            random_state=settings.RANDOM_STATE,
            max_iter=1000,
            multi_class='ovr'
        )
        model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate top-3 accuracy
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        top_3_accuracy = calculate_top_k_accuracy(y_test, y_proba, k=3)
    else:
        top_3_accuracy = accuracy  # Fallback for models without predict_proba
    
    print(f"Model: {type(model).__name__}")
    print(f"Top-1 Accuracy: {accuracy:.4f}")
    print(f"Top-3 Accuracy: {top_3_accuracy:.4f}")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of clusters: {len(np.unique(y))}")
    
    # Calculate cluster sizes
    cluster_sizes = {}
    unique_clusters, cluster_counts = np.unique(y, return_counts=True)
    for cluster_id, count in zip(unique_clusters, cluster_counts):
        cluster_sizes[int(cluster_id)] = int(count)
    
    metadata = {
        'model_type': type(model).__name__,
        'top_1_accuracy': accuracy,
        'top_3_accuracy': top_3_accuracy,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_clusters': len(np.unique(y)),
        'feature_columns': settings.FEATURE_COLUMNS,
        'cluster_labels': sorted(np.unique(y).tolist()),
        'cluster_sizes': cluster_sizes
    }
    
    return model, metadata


def calculate_top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    """Calculate top-k accuracy."""
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    top_k_accuracy = np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return top_k_accuracy


def save_model(model: Any, metadata: Dict[str, Any], kmeans_model: Any, file_path: str) -> None:
    """Save trained model and metadata to pickle file."""
    model_data = {
        'classifier': model,
        'kmeans': kmeans_model,
        'metadata': metadata
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {file_path}")


def main():
    """Main training pipeline."""
    print("Starting training pipeline...")
    
    # Load data
    print("Loading data...")
    df = load_data(settings.DATA_PATH)
    print(f"Loaded {len(df)} records")
    
    # Extract trips
    print("Extracting trips...")
    A_points, C_points, B_points = extract_trips(df)
    print(f"Found {len(A_points)} start points, {len(C_points)} intermediate points, {len(B_points)} end points")
    
    # Cluster destinations
    print("Clustering destinations...")
    B_clustered, kmeans_model = cluster_destinations(B_points)
    print(f"Created {settings.N_CLUSTERS} destination clusters")
    
    # Print cluster distribution
    cluster_counts = B_clustered['cluster_id'].value_counts().sort_index()
    print("Cluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} destinations")
    
    # Build features
    print("Building features...")
    features_df = build_features(A_points, B_clustered, C_points)
    print(f"Built features for {len(features_df)} trips")
    
    # Filter out trips with insufficient data
    valid_trips = features_df.dropna(subset=['cluster_id'])
    print(f"Training on {len(valid_trips)} trips with complete data")
    
    # Train model
    print("Training destination model...")
    model, metadata = train_destination_model(valid_trips)
    
    # Save model
    save_model(model, metadata, kmeans_model, settings.DESTINATION_MODEL_PATH)
    
    print("Destination model training completed successfully!")


if __name__ == "__main__":
    main()