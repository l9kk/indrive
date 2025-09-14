"""Data preprocessing utilities for trip extraction and feature engineering."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.cluster import KMeans

from ..core.config import settings


def load_data(file_path: str) -> pd.DataFrame:
    """Load the GPS tracking data from CSV."""
    df = pd.read_csv(file_path)
    return df


def extract_trips(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract trip start points (A), intermediate points (C), and end points (B).
    
    Returns:
        Tuple of (A_points, C_points, B_points) dataframes
    """
    # Sort by trip ID and sequence
    df = df.sort_values(['randomized_id', 'sequence']).reset_index(drop=True)
    
    # Extract start points (status = A)
    A_points = df[df['status'] == 'A'].copy()
    
    # Extract intermediate points (status = C) 
    C_points = df[df['status'] == 'C'].copy()
    
    # Extract end points (status = B)
    B_points = df[df['status'] == 'B'].copy()
    
    return A_points, C_points, B_points


def cluster_destinations(B_points: pd.DataFrame, n_clusters: int = None) -> Tuple[pd.DataFrame, KMeans]:
    """
    Cluster destination points (B) using KMeans.
    
    Returns:
        Tuple of (B_points with cluster_id, fitted KMeans model)
    """
    if n_clusters is None:
        n_clusters = settings.N_CLUSTERS
    
    # Prepare coordinates for clustering
    coords = B_points[['lat', 'lng']].values
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=settings.RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)
    
    # Add cluster labels to B points
    B_clustered = B_points.copy()
    B_clustered['cluster_id'] = cluster_labels
    
    return B_clustered, kmeans


def build_features(A_points: pd.DataFrame, B_clustered: pd.DataFrame, C_points: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix for training/inference.
    
    Returns:
        DataFrame with features and target cluster_id
    """
    features_list = []
    
    for _, a_row in A_points.iterrows():
        trip_id = a_row['randomized_id']
        
        # Check if this trip has a destination in B_clustered
        b_row = B_clustered[B_clustered['randomized_id'] == trip_id]
        if len(b_row) == 0:
            continue  # Skip trips without destinations
        
        cluster_id = b_row.iloc[0]['cluster_id']
        
        # Only start point features
        features = {
            'trip_id': trip_id,
            'start_lat': a_row['lat'],
            'start_lng': a_row['lng'],
            'cluster_id': cluster_id
        }
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    return features_df


def prepare_inference_features(start_lat: float, start_lng: float) -> np.ndarray:
    """
    Prepare features for a single prediction request.
    
    Args:
        start_lat: Starting latitude
        start_lng: Starting longitude  
    
    Returns:
        Feature array ready for model prediction
    """
    features = {
        'start_lat': start_lat,
        'start_lng': start_lng,
    }
    
    # Convert to array in the correct order
    feature_array = np.array([features[col] for col in settings.FEATURE_COLUMNS])
    return feature_array.reshape(1, -1)