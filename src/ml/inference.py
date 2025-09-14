"""Inference module for destination prediction."""

import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import os

from ..core.config import settings
from ..utils.preprocessing import prepare_inference_features


class DestinationModelInference:
    """Class for loading destination model and making predictions."""
    
    def __init__(self, model_path: str = None):
        """Initialize inference class."""
        self.model_path = model_path or settings.DESTINATION_MODEL_PATH
        self.classifier = None
        self.kmeans = None
        self.metadata = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model and metadata."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.kmeans = model_data['kmeans']
        self.metadata = model_data['metadata']
        
        print(f"Loaded {self.metadata['model_type']} model with {self.metadata['n_clusters']} clusters")
    
    def predict_top_k(self, start_lat: float, start_lng: float, 
                     k: int = None) -> List[Dict[str, Any]]:
        """
        Predict top-k destination clusters with probabilities.
        
        Args:
            start_lat: Starting latitude
            start_lng: Starting longitude
            k: Number of top predictions to return
        
        Returns:
            List of prediction dictionaries with cluster_id and probability
        """
        if k is None:
            k = settings.TOP_K_PREDICTIONS
        
        # Prepare features
        features = prepare_inference_features(start_lat, start_lng)
        
        # Get prediction probabilities
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(features)[0]
            classes = self.classifier.classes_
        else:
            # Fallback for models without predict_proba
            prediction = self.classifier.predict(features)[0]
            probabilities = np.zeros(len(self.metadata['cluster_labels']))
            probabilities[prediction] = 1.0
            classes = np.array(self.metadata['cluster_labels'])
        
        # Get top-k predictions
        top_k_indices = np.argsort(probabilities)[-k:][::-1]
        
        predictions = []
        for idx in top_k_indices:
            cluster_id = int(classes[idx])
            probability = float(probabilities[idx])
            
            # Get cluster center coordinates
            cluster_center = self.kmeans.cluster_centers_[cluster_id]
            
            predictions.append({
                'cluster_id': cluster_id,
                'probability': probability,
                'cluster_center': {
                    'lat': float(cluster_center[0]),
                    'lng': float(cluster_center[1])
                }
            })
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and information."""
        return self.metadata.copy() if self.metadata else {}


# Global inference instance
_destination_inference_instance = None


def get_destination_inference_instance() -> DestinationModelInference:
    """Get singleton destination inference instance."""
    global _destination_inference_instance
    if _destination_inference_instance is None:
        _destination_inference_instance = DestinationModelInference()
    return _destination_inference_instance


# Legacy support - keep old function names for backward compatibility
def get_inference_instance() -> DestinationModelInference:
    """Legacy function - use get_destination_inference_instance instead."""
    return get_destination_inference_instance()


ModelInference = DestinationModelInference


def predict_destination(start_lat: float, start_lng: float) -> List[Dict[str, Any]]:
    """
    Convenience function for destination prediction.
    
    Args:
        start_lat: Starting latitude
        start_lng: Starting longitude
    
    Returns:
        List of top-k predictions with cluster_id and probability
    """
    inference = get_inference_instance()
    return inference.predict_top_k(start_lat, start_lng)