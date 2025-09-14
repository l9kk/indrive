"""Inference module for chance model - predicts probability of ride start at given location."""

import pickle
import numpy as np
from typing import Dict, Any
import os

from ..core.config import settings


class ChanceModelInference:
    """Class for loading chance model and making probability predictions."""
    
    def __init__(self, model_path: str = None):
        """Initialize chance model inference class."""
        self.model_path = model_path or settings.CHANCE_MODEL_PATH
        self.model = None
        self.metadata = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained chance model and metadata."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Chance model file not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.metadata = model_data['metadata']
        
        print(f"Loaded {self.metadata['model_type']} chance model with {self.metadata['n_features']} features")
        print(f"Model AUC Score: {self.metadata.get('auc_score', 'N/A'):.4f}")
    
    def predict_ride_start_probability(self, lat: float, lng: float) -> Dict[str, Any]:
        """
        Predict the probability that a ride starts at the given location.
        
        Args:
            lat: Latitude
            lng: Longitude
        
        Returns:
            Dictionary with probability and metadata
        """
        # Create spatial features (same as in training)
        features = self._prepare_features(lat, lng)
        
        # Get probability
        probability = self.model.predict_proba(features.reshape(1, -1))[0, 1]
        
        # Get prediction class (for reference)
        prediction = self.model.predict(features.reshape(1, -1))[0]
        
        return {
            'lat': lat,
            'lng': lng,
            'ride_start_probability': float(probability),
            'predicted_class': int(prediction),
            'confidence_level': self._get_confidence_level(probability),
            'model_info': {
                'model_type': self.metadata['model_type'],
                'auc_score': self.metadata.get('auc_score', 0),
                'feature_count': self.metadata['n_features']
            }
        }
    
    def _prepare_features(self, lat: float, lng: float) -> np.ndarray:
        """
        Prepare features for a single prediction request.
        Must match the feature engineering from training.
        """
        # Basic features
        lat_lng_interaction = lat * lng
        
        # Distance from center (use metadata if available, otherwise estimate)
        if 'center_lat' in self.metadata and 'center_lng' in self.metadata:
            center_lat = self.metadata['center_lat']
            center_lng = self.metadata['center_lng']
        else:
            # Fallback - use rough city center
            center_lat = 51.0945  # Approximate center of the data
            center_lng = 71.4179
        
        dist_from_center = np.sqrt((lat - center_lat)**2 + (lng - center_lng)**2)
        
        # Grid features
        grid_precision = 0.001
        lat_grid = np.round(lat / grid_precision) * grid_precision
        lng_grid = np.round(lng / grid_precision) * grid_precision
        
        # Feature array (must match training order)
        features = np.array([
            lat, lng, lat_lng_interaction, dist_from_center, lat_grid, lng_grid
        ])
        
        return features
    
    def _get_confidence_level(self, probability: float) -> str:
        """Convert probability to human-readable confidence level."""
        if probability >= 0.8:
            return "Very High"
        elif probability >= 0.6:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        elif probability >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get chance model information and metadata."""
        return {
            'model_type': self.metadata['model_type'],
            'accuracy': self.metadata.get('accuracy', 0),
            'auc_score': self.metadata.get('auc_score', 0),
            'n_samples': self.metadata.get('n_samples', 0),
            'n_features': self.metadata.get('n_features', 0),
            'feature_columns': self.metadata.get('feature_columns', []),
            'positive_class_ratio': self.metadata.get('positive_class_ratio', 0),
            'feature_importance': self.metadata.get('feature_importance', {})
        }


# Singleton instance for chance model
_chance_inference_instance = None


def get_chance_inference_instance() -> ChanceModelInference:
    """Get singleton chance inference instance."""
    global _chance_inference_instance
    if _chance_inference_instance is None:
        _chance_inference_instance = ChanceModelInference()
    return _chance_inference_instance


def predict_ride_start_chance(lat: float, lng: float) -> Dict[str, Any]:
    """
    Convenience function for ride start probability prediction.
    
    Args:
        lat: Latitude
        lng: Longitude
    
    Returns:
        Dictionary with probability and metadata
    """
    inference = get_chance_inference_instance()
    return inference.predict_ride_start_probability(lat, lng)