"""Configuration settings for the destination prediction service."""

import os
from typing import List

class Settings:
    """Application settings and configuration."""
    
    # Data paths
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "geo_locations_labeled_advanced.csv")
    DESTINATION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "destination_model.pkl")
    CHANCE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "chance_model.pkl")
    
    # API settings
    API_TITLE = "Destination Prediction API"
    API_DESCRIPTION = "FastAPI service for predicting trip destinations and ride start probabilities"
    API_VERSION = "1.0.0"
    
    # ML settings
    N_CLUSTERS = 15  # Number of destination clusters (increased from 7)
    RANDOM_STATE = 42
    
    # Model parameters
    MODEL_TYPE = "catboost"  # Options: "catboost", "logistic"
    
    # Feature settings
    FEATURE_COLUMNS = [
        "start_lat", "start_lng",  # Starting point only
    ]
    
    # Prediction settings
    TOP_K_PREDICTIONS = 3

settings = Settings()