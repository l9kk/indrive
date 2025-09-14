"""Pydantic schemas for prediction API requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Request model for destination prediction."""
    start_lat: float = Field(..., description="Starting latitude", ge=-90, le=90)
    start_lng: float = Field(..., description="Starting longitude", ge=-180, le=180)


class ClusterCenter(BaseModel):
    """Model for cluster center coordinates."""
    lat: float = Field(..., description="Cluster center latitude")
    lng: float = Field(..., description="Cluster center longitude")


class PredictionResult(BaseModel):
    """Model for a single prediction result."""
    cluster_id: int = Field(..., description="Predicted destination cluster ID")
    probability: float = Field(..., description="Prediction probability", ge=0, le=1)
    cluster_center: ClusterCenter = Field(..., description="Coordinates of cluster center")


class PredictionResponse(BaseModel):
    """Response model for destination prediction."""
    predictions: List[PredictionResult] = Field(..., description="Top-k destination predictions")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")
    request_summary: Dict[str, Any] = Field(..., description="Summary of input request")


class ClusterInfo(BaseModel):
    """Model for detailed cluster information."""
    cluster_id: int = Field(..., description="Cluster ID")
    center: ClusterCenter = Field(..., description="Cluster center coordinates")
    size: int = Field(..., description="Number of destinations in this cluster")
    description: Optional[str] = Field(None, description="Human-readable cluster description")


class StartPointHeatmap(BaseModel):
    """Model for start point heatmap data."""
    lat: float = Field(..., description="Start point latitude")
    lng: float = Field(..., description="Start point longitude")
    intensity: int = Field(..., description="Number of trips from this point")


class ClusterHeatmapInfo(BaseModel):
    """Model for cluster heatmap information."""
    cluster_id: int = Field(..., description="Start cluster ID")
    start_center: ClusterCenter = Field(..., description="Start cluster center coordinates")
    trip_count: int = Field(..., description="Total trips starting from this cluster")
    avg_distance: Optional[float] = Field(None, description="Average distance within cluster (dispersion)")


class HeatmapResponse(BaseModel):
    """Response model for cluster heatmap data."""
    clusters: List[ClusterHeatmapInfo] = Field(..., description="Heatmap data for all clusters")
    total_clusters: int = Field(..., description="Total number of clusters")
    total_trips: int = Field(..., description="Total trips analyzed")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")


class ClustersResponse(BaseModel):
    """Response model for clusters information."""
    clusters: List[ClusterInfo] = Field(..., description="List of all clusters")
    total_clusters: int = Field(..., description="Total number of clusters")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_type: str = Field(..., description="Type of ML model used")
    n_clusters: int = Field(..., description="Number of destination clusters")
    n_features: int = Field(..., description="Number of input features")
    n_samples: int = Field(..., description="Number of training samples")
    top_1_accuracy: float = Field(..., description="Top-1 prediction accuracy")
    top_3_accuracy: float = Field(..., description="Top-3 prediction accuracy")
    feature_columns: List[str] = Field(..., description="List of feature column names")
    cluster_labels: List[int] = Field(..., description="Available cluster IDs")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: Optional[str] = Field(None, description="Type of loaded model")
    n_clusters: Optional[int] = Field(None, description="Number of clusters")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class ErrorResponse(BaseModel):
    """Response model for API errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")


# Chance Model Schemas
class ChanceRequest(BaseModel):
    """Request model for ride start probability prediction."""
    lat: float = Field(..., description="Latitude", ge=-90, le=90)
    lng: float = Field(..., description="Longitude", ge=-180, le=180)


class ChanceResponse(BaseModel):
    """Response model for ride start probability prediction."""
    lat: float = Field(..., description="Query latitude")
    lng: float = Field(..., description="Query longitude")
    ride_start_probability: float = Field(..., description="Probability that a ride starts at this location", ge=0, le=1)
    predicted_class: int = Field(..., description="Predicted class (0 or 1)")
    confidence_level: str = Field(..., description="Human-readable confidence level")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")


class ChanceModelInfoResponse(BaseModel):
    """Response model for chance model information."""
    model_type: str = Field(..., description="Type of ML model used")
    accuracy: float = Field(..., description="Model accuracy")
    auc_score: float = Field(..., description="Area Under Curve score")
    n_samples: int = Field(..., description="Number of training samples")
    n_features: int = Field(..., description="Number of input features")
    feature_columns: List[str] = Field(..., description="List of feature column names")
    positive_class_ratio: float = Field(..., description="Ratio of positive class in training data")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")


# Client Management Schemas
class Location(BaseModel):
    """Location with coordinates only."""
    lat: float = Field(..., description="Latitude", ge=-90, le=90)
    lng: float = Field(..., description="Longitude", ge=-180, le=180)


class DestinationAreaPrediction(BaseModel):
    """Model for destination area prediction based on client trips."""
    destination_area: str = Field(..., description="Destination area/district name")
    trip_count: int = Field(..., description="Number of client trips to this area")
    percentage: float = Field(..., description="Percentage of trips to this area", ge=0, le=100)
    coordinates: Dict[str, float] = Field(..., description="Representative coordinates for the area")


class AreaBasedPredictionResponse(BaseModel):
    """Response model for area-based destination prediction."""
    predictions: List[DestinationAreaPrediction] = Field(..., description="Top-3 destination areas")
    total_nearby_clients: int = Field(..., description="Number of clients found in search area")
    search_radius_km: float = Field(..., description="Search radius used in kilometers")
    query_location: Dict[str, float] = Field(..., description="Query coordinates")
    

class Client(BaseModel):
    """Client model with minimal data."""
    id: int = Field(..., description="Client ID")
    name: str = Field(..., description="Client name")
    age: int = Field(..., description="Client age", ge=16, le=100)
    current_location: Location = Field(..., description="Current location coordinates")
    destination: Location = Field(..., description="Destination coordinates")
    estimated_duration: int = Field(..., description="Estimated trip duration in minutes")


class ClientAnalysis(BaseModel):
    """Analysis results for a client."""
    client: Client = Field(..., description="Client information")
    destination_prediction: List[PredictionResult] = Field(..., description="Destination predictions")
    start_chance: ChanceResponse = Field(..., description="Ride start probability")
    destination_chance: ChanceResponse = Field(..., description="Ride start probability at destination")


class ClientsResponse(BaseModel):
    """Response with list of clients."""
    clients: List[Client] = Field(..., description="List of clients")
    total_clients: int = Field(..., description="Total number of clients")


# HINT System Schemas
class HintLocationRequest(BaseModel):
    """Request model for hint location check."""
    lat: float = Field(..., description="Latitude", ge=-90, le=90)
    lng: float = Field(..., description="Longitude", ge=-180, le=180)
    radius_km: Optional[float] = Field(0.5, description="Search radius in kilometers", ge=0.1, le=5.0)


class HintHeatmapPoint(BaseModel):
    """Model for a single hint heatmap point."""
    area_name: str = Field(..., description="Name of the destination area")
    coordinates: Dict[str, float] = Field(..., description="Average coordinates for the area")
    order_probability: float = Field(..., description="Probability of getting an order", ge=0, le=1)
    confidence_level: str = Field(..., description="Human-readable confidence level")
    predicted_class: int = Field(..., description="Predicted class from chance model")
    incoming_clients_count: int = Field(..., description="Number of clients coming to this area")
    hint_score: float = Field(..., description="Composite hint score (0-100)", ge=0, le=100)
    recommendation: str = Field(..., description="Human-readable recommendation for drivers")


class HintHeatmapResponse(BaseModel):
    """Response model for hint heatmap data."""
    hint_points: List[HintHeatmapPoint] = Field(..., description="List of hint heatmap points")
    total_areas: int = Field(..., description="Total number of destination areas analyzed")
    total_clients: int = Field(..., description="Total number of clients analyzed")
    model_info: Dict[str, Any] = Field(..., description="Chance model metadata")
    generated_at: str = Field(..., description="Timestamp when data was generated")


class HintLocationResponse(BaseModel):
    """Response model for specific location hint."""
    location: Dict[str, float] = Field(..., description="Query coordinates")
    order_probability: float = Field(..., description="Probability of getting an order", ge=0, le=1)
    confidence_level: str = Field(..., description="Human-readable confidence level")
    nearby_destinations: List[Dict[str, Any]] = Field(..., description="Nearby client destinations")
    nearby_demand_count: int = Field(..., description="Number of nearby destinations")
    hint_score: float = Field(..., description="Composite hint score (0-100)", ge=0, le=100)
    recommendation: str = Field(..., description="Human-readable recommendation")
    search_radius_km: float = Field(..., description="Search radius used")


class TopHintLocationsResponse(BaseModel):
    """Response model for top hint locations."""
    top_locations: List[HintHeatmapPoint] = Field(..., description="Top recommended locations")
    limit: int = Field(..., description="Maximum number of locations returned")
    total_available: int = Field(..., description="Total locations available")
    model_info: Dict[str, Any] = Field(..., description="Chance model metadata")