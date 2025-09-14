"""API router for destination prediction endpoints."""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from ..schemas.prediction import (
    PredictionRequest, PredictionResponse, PredictionResult,
    ModelInfoResponse, HealthResponse, ErrorResponse, ClusterCenter,
    ClustersResponse, HeatmapResponse,
    ChanceRequest, ChanceResponse, ChanceModelInfoResponse,
    Client, ClientAnalysis, ClientsResponse,
    DestinationAreaPrediction, AreaBasedPredictionResponse,
    HintLocationRequest, HintHeatmapPoint, HintHeatmapResponse,
    HintLocationResponse, TopHintLocationsResponse
)
from ..services.model_service import get_model_service
from ..services.client_service import get_client_service
from ..services.hint_service import get_hint_service
from ..ml.chance_inference import get_chance_inference_instance


router = APIRouter(prefix="/api/v1", tags=["prediction"])


@router.post("/predict", response_model=AreaBasedPredictionResponse)
async def predict_destination(request: PredictionRequest):
    """
    Predict destination areas based on client trips from nearby locations.
    
    This endpoint analyzes real client trip data to predict where people
    typically go from the specified starting point.
    
    - **start_lat**: Starting latitude coordinate (-90 to 90)
    - **start_lng**: Starting longitude coordinate (-180 to 180)
    
    Returns top-3 destination areas with percentages based on actual client trips.
    """
    try:
        client_service = get_client_service()
        
        # Get destination predictions from client data
        predictions = client_service.get_destination_predictions_by_area(
            start_lat=request.start_lat,
            start_lng=request.start_lng,
            radius_km=1.0
        )
        
        # Convert to response format
        prediction_results = [
            DestinationAreaPrediction(
                destination_area=pred["destination_area"],
                trip_count=pred["trip_count"],
                percentage=pred["percentage"],
                coordinates=pred["coordinates"]
            )
            for pred in predictions
        ]
        
        return AreaBasedPredictionResponse(
            predictions=prediction_results,
            total_nearby_clients=sum(pred["trip_count"] for pred in predictions),
            search_radius_km=1.0,
            query_location={"lat": request.start_lat, "lng": request.start_lng}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to predict destinations: {str(e)}"
        )


@router.post("/predict/ml", response_model=PredictionResponse)
async def predict_destination_ml(request: PredictionRequest):
    """
    Predict destination clusters using ML model (alternative to area-based prediction).
    
    - **start_lat**: Starting latitude coordinate (-90 to 90)
    - **start_lng**: Starting longitude coordinate (-180 to 180)
    
    Returns top-3 destination cluster predictions with probabilities using ML model.
    """
    try:
        model_service = get_model_service()
        
        # Get predictions
        predictions = await model_service.predict_destination(
            start_lat=request.start_lat,
            start_lng=request.start_lng
        )
        
        # Get model info
        model_info = await model_service.get_model_info()
        
        # Convert to response format
        prediction_results = [
            PredictionResult(
                cluster_id=pred["cluster_id"],
                probability=pred["probability"],
                cluster_center=ClusterCenter(
                    lat=pred["cluster_center"]["lat"],
                    lng=pred["cluster_center"]["lng"]
                )
            )
            for pred in predictions
        ]
        
        # Create request summary
        request_summary = {
            "start_point": {"lat": request.start_lat, "lng": request.start_lng}
        }
        
        return PredictionResponse(
            predictions=prediction_results,
            model_info=model_info,
            request_summary=request_summary
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ML prediction failed: {str(e)}"
        )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the loaded model including accuracy metrics and configuration.
    """
    try:
        model_service = get_model_service()
        info = await model_service.get_model_info()
        
        return ModelInfoResponse(**info)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get("/clusters", response_model=ClustersResponse)
async def get_clusters():
    """
    Get detailed information about all destination clusters including coordinates,
    sizes, and descriptions.
    
    Returns information about all available destination clusters with their:
    - Cluster ID
    - Center coordinates (lat, lng) 
    - Number of destinations in cluster
    - Human-readable description
    """
    try:
        model_service = get_model_service()
        clusters_info = await model_service.get_clusters_info()
        
        return ClustersResponse(**clusters_info)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get clusters info: {str(e)}"
        )


@router.get("/heatmap", response_model=HeatmapResponse)
async def get_heatmap():
    """
    Get heatmap data showing clusters of trip start points (A-points).
    
    This endpoint analyzes the original trip data to cluster start points and show
    where trips typically begin. Returns data suitable for creating heatmaps
    showing trip origin concentrations.
    
    - **cluster_id**: Start point cluster identifier
    - **start_center**: Center coordinates of the start point cluster
    - **trip_count**: Total number of trips starting from this cluster
    - **avg_distance**: Average distance within cluster (dispersion)
    
    Data is sorted by trip count (most popular start areas first).
    """
    try:
        model_service = get_model_service()
        heatmap_data = await model_service.get_heatmap_data()
        
        return HeatmapResponse(**heatmap_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate heatmap data: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Check the health status of the prediction service and model availability.
    """
    try:
        model_service = get_model_service()
        health_status = model_service.health_check()
        
        return HealthResponse(**health_status)
        
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            error=str(e)
        )


@router.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Destination Prediction API with HINT System",
        "version": "2.1.0", 
        "description": "API with client database predictions and HINT system for driver guidance",
        "endpoints": {
            "predict": "/api/v1/predict (NEW: area-based predictions from client data)",
            "predict_ml": "/api/v1/predict/ml (ML model predictions)",
            "hint_heatmap": "/api/v1/hint/heatmap (NEW: driver hint heatmap)",
            "hint_top": "/api/v1/hint/top (NEW: top driver locations)",
            "hint_check": "/api/v1/hint/check (NEW: check specific location)",
            "model_info": "/api/v1/model/info",
            "clusters": "/api/v1/clusters",
            "heatmap": "/api/v1/heatmap", 
            "chance": "/api/v1/chance",
            "chance_info": "/api/v1/chance/info",
            "clients": "/api/v1/clients",
            "client_analysis": "/api/v1/clients/{id}/analyze",
            "clients_summary": "/api/v1/clients/summary",
            "health": "/api/v1/health"
        },
        "docs": "/docs"
    }


@router.get("/clients", response_model=ClientsResponse)
async def get_clients():
    """
    Get list of all clients with their information.
    
    Returns all clients from the database with their current locations,
    destinations, and trip preferences.
    """
    try:
        client_service = get_client_service()
        clients = client_service.load_clients()
        
        return ClientsResponse(
            clients=clients,
            total_clients=len(clients)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load clients: {str(e)}"
        )


@router.get("/clients/{client_id}/analyze", response_model=ClientAnalysis)
async def analyze_client(client_id: int):
    """
    Perform comprehensive AI analysis of a specific client.
    
    This endpoint combines both AI models to provide:
    - Destination predictions based on current location (destination_model)
    - Probability that a ride starts at current location (chance_model)
    - Probability that a ride starts at destination location (chance_model)
    
    **Path Parameters:**
    - **client_id**: ID of the client to analyze (1-20)
    """
    try:
        client_service = get_client_service()
        analysis = await client_service.analyze_client(client_id)
        
        return analysis
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/clients/summary")
async def get_clients_summary():
    """
    Get summary statistics about all clients.
    
    Returns aggregated information about client demographics, trip purposes,
    transport preferences, and trip durations.
    """
    try:
        client_service = get_client_service()
        summary = client_service.get_clients_summary()
        
        return summary
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}"
        )


@router.post("/chance", response_model=ChanceResponse)
async def predict_ride_start_chance(request: ChanceRequest):
    """
    Predict the probability that a ride starts at given coordinates.
    
    This endpoint uses a machine learning model trained on historical trip data
    to estimate the likelihood that a new ride will begin at the specified location.
    
    - **lat**: Latitude coordinate (-90 to 90)
    - **lng**: Longitude coordinate (-180 to 180)
    
    Returns probability score (0.0 to 1.0) and confidence level.
    Higher scores indicate locations where rides are more likely to start.
    """
    try:
        chance_inference = get_chance_inference_instance()
        result = chance_inference.predict_ride_start_probability(
            lat=request.lat,
            lng=request.lng
        )
        
        return ChanceResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chance prediction failed: {str(e)}"
        )


@router.get("/chance/info", response_model=ChanceModelInfoResponse)
async def get_chance_model_info():
    """
    Get information about the chance model including performance metrics.
    
    Returns model type, accuracy, AUC score, feature importance and other metadata.
    """
    try:
        chance_inference = get_chance_inference_instance()
        info = chance_inference.get_model_info()
        
        return ChanceModelInfoResponse(**info)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chance model info: {str(e)}"
        )


# HINT System Endpoints

@router.get("/hint/heatmap", response_model=HintHeatmapResponse)
async def get_hint_heatmap():
    """
    Get HINT heatmap data showing optimal locations for drivers to wait for orders.
    
    This endpoint analyzes client destinations and uses the chance model to predict
    where drivers are most likely to get new orders. Perfect for creating driver
    guidance heatmaps.
    
    Returns areas sorted by hint score (best opportunities first).
    """
    try:
        from datetime import datetime
        
        hint_service = get_hint_service()
        chance_inference = get_chance_inference_instance()
        
        # Get hint heatmap data
        hint_points_data = hint_service.calculate_order_probability_heatmap()
        
        # Convert to response format
        hint_points = [
            HintHeatmapPoint(**point_data)
            for point_data in hint_points_data
        ]
        
        # Get model info
        model_info = chance_inference.get_model_info()
        
        return HintHeatmapResponse(
            hint_points=hint_points,
            total_areas=len(hint_points),
            total_clients=sum(point.incoming_clients_count for point in hint_points),
            model_info=model_info,
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate hint heatmap: {str(e)}"
        )


@router.get("/hint/top", response_model=TopHintLocationsResponse)
async def get_top_hint_locations(limit: int = 5):
    """
    Get top recommended locations for drivers to wait for orders.
    
    Query Parameters:
    - limit: Maximum number of locations to return (default 5, max 20)
    
    Returns the best locations sorted by hint score.
    """
    try:
        # Validate limit
        if limit < 1 or limit > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be between 1 and 20"
            )
        
        hint_service = get_hint_service()
        chance_inference = get_chance_inference_instance()
        
        # Get top locations
        top_locations_data = hint_service.get_top_hint_locations(limit=limit)
        
        # Convert to response format
        top_locations = [
            HintHeatmapPoint(**location_data)
            for location_data in top_locations_data
        ]
        
        # Get total available locations
        all_locations = hint_service.calculate_order_probability_heatmap()
        
        # Get model info
        model_info = chance_inference.get_model_info()
        
        return TopHintLocationsResponse(
            top_locations=top_locations,
            limit=limit,
            total_available=len(all_locations),
            model_info=model_info
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get top hint locations: {str(e)}"
        )


@router.post("/hint/check", response_model=HintLocationResponse)
async def check_hint_location(request: HintLocationRequest):
    """
    Check hint information for a specific location.
    
    Analyzes the order probability for a given coordinate and provides
    recommendations for drivers.
    
    - **lat**: Latitude coordinate (-90 to 90)
    - **lng**: Longitude coordinate (-180 to 180)  
    - **radius_km**: Search radius for nearby destinations (0.1 to 5.0 km)
    
    Returns probability, nearby demand, and driver recommendations.
    """
    try:
        hint_service = get_hint_service()
        
        # Get hint information for the location
        hint_data = hint_service.get_hint_for_location(
            lat=request.lat,
            lng=request.lng,
            radius_km=request.radius_km
        )
        
        return HintLocationResponse(**hint_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check hint location: {str(e)}"
        )