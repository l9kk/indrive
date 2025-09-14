"""Model service for handling prediction requests."""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from collections import defaultdict
from ..ml.inference import get_inference_instance, predict_destination
from ..core.config import settings


class ModelService:
    """Service class for model operations."""
    
    def __init__(self):
        """Initialize model service."""
        self.inference = get_inference_instance()
    
    async def predict_destination(self, start_lat: float, start_lng: float) -> List[Dict[str, Any]]:
        """
        Predict destination clusters for a trip.
        
        Args:
            start_lat: Starting latitude
            start_lng: Starting longitude
        
        Returns:
            List of predictions with cluster_id, probability, and cluster_center
        """
        try:
            predictions = self.inference.predict_top_k(
                start_lat=start_lat,
                start_lng=start_lng
            )
            return predictions
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        try:
            info = self.inference.get_model_info()
            return info
        except Exception as e:
            raise RuntimeError(f"Failed to get model info: {str(e)}")
    
    async def get_clusters_info(self) -> Dict[str, Any]:
        """Get detailed information about all clusters."""
        try:
            info = self.inference.get_model_info()
            clusters_data = []
            
            # Get cluster centers and metadata
            cluster_centers = self.inference.kmeans.cluster_centers_
            cluster_labels = info.get("cluster_labels", [])
            
            # Get cluster sizes from training metadata if available
            cluster_sizes = info.get("cluster_sizes", {})
            
            for i, cluster_id in enumerate(cluster_labels):
                cluster_info = {
                    "cluster_id": int(cluster_id),
                    "center": {
                        "lat": float(cluster_centers[cluster_id][0]),
                        "lng": float(cluster_centers[cluster_id][1])
                    },
                    "size": cluster_sizes.get(cluster_id, 0),
                    "description": f"Destination cluster {cluster_id}"
                }
                clusters_data.append(cluster_info)
            
            return {
                "clusters": clusters_data,
                "total_clusters": len(cluster_labels),
                "model_info": info
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get clusters info: {str(e)}")
    
    async def get_heatmap_data(self) -> Dict[str, Any]:
        """Get heatmap data showing clusters of trip start points (A-points)."""
        try:
            # Load original data
            df = pd.read_csv(settings.DATA_PATH)
            print(f"Loaded data with columns: {df.columns.tolist()}")
            
            # Get model info
            info = self.inference.get_model_info()
            
            # Extract A points (start points)
            A_points = df[df['status'] == 'A'][['randomized_id', 'lat', 'lng']].copy()
            print(f"Found {len(A_points)} start points to cluster")
            
            # Cluster A points (start points) using KMeans with same number of clusters
            from sklearn.cluster import KMeans
            A_coords = A_points[['lat', 'lng']].values
            A_kmeans = KMeans(n_clusters=settings.N_CLUSTERS, random_state=settings.RANDOM_STATE, n_init=10)
            A_cluster_labels = A_kmeans.fit_predict(A_coords)
            A_points['cluster_id'] = A_cluster_labels
            
            # Get cluster centers
            cluster_centers = A_kmeans.cluster_centers_
            
            # Generate heatmap data for each cluster
            clusters_heatmap = []
            total_trips = 0
            
            for cluster_id in range(len(cluster_centers)):
                # Get all trips starting in this cluster
                cluster_trips = A_points[A_points['cluster_id'] == cluster_id]
                
                if len(cluster_trips) == 0:
                    continue
                
                # Calculate cluster center
                start_center = cluster_centers[cluster_id]
                
                # Calculate average distance within cluster (dispersion)
                distances = []
                for _, row in cluster_trips.iterrows():
                    # Distance from each point to cluster center
                    dist = np.sqrt(
                        (row['lat'] - start_center[0])**2 + 
                        (row['lng'] - start_center[1])**2
                    )
                    distances.append(dist)
                avg_distance = np.mean(distances) if distances else 0
                
                cluster_heatmap = {
                    "cluster_id": int(cluster_id),
                    "start_center": {
                        "lat": float(start_center[0]),
                        "lng": float(start_center[1])
                    },
                    "trip_count": int(len(cluster_trips)),
                    "avg_distance": float(avg_distance)
                }
                
                clusters_heatmap.append(cluster_heatmap)
                total_trips += len(cluster_trips)
            
            # Sort clusters by trip count (descending)
            clusters_heatmap.sort(key=lambda x: x['trip_count'], reverse=True)
            
            return {
                "clusters": clusters_heatmap,
                "total_clusters": len(clusters_heatmap),
                "total_trips": total_trips,
                "model_info": info
            }
            
        except Exception as e:
            print(f"Heatmap error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate heatmap data: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check if model service is healthy."""
        try:
            # Simple check - verify model is loaded
            info = self.inference.get_model_info()
            return {
                "status": "healthy",
                "model_loaded": True,
                "model_type": info.get("model_type", "unknown"),
                "n_clusters": info.get("n_clusters", 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "model_loaded": False,
                "error": str(e)
            }


# Global service instance
_service_instance = None


def get_model_service() -> ModelService:
    """Get singleton model service instance."""
    global _service_instance
    # Always create new instance to ensure fresh code
    _service_instance = ModelService()
    return _service_instance