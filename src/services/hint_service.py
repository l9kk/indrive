"""Hint service for providing intelligent suggestions to drivers about where to wait for orders."""

import json
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict
import statistics

from ..core.config import settings
from ..ml.chance_inference import get_chance_inference_instance
from ..services.client_service import get_client_service


class HintService:
    """Service for providing driver hints about optimal locations to wait for orders."""
    
    def __init__(self):
        """Initialize hint service."""
        self.chance_inference = get_chance_inference_instance()
        self.client_service = get_client_service()
    
    def calculate_order_probability_heatmap(self) -> List[Dict[str, Any]]:
        """
        Calculate heatmap data showing probability of getting new orders in different areas.
        
        Logic: When clients arrive at destinations, there's a chance someone in that area
        will want to start a new trip. We use the chance model to calculate this probability.
        
        Returns:
            List of heatmap points with order probability data
        """
        # Get full client data for internal processing
        full_clients = self.client_service.load_full_clients_data()
        
        # Group destinations by area/district
        destination_areas = defaultdict(list)
        
        for client in full_clients:
            area_name = client['destination']['address']
            destination_info = {
                'lat': client['destination']['lat'],
                'lng': client['destination']['lng'],
                'client_id': client['id']
            }
            destination_areas[area_name].append(destination_info)
        
        # Calculate chance probability for each area
        heatmap_points = []
        
        for area_name, destinations in destination_areas.items():
            # Calculate average coordinates for the area
            avg_lat = statistics.mean(dest['lat'] for dest in destinations)
            avg_lng = statistics.mean(dest['lng'] for dest in destinations)
            
            # Get order probability using chance model
            chance_result = self.chance_inference.predict_ride_start_probability(
                lat=avg_lat,
                lng=avg_lng
            )
            
            # Create heatmap point
            heatmap_point = {
                'area_name': area_name,
                'coordinates': {
                    'lat': avg_lat,
                    'lng': avg_lng
                },
                'order_probability': chance_result['ride_start_probability'],
                'confidence_level': chance_result['confidence_level'],
                'predicted_class': chance_result['predicted_class'],
                'incoming_clients_count': len(destinations),
                'hint_score': self._calculate_hint_score(
                    chance_result['ride_start_probability'],
                    len(destinations)
                ),
                'recommendation': self._get_recommendation(
                    chance_result['ride_start_probability'],
                    len(destinations)
                )
            }
            
            heatmap_points.append(heatmap_point)
        
        # Sort by hint score (best opportunities first)
        heatmap_points.sort(key=lambda x: x['hint_score'], reverse=True)
        
        return heatmap_points
    
    def _calculate_hint_score(self, probability: float, client_count: int) -> float:
        """
        Calculate a composite hint score combining probability and demand.
        
        Args:
            probability: Ride start probability from chance model
            client_count: Number of clients coming to this area
            
        Returns:
            Hint score (0-100, higher is better)
        """
        # Normalize client count (assume max 5 clients per area is high demand)
        demand_factor = min(client_count / 5.0, 1.0)
        
        # Combine probability and demand with weights
        hint_score = (probability * 0.7 + demand_factor * 0.3) * 100
        
        return round(hint_score, 2)
    
    def _get_recommendation(self, probability: float, client_count: int) -> str:
        """
        Get human-readable recommendation for drivers.
        
        Args:
            probability: Ride start probability
            client_count: Number of incoming clients
            
        Returns:
            Recommendation string
        """
        if probability >= 0.7 and client_count >= 3:
            return "🔥 Отличное место! Высокая вероятность заказов"
        elif probability >= 0.5 and client_count >= 2:
            return "👍 Хорошее место для ожидания заказов"
        elif probability >= 0.4 or client_count >= 2:
            return "⚡ Средняя активность, можно попробовать"
        elif probability >= 0.3:
            return "⏰ Низкая активность, лучше поискать другое место"
        else:
            return "❌ Очень низкая вероятность заказов"
    
    def get_top_hint_locations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get top recommended locations for drivers to wait for orders.
        
        Args:
            limit: Maximum number of locations to return
            
        Returns:
            List of top hint locations
        """
        heatmap_data = self.calculate_order_probability_heatmap()
        return heatmap_data[:limit]
    
    def get_hint_for_location(self, lat: float, lng: float, radius_km: float = 0.5) -> Dict[str, Any]:
        """
        Get hint information for a specific location.
        
        Args:
            lat: Latitude
            lng: Longitude  
            radius_km: Search radius in kilometers
            
        Returns:
            Hint information for the area
        """
        # Get chance probability for the exact location
        chance_result = self.chance_inference.predict_ride_start_probability(lat, lng)
        
        # Find nearby client destinations using full data
        full_clients = self.client_service.load_full_clients_data()
        nearby_destinations = []
        
        for client in full_clients:
            distance = self.client_service.calculate_distance(
                lat, lng,
                client['destination']['lat'], client['destination']['lng']
            )
            if distance <= radius_km:
                nearby_destinations.append({
                    'area': client['destination']['address'],
                    'distance_km': round(distance, 2)
                })
        
        return {
            'location': {'lat': lat, 'lng': lng},
            'order_probability': chance_result['ride_start_probability'],
            'confidence_level': chance_result['confidence_level'],
            'nearby_destinations': nearby_destinations,
            'nearby_demand_count': len(nearby_destinations),
            'hint_score': self._calculate_hint_score(
                chance_result['ride_start_probability'],
                len(nearby_destinations)
            ),
            'recommendation': self._get_recommendation(
                chance_result['ride_start_probability'],
                len(nearby_destinations)
            ),
            'search_radius_km': radius_km
        }
    
    def get_hints_for_area(self, min_lat: float, max_lat: float, 
                          min_lng: float, max_lng: float, 
                          grid_size: int = 5) -> Dict[str, Any]:
        """
        Get hint information for a rectangular area with grid analysis.
        
        Args:
            min_lat, max_lat: Latitude bounds
            min_lng, max_lng: Longitude bounds
            grid_size: Number of grid points per dimension
            
        Returns:
            Grid-based hint analysis for the area
        """
        area_hints = []
        
        # Create grid of points
        lat_step = (max_lat - min_lat) / (grid_size - 1) if grid_size > 1 else 0
        lng_step = (max_lng - min_lng) / (grid_size - 1) if grid_size > 1 else 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                lat = min_lat + i * lat_step
                lng = min_lng + j * lng_step
                
                hint = self.get_hint_for_location(lat, lng, radius_km=0.3)
                area_hints.append(hint)
        
        # Get full client data for analysis
        full_clients = self.client_service.load_full_clients_data()
        
        # Analyze destinations in area
        destinations_in_area = []
        for client in full_clients:
            dest_lat = client['destination']['lat']
            dest_lng = client['destination']['lng']
            
            if (min_lat <= dest_lat <= max_lat and 
                min_lng <= dest_lng <= max_lng):
                destinations_in_area.append({
                    'coordinates': {
                        'lat': dest_lat,
                        'lng': dest_lng
                    },
                    'area': client['destination']['address']
                })
        
        # Find best hint locations
        best_hints = sorted(area_hints, key=lambda x: x['hint_score'], reverse=True)[:3]
        
        return {
            'area_bounds': {
                'min_lat': min_lat, 'max_lat': max_lat,
                'min_lng': min_lng, 'max_lng': max_lng
            },
            'grid_analysis': area_hints,
            'destinations_in_area': destinations_in_area,
            'total_destinations': len(destinations_in_area),
            'best_waiting_spots': best_hints,
            'area_summary': {
                'avg_order_probability': sum(h['order_probability'] for h in area_hints) / len(area_hints),
                'max_hint_score': max(h['hint_score'] for h in area_hints),
                'grid_size': f"{grid_size}x{grid_size}"
            }
        }


# Global service instance
_hint_service_instance = None


def get_hint_service() -> HintService:
    """Get singleton hint service instance."""
    global _hint_service_instance
    if _hint_service_instance is None:
        _hint_service_instance = HintService()
    return _hint_service_instance