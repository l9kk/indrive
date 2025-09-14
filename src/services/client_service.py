"""Client service for managing client data and providing analysis."""

import json
import os
from typing import List, Dict, Any, Optional
from ..core.config import settings
from ..schemas.prediction import Client, ClientAnalysis, ClientsResponse, Location
from ..services.model_service import get_model_service
from ..ml.chance_inference import get_chance_inference_instance


class ClientService:
    """Service for managing clients and providing AI-powered analysis."""
    
    def __init__(self):
        """Initialize client service."""
        self.clients_file = os.path.join(
            os.path.dirname(settings.DATA_PATH), 
            "clients.json"
        )
        self._clients_cache = None
    
    def load_clients(self) -> List[Client]:
        """Load clients from JSON file and convert to API format (without sensitive fields)."""
        if self._clients_cache is not None:
            return self._clients_cache
        
        if not os.path.exists(self.clients_file):
            raise FileNotFoundError(f"Clients file not found: {self.clients_file}")
        
        with open(self.clients_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        clients = []
        for client_data in data['clients']:
            # Create client with only essential fields for API
            client = Client(
                id=client_data['id'],
                name=client_data['name'],
                age=client_data['age'],
                current_location={
                    'lat': client_data['current_location']['lat'],
                    'lng': client_data['current_location']['lng']
                },
                destination={
                    'lat': client_data['destination']['lat'],
                    'lng': client_data['destination']['lng']
                },
                estimated_duration=client_data['estimated_duration']
            )
            clients.append(client)
        
        self._clients_cache = clients
        return clients
    
    def load_full_clients_data(self) -> List[Dict[str, Any]]:
        """Load full client data from JSON file including all fields for internal processing."""
        if not os.path.exists(self.clients_file):
            raise FileNotFoundError(f"Clients file not found: {self.clients_file}")
        
        with open(self.clients_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data['clients']
    
    def get_client_by_id(self, client_id: int) -> Optional[Client]:
        """Get client by ID."""
        clients = self.load_clients()
        for client in clients:
            if client.id == client_id:
                return client
        return None
    
    async def analyze_client(self, client_id: int) -> ClientAnalysis:
        """
        Perform comprehensive analysis of a client using both AI models.
        
        Args:
            client_id: Client ID to analyze
            
        Returns:
            ClientAnalysis with destination predictions and chance probabilities
        """
        client = self.get_client_by_id(client_id)
        if not client:
            raise ValueError(f"Client with ID {client_id} not found")
        
        # Get model services
        model_service = get_model_service()
        chance_inference = get_chance_inference_instance()
        
        # Destination prediction from current location
        destination_predictions = await model_service.predict_destination(
            start_lat=client.current_location.lat,
            start_lng=client.current_location.lng
        )
        
        # Chance of ride starting at current location
        start_chance = chance_inference.predict_ride_start_probability(
            lat=client.current_location.lat,
            lng=client.current_location.lng
        )
        
        # Chance of ride starting at destination (for comparison)
        destination_chance = chance_inference.predict_ride_start_probability(
            lat=client.destination.lat,
            lng=client.destination.lng
        )
        
        return ClientAnalysis(
            client=client,
            destination_prediction=destination_predictions,
            start_chance=start_chance,
            destination_chance=destination_chance
        )
    
    def get_clients_summary(self) -> Dict[str, Any]:
        """Get summary statistics about clients."""
        clients = self.load_clients()
        
        # Age statistics
        ages = [client.age for client in clients]
        avg_age = sum(ages) / len(ages)
        
        # Trip purposes
        purposes = {}
        transports = {}
        durations = []
        
        for client in clients:
            purposes[client.trip_purpose] = purposes.get(client.trip_purpose, 0) + 1
            transports[client.preferred_transport] = transports.get(client.preferred_transport, 0) + 1
            durations.append(client.estimated_duration)
        
        avg_duration = sum(durations) / len(durations)
        
        return {
            "total_clients": len(clients),
            "average_age": round(avg_age, 1),
            "age_range": {"min": min(ages), "max": max(ages)},
            "trip_purposes": purposes,
            "transport_preferences": transports,
            "average_trip_duration": round(avg_duration, 1),
            "trip_duration_range": {"min": min(durations), "max": max(durations)}
        }
    
    def calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        Calculate distance between two points in kilometers using Haversine formula.
        """
        import math
        
        # Convert latitude and longitude from degrees to radians
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r
    
    def get_destination_predictions_by_area(self, start_lat: float, start_lng: float, radius_km: float = 1.0) -> List[Dict[str, Any]]:
        """
        Get destination predictions based on client trips from nearby areas.
        
        Args:
            start_lat: Starting latitude
            start_lng: Starting longitude  
            radius_km: Search radius in kilometers (default 1.0km)
            
        Returns:
            List of top-3 destination areas with percentages
        """
        full_clients = self.load_full_clients_data()
        
        # Find clients within radius
        nearby_clients = []
        for client in full_clients:
            distance = self.calculate_distance(
                start_lat, start_lng,
                client['current_location']['lat'], client['current_location']['lng']
            )
            if distance <= radius_km:
                nearby_clients.append(client)
        
        if not nearby_clients:
            # If no clients in radius, expand search
            radius_km *= 2
            for client in full_clients:
                distance = self.calculate_distance(
                    start_lat, start_lng,
                    client['current_location']['lat'], client['current_location']['lng']
                )
                if distance <= radius_km:
                    nearby_clients.append(client)
        
        if not nearby_clients:
            # If still no clients, return empty result
            return []
        
        # Group destinations by area (using address from full data)
        destination_counts = {}
        destination_coords = {}
        
        for client in nearby_clients:
            area = client['destination']['address']
            if area not in destination_counts:
                destination_counts[area] = 0
                destination_coords[area] = {
                    "lat": client['destination']['lat'],
                    "lng": client['destination']['lng']
                }
            destination_counts[area] += 1
        
        # Calculate percentages and create results
        total_trips = len(nearby_clients)
        results = []
        
        for area, count in destination_counts.items():
            percentage = (count / total_trips) * 100
            results.append({
                "destination_area": area,
                "trip_count": count,
                "percentage": round(percentage, 2),
                "coordinates": destination_coords[area]
            })
        
        # Sort by trip count and return top 3
        results.sort(key=lambda x: x['trip_count'], reverse=True)
        return results[:3]


# Global service instance
_client_service_instance = None


def get_client_service() -> ClientService:
    """Get singleton client service instance."""
    global _client_service_instance
    if _client_service_instance is None:
        _client_service_instance = ClientService()
    return _client_service_instance