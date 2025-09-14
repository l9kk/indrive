# Decentra3 - Destination Prediction API

FastAPI service for predicting taxi destinations with HINT system for driver guidance.

## 🚀 Quick Docker Deployment

### Prerequisites
- Docker & Docker Compose installed

### Deploy with Docker

```bash
# Quick deployment (Linux/Mac)
./deploy.sh

# Windows
deploy.bat

# Or manually:
docker-compose up --build -d
```

### API Endpoints

- **API Root**: http://localhost:8001/api/v1/
- **Documentation**: http://localhost:8001/docs
- **Clients**: `GET /api/v1/clients`
- **Predict Destination**: `POST /api/v1/predict`
- **HINT Heatmap**: `GET /api/v1/hint/heatmap`
- **HINT Top Locations**: `GET /api/v1/hint/top`
- **HINT Check Location**: `POST /api/v1/hint/check`

### Example Requests

```bash
# Get clients list
curl http://localhost:8001/api/v1/clients

# Predict destination
curl -X POST "http://localhost:8001/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"start_lat": 51.0845, "start_lng": 71.4127}'

# Check HINT for location
curl -X POST "http://localhost:8001/api/v1/hint/check" \
     -H "Content-Type: application/json" \
     -d '{"lat": 51.0845, "lng": 71.4127, "radius_km": 0.5}'
```

### Docker Commands

```bash
# View logs
docker-compose logs -f

# Stop containers
docker-compose down

# Rebuild and restart
docker-compose up --build -d

# Check container status
docker-compose ps
```

## Project Structure

```
decentra3/
├── data/
│   └── geo_locations_labeled_advanced.csv
├── src/
│   ├── main.py                 # FastAPI application entry point
│   ├── core/
│   │   └── config.py          # Configuration settings
│   ├── routers/
│   │   └── prediction.py      # API endpoints
│   ├── schemas/
│   │   └── prediction.py      # Pydantic models
│   ├── services/
│   │   └── model_service.py   # Business logic layer
│   ├── ml/
│   │   ├── train.py          # Model training script
│   │   ├── inference.py      # Model inference
│   │   └── model.pkl         # Trained model (generated)
│   └── utils/
│       └── preprocessing.py   # Data preprocessing utilities
├── requirements.txt           # Python dependencies
├── train_model.py            # Standalone training script
├── run_server.py            # Server startup script
└── README.md                # This file
```

## Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**

   ```bash
   python train_model.py
   ```

3. **Run the API server:**

   ```bash
   python run_server.py
   ```

4. **Access the API:**
   - API Documentation: http://localhost:8001/docs
   - Health Check: http://localhost:8001/api/v1/health
   - Predictions: POST http://localhost:8001/api/v1/predict

## API Usage

### Predict Destination

```bash
curl -X POST "http://localhost:8001/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "start_lat": 51.0829583,
       "start_lng": 71.4223554,
       "direction_points": [
         {"lat": 51.0834556, "lng": 71.4225399},
         {"lat": 51.0873756, "lng": 71.4194807}
       ]
     }'
```

### Example Response

```json
{
  "predictions": [
    {
      "cluster_id": 2,
      "probability": 0.45,
      "cluster_center": { "lat": 51.1234, "lng": 71.5678 }
    },
    {
      "cluster_id": 5,
      "probability": 0.32,
      "cluster_center": { "lat": 51.2345, "lng": 71.6789 }
    },
    {
      "cluster_id": 1,
      "probability": 0.23,
      "cluster_center": { "lat": 51.3456, "lng": 71.789 }
    }
  ],
  "model_info": {
    "model_type": "CatBoostClassifier",
    "n_clusters": 7,
    "top_1_accuracy": 0.78,
    "top_3_accuracy": 0.92
  },
  "request_summary": {
    "start_point": { "lat": 51.0829583, "lng": 71.4223554 },
    "n_direction_points": 2
  }
}
```

## Features

- **Clustering**: KMeans clustering of destination points into 5-10 zones
- **Feature Engineering**: Start coordinates, direction points, and displacement vectors
- **ML Models**: CatBoost or Logistic Regression for classification
- **API**: FastAPI with automatic documentation and validation
- **Production Ready**: Proper error handling, logging, and health checks

## Model Details

The system:

1. Extracts trips from GPS data (A → C → B points)
2. Clusters destination points (B) using KMeans
3. Builds features from start point (A) and direction points (C)
4. Trains a classifier to predict destination cluster
5. Provides top-3 predictions with probabilities
