# Decentra4 - Destination Prediction API

FastAPI service for predicting taxi destinations with HINT system for driver guidance.

## Features

- **Clustering**: KMeans clustering of destination points into 5-10 zones
- **Feature Engineering**: Start coordinates, direction points, and displacement vectors
- **ML Models**: CatBoost or Logistic Regression for classification
- **API**: FastAPI with automatic documentation and validation
- **Production Ready**: Proper error handling, logging, and health checks
