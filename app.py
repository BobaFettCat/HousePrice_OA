import os
import time
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# 1. Setup Structured Telemetry Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ProductionInferenceAPI")

# 2. Initialize FastAPI App with Telemetry metadata
app = FastAPI(
    title="Optimized California Housing Price Estimation API",
    description="Real-time inference service with caching, monitoring, and telemetry.",
    version="2.0.0"
)

# 3. Simple In-Memory Cache for Optimization
# Stores request signatures as keys and prediction results as values to minimize redundant calculations
prediction_cache = {}

# Telemetry Counters
metrics = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "total_latency_ms": 0.0
}

# 4. Load Model Artifacts
ARTIFACT_PATH = "/content/model_artifacts/housing_v1_pipeline.pkl"
if not os.path.exists(ARTIFACT_PATH):
    ARTIFACT_PATH = "model_artifacts/housing_v1_pipeline.pkl"

try:
    logger.info(f"Loading model artifacts from {ARTIFACT_PATH}...")
    artifacts = joblib.load(ARTIFACT_PATH)
    model = artifacts["model"]
    model_features = artifacts["features"]
    logger.info("Model artifacts loaded successfully!")
except Exception as e:
    logger.error(f"Critical Error loading model artifacts: {e}")
    raise RuntimeError(f"Failed to start API: {e}")

# 5. Input Validation Schema
class HousingInferencePayload(BaseModel):
    longitude: float = Field(..., description="Geographic longitude")
    latitude: float = Field(..., description="Geographic latitude")
    housing_median_age: float = Field(..., ge=0, description="Median age of houses in block group")
    total_rooms: float = Field(..., gt=0, description="Total room count")
    total_bedrooms: float = Field(..., gt=0, description="Total bedroom count")
    population: float = Field(..., gt=0, description="Block group population")
    households: float = Field(..., gt=0, description="Total households")
    median_income: float = Field(..., gt=0, description="Median household income in block group")

    @field_validator('latitude')
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if not (32.0 <= v <= 42.0):
            raise ValueError("Latitude outside valid California limits [32.0, 42.0]")
        return v

    @field_validator('longitude')
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if not (-124.5 <= v <= -114.0):
            raise ValueError("Longitude outside valid California limits [-124.5, -114.0]")
        return v

# 6. Routes
@app.get("/")
def health_check():
    # Returns active model information and telemetry dashboard link
    return {
        "status": "healthy",
        "model_version": "2.0.0",
        "telemetry_summary": {
            "total_requests_received": metrics["total_requests"],
            "cache_hit_rate_pct": round((metrics["cache_hits"] / max(1, metrics["total_requests"])) * 100, 2),
            "average_latency_ms": round(metrics["total_latency_ms"] / max(1, metrics["total_requests"]), 3)
        }
    }

@app.post("/predict")
def predict(payload: HousingInferencePayload):
    start_time = time.time()
    metrics["total_requests"] += 1
    
    try:
        # Generate a unique cache key from inputs
        cache_key = (
            payload.longitude, payload.latitude, payload.housing_median_age,
            payload.total_rooms, payload.total_bedrooms, payload.population,
            payload.households, payload.median_income
        )

        # Optimisation Layer: Check in-memory cache
        if cache_key in prediction_cache:
            metrics["cache_hits"] += 1
            execution_time_ms = (time.time() - start_time) * 1000.0
            metrics["total_latency_ms"] += execution_time_ms
            
            response = prediction_cache[cache_key]
            logger.info(f"Cache HIT for coords ({payload.latitude}, {payload.longitude}). Latency: {execution_time_ms:.3f}ms")
            return response

        # Cache Miss - Proceed to Inference
        metrics["cache_misses"] += 1
        input_dict = payload.model_dump()
        input_df = pd.DataFrame([input_dict])[model_features]
        
        # Predict using core estimator
        prediction = model.predict(input_df)[0]
        prediction_usd = round(float(prediction), 2)
        capped_warning = bool(prediction_usd >= 500000.0)

        response_payload = {
            "prediction_usd": prediction_usd,
            "model_version": "2.0.0",
            "capped_warning": capped_warning,
            "cached_response": False
        }

        # Save output to cache
        prediction_cache[cache_key] = response_payload.copy()
        # Tag response payload to distinguish cache sources from new evaluations
        response_payload["cached_response"] = True
        prediction_cache[cache_key] = response_payload

        # Log Latency
        execution_time_ms = (time.time() - start_time) * 1000.0
        metrics["total_latency_ms"] += execution_time_ms
        logger.info(f"Cache MISS. Model prediction executed for coords ({payload.latitude}, {payload.longitude}). Latency: {execution_time_ms:.3f}ms")

        # Return payload with cached status flagged False for this initial run
        final_response = response_payload.copy()
        final_response["cached_response"] = False
        return final_response

    except Exception as e:
        logger.error(f"Inference runtime error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal inference exception: {str(e)}")
