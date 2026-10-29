import os
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("InferenceAPI")

# 2. Initialize FastAPI
app = FastAPI(
    title="California Housing Price Estimation API",
    description="Real-time inference service for block group property appraisal estimations.",
    version="1.0.0"
)

# 3. Load Model Artifacts
ARTIFACT_PATH = "/content/model_artifacts/housing_v1_pipeline.pkl"
if not os.path.exists(ARTIFACT_PATH):
    # Local development fallback path
    ARTIFACT_PATH = "model_artifacts/housing_v1_pipeline.pkl"

try:
    logger.info(f"Loading model artifacts from {ARTIFACT_PATH}...")
    artifacts = joblib.load(ARTIFACT_PATH)
    model = artifacts["model"]
    model_features = artifacts["features"]
    logger.info("Model artifacts loaded successfully!")
except Exception as e:
    logger.error(f"Critical Error loading model artifacts: {e}")
    raise RuntimeError(f"Failed to start API due to missing/invalid model pipeline: {e}")

# 4. Input Verification Schema (Data Contract)
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

# 5. Routes
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model_version": "1.0.0",
        "features_required": model_features
    }

@app.post("/predict")
def predict(payload: HousingInferencePayload):
    try:
        # Log incoming request safely
        logger.info(f"Received prediction request for coordinates: ({payload.latitude}, {payload.longitude})")

        # Convert payload to dataframe matching feature ordering
        input_dict = payload.model_dump()
        input_df = pd.DataFrame([input_dict])[model_features]

        # Predict
        prediction = model.predict(input_df)[0]

        # Return structured response, ensuring native Python types for serialization
        return {
            "prediction_usd": round(float(prediction), 2),
            "model_version": "1.0.0",
            "capped_warning": bool(prediction >= 500000.0)
        }
    except Exception as e:
        logger.error(f"Inference runtime error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal inference exception: {str(e)}")
