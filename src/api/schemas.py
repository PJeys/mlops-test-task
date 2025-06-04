from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Define the expected input schema for a single sensor reading
class SensorDataInput(BaseModel):
    # Required fields based on sample data and preprocessor expectations
    sensor_id: int = Field(..., description="Unique identifier for the sensor.")
    timestamp: str = Field(..., description="Timestamp of the sensor reading (ISO 8601 format).")
    temperature_sensor: Optional[float] = Field(None, description="Temperature reading from the sensor.")
    firmware_version_sensor: Optional[str] = Field(None, description="Firmware version of the sensor.")
    gateway_id: Optional[str] = Field(None, description="Identifier for the gateway.")
    timestamp_gateway: Optional[str] = Field(None, description="Timestamp of the gateway reading (ISO 8601 format).")
    ihs_to_gw_transmission_strength: Optional[float] = Field(None, description="Transmission strength between IHS and gateway.")
    firmware_version_gateway: Optional[str] = Field(None, description="Firmware version of the gateway.")
    temperature_gateway: Optional[float] = Field(None, description="Temperature reading from the gateway.")
    experiment_point_id: Optional[str] = Field(None, description="Identifier for the experiment point.")

    class Config:
        # Example for FastAPI's auto-generated OpenAPI schema
        schema_extra = {
            "example": {
                "sensor_id": 123,
                "timestamp": "2023-10-27T10:30:00Z",
                "temperature_sensor": 28.5,
                "firmware_version_sensor": "1.0.1",
                "gateway_id": "gw001",
                "timestamp_gateway": "2023-10-27T10:30:05Z",
                "ihs_to_gw_transmission_strength": -65.0,
                "firmware_version_gateway": "2.1.0",
                "temperature_gateway": 25.1,
                "experiment_point_id": "expABC"
            }
        }

# Define the expected output schema for the prediction
class PredictionResponse(BaseModel):
    predicted_strength: str = Field(..., description="The predicted colony strength (S, M, or L).")
    prediction_probabilities: Dict[str, float] = Field(..., description="Probabilities for each strength class.")
    model_version: str = Field(..., description="Version of the model used for prediction.")
    message: str = Field("Prediction successful.", description="A message regarding the prediction.")

    class Config:
        schema_extra = {
            "example": {
                "predicted_strength": "M",
                "prediction_probabilities": {"S": 0.1, "M": 0.7, "L": 0.2},
                "model_version": "20231027_103045",
                "message": "Prediction successful."
            }
        }

class HealthCheckResponse(BaseModel):
    status: str = Field("ok", description="Status of the API service.")
    model_loaded: bool = Field(..., description="Indicates if the ML model is loaded.")
    preprocessor_loaded: bool = Field(..., description="Indicates if the preprocessor is loaded.")
    model_version: Optional[str] = Field(None, description="Version of the loaded model.")
    message: Optional[str] = Field(None, description="Additional health message.")