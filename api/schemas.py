# em_prediction_service - API schemas (Pydantic models)
from datetime import datetime, date
from typing import Optional
from pydantic import BaseModel, Field


class PredictionPoint(BaseModel):
    """Single 15-min price prediction."""
    period: int = Field(..., ge=0, le=95, description="15-min period index (0-95)")
    time: str = Field(..., description="HH:MM format")
    price: float = Field(..., description="Predicted price (元/MWh)")
    segment: str = Field(..., description="valley / peak / base")


class PredictionSummary(BaseModel):
    avg_price: float
    peak_price: float
    peak_period: int
    valley_price: float
    valley_period: int


class PredictionsResponse(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD")
    model_version: str
    generated_at: str
    predictions: list[PredictionPoint]
    summary: PredictionSummary


class ModelInfo(BaseModel):
    version_name: str
    model_type: str
    metrics: Optional[dict] = None
    is_active: bool
    created_at: Optional[str] = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    models_loaded: int
    feature_version: str
    data_date_range: Optional[str] = None


class HistoryDay(BaseModel):
    """One day of actual vs predicted prices."""
    date: str = Field(..., description="YYYY-MM-DD")
    actual: list[Optional[float]] = Field(..., description="96 actual prices (null if unavailable)")
    predicted: list[Optional[float]] = Field(..., description="96 predicted prices (null if unavailable)")


class HistoryResponse(BaseModel):
    """Multi-day history comparison."""
    start: str
    end: str
    days: list[HistoryDay]
