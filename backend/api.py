"""
FastAPI Backend Simulator for Traffic Congestion ML Model
==========================================================
Simulates ML model predictions with multi-horizon forecasts.

Returns predictions with:
- Location (latitude/longitude as integers)
- 6 time horizons (t+1h through t+6h)
- Normalized congestion values (can be negative, range ~-1 to 1)
"""

from datetime import datetime, timedelta
from typing import List
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ── Models ───────────────────────────────────────────────────────────────────


class HorizonPrediction(BaseModel):
    horizon: str  # e.g., "t+1h", "t+2h", etc.
    DateTime: str
    predicted_congestion: float  # Normalized value, can be negative


class LocationPrediction(BaseModel):
    latitude: int  # Integer format (e.g., 40742511 = 40.742511°)
    longitude: int  # Integer format (e.g., -73949134 = -73.949134°)
    horizons: List[HorizonPrediction]


class OutputList(BaseModel):
    predictions: List[LocationPrediction]


# ── App Setup ────────────────────────────────────────────────────────────────


app = FastAPI(title="Traffic Congestion Simulator API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Simulation Data ──────────────────────────────────────────────────────────


# Manhattan-like bounding box (as integers)
LATITUDE_RANGE = (40700000, 40800000)   # 40.700 to 40.800
LONGITUDE_RANGE = (-74020000, -73940000)  # -74.020 to -73.940

NUM_HORIZONS = 6  # t+1h through t+6h

# Define key hotspot areas
HOTSPOTS = [
    {"lat": 40758000, "lon": -73985000, "name": "Times Square", "base": 0.7},
    {"lat": 40748000, "lon": -73986000, "name": "Herald Square", "base": 0.65},
    {"lat": 40761000, "lon": -73977000, "name": "Central Park South", "base": 0.6},
    {"lat": 40706000, "lon": -74009000, "name": "Financial District", "base": 0.75},
    {"lat": 40715000, "lon": -74005000, "name": "Tribeca", "base": 0.5},
    {"lat": 40730000, "lon": -73995000, "name": "Greenwich Village", "base": 0.55},
    {"lat": 40750000, "lon": -73975000, "name": "Midtown East", "base": 0.68},
    {"lat": 40783000, "lon": -73971000, "name": "Upper East Side", "base": 0.45},
]


def _to_int_coord(float_coord: float, scale: int = 1000000) -> int:
    """Convert float coordinate to integer format."""
    return int(float_coord * scale)


def _from_int_coord(int_coord: int, scale: int = 1000000) -> float:
    """Convert integer coordinate back to float."""
    return int_coord / scale


def _time_of_day_factor(hour: int) -> float:
    """Return congestion multiplier based on hour (0-23)."""
    if 7 <= hour <= 9:
        return 1.4  # Morning rush
    elif 16 <= hour <= 19:
        return 1.5  # Evening rush
    elif 22 <= hour or hour <= 5:
        return 0.4  # Late night
    elif 11 <= hour <= 14:
        return 1.1  # Lunch traffic
    else:
        return 0.8  # Off-peak


def _normalize_congestion(congestion_0_1: float) -> float:
    """
    Convert congestion from [0, 1] to normalized range with mean ~0.
    This simulates a standardized ML model output.
    """
    # Transform to approximately [-1, 1] range with some variance
    normalized = (congestion_0_1 - 0.5) * 2.0
    # Add slight noise
    normalized += random.gauss(0, 0.15)
    # Clamp to reasonable range
    return max(-1.5, min(1.5, normalized))


def _generate_location_prediction(base_time: datetime) -> LocationPrediction:
    """Generate a single location with 6 horizon predictions."""
    # Pick a location (hotspot or random)
    if random.random() < 0.7:  # 70% near hotspots
        spot = random.choice(HOTSPOTS)
        lat = spot["lat"] + random.randint(-5000, 5000)  # ~500m variance
        lon = spot["lon"] + random.randint(-5000, 5000)
        base_congestion = spot["base"]
    else:  # 30% random locations
        lat = random.randint(*LATITUDE_RANGE)
        lon = random.randint(*LONGITUDE_RANGE)
        base_congestion = random.uniform(0.2, 0.6)

    # Generate predictions for 6 horizons
    horizons = []
    for h in range(1, NUM_HORIZONS + 1):
        future_time = base_time + timedelta(hours=h)
        time_factor = _time_of_day_factor(future_time.hour)
        
        # Congestion evolves over time
        evolution = random.gauss(0, 0.1)  # Random walk
        congestion_raw = base_congestion * time_factor + evolution * h
        congestion_raw = max(0.0, min(1.0, congestion_raw))
        
        # Normalize to ML model output format
        normalized_congestion = _normalize_congestion(congestion_raw)
        
        horizons.append(
            HorizonPrediction(
                horizon=f"t+{h}h",
                DateTime=future_time.isoformat(),
                predicted_congestion=round(normalized_congestion, 4),
            )
        )
    
    return LocationPrediction(
        latitude=lat,
        longitude=lon,
        horizons=horizons,
    )


# ── API Endpoints ────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {
        "service": "Traffic Congestion ML Simulator (Multi-Horizon)",
        "format": "6 horizons per location (t+1h through t+6h)",
        "endpoints": {
            "/predictions/current": "Get predictions for current time + 6 horizons",
            "/predictions/batch": "Get batch predictions for multiple locations",
        }
    }


@app.get("/predictions/current", response_model=OutputList)
def get_current_predictions(locations: int = 20):
    """
    Get multi-horizon predictions for N locations.
    
    Args:
        locations: Number of locations to predict (default 20)
    """
    now = datetime.now()
    predictions = [_generate_location_prediction(now) for _ in range(locations)]
    return OutputList(predictions=predictions)


@app.get("/predictions/batch", response_model=OutputList)
def get_batch_predictions(
    locations: int = 50,
    base_hour: int = None
):
    """
    Get batch predictions for many locations.
    
    Args:
        locations: Number of locations (default 50)
        base_hour: Starting hour (0-23), defaults to current
    """
    if base_hour is None:
        base_time = datetime.now()
    else:
        base_time = datetime.now().replace(hour=base_hour, minute=0, second=0)
    
    predictions = [_generate_location_prediction(base_time) for _ in range(locations)]
    return OutputList(predictions=predictions)


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ── Run Instructions ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Traffic Congestion Simulator API (Multi-Horizon)...")
    print("📍 API docs: http://localhost:8000/docs")
    print("📍 Frontend CORS: enabled for all origins")
    print("\n💡 Run with: uvicorn api:app --reload (for hot reload)")
    print("   Or just:  python api.py\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
