"""
FastAPI Backend for Traffic Congestion ML Model
==========================================================
Serves real ML model predictions from forecasts.json with multi-horizon forecasts.

Returns predictions with:
- Location (latitude/longitude as integers)
- 6 time horizons (t+1h through t+6h)
- Normalized congestion values
"""

from datetime import datetime, timedelta
from typing import List
import json
from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ── Models ───────────────────────────────────────────────────────────────────


class HorizonPrediction(BaseModel):
    horizon: str  # e.g., "t+1h", "t+2h", etc.
    DateTime: str
    predicted_congestion: float  # Normalized value, can be negative


class LocationPrediction(BaseModel):
    latitude: float  # Can be decimal (19.085912) or integer (19085912)
    longitude: float  # Can be decimal (72.866783) or integer (72866783)
    horizons: List[HorizonPrediction]


class OutputList(BaseModel):
    predictions: List[LocationPrediction]


# ── App Setup ────────────────────────────────────────────────────────────────


app = FastAPI(title="Traffic Congestion ML API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load real forecast data
FORECASTS_FILE = Path(__file__).parent / "outputs" / "forecasts.json"
FORECASTS_GRID_FILE = Path(__file__).parent / "outputs" / "forecasts_grid.json"
FORECAST_DATA = {}
GRID_DATA = {}

def load_forecasts():
    """Load and structure forecast data from JSON file."""
    global FORECAST_DATA
    print("📊 Loading forecasts from forecasts.json...")
    
    with open(FORECASTS_FILE, 'r') as f:
        data = json.load(f)
    
    # Group by location and time
    # Structure: {(lat, lon): {datetime_str: congestion_value}}
    for record in data['outputs']:
        lat = record['latitude']
        lon = record['longitude']
        dt = record['DateTime']
        congestion = record['predicted_congestion_level']
        
        # Handle both decimal and integer formats
        if isinstance(lat, int) or lat > 180 or lat < -180:
            lat = lat / 1_000_000 if abs(lat) > 180 else lat
        if isinstance(lon, int) or lon > 180 or lon < -180:
            lon = lon / 1_000_000 if abs(lon) > 180 else lon
        
        key = (lat, lon)
        if key not in FORECAST_DATA:
            FORECAST_DATA[key] = {}
        FORECAST_DATA[key][dt] = congestion
    
    print(f"✅ Loaded {len(FORECAST_DATA)} unique locations with {len(data['outputs'])} total predictions")

def load_grid_data():
    """Load grid-based forecast data from forecasts_grid.json."""
    global GRID_DATA
    print("📊 Loading grid data from forecasts_grid.json...")
    
    try:
        with open(FORECASTS_GRID_FILE, 'r') as f:
            GRID_DATA = json.load(f)
        print(f"✅ Loaded {len(GRID_DATA.get('frames', []))} time frames with {GRID_DATA.get('metadata', {}).get('grid_size', 0)}x{GRID_DATA.get('metadata', {}).get('grid_size', 0)} grid")
    except FileNotFoundError:
        print("⚠️ forecasts_grid.json not found, grid data will not be available")
        GRID_DATA = {"metadata": {"grid_size": 25}, "frames": []}

# Load data on startup
load_forecasts()
load_grid_data()


# ── Data Processing ──────────────────────────────────────────────────────────


NUM_HORIZONS = 6  # t+1h through t+6h


def _normalize_congestion(congestion_0_1: float) -> float:
    """
    Convert congestion from [0, 1] to normalized range with mean ~0.
    This matches the ML model output format expected by frontend.
    """
    # Transform to approximately [-1, 1] range
    normalized = (congestion_0_1 - 0.5) * 2.0
    # Clamp to reasonable range
    return max(-1.5, min(1.5, normalized))


def _get_future_prediction(location_data: dict, base_time: datetime, hours_ahead: int) -> float:
    """Get prediction for a specific time horizon from location data."""
    target_time = base_time + timedelta(hours=hours_ahead)
    target_str = target_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Round to nearest 20 minutes (since data is in 20-minute intervals)
    minutes = (target_time.minute // 20) * 20
    rounded_time = target_time.replace(minute=minutes, second=0)
    rounded_str = rounded_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Get prediction from data, or use nearest available
    if rounded_str in location_data:
        return location_data[rounded_str]
    
    # Fallback: find closest timestamp
    available_times = sorted(location_data.keys())
    if available_times:
        # Use first available prediction as fallback
        return location_data[available_times[0]]
    
    return 0.5  # Default fallback


def _generate_location_prediction(base_time: datetime, location_key: tuple) -> LocationPrediction:
    """Generate predictions for a specific location using real forecast data."""
    lat, lon = location_key
    location_data = FORECAST_DATA.get(location_key, {})
    
    # Generate predictions for 6 horizons
    horizons = []
    for h in range(1, NUM_HORIZONS + 1):
        future_time = base_time + timedelta(hours=h)
        
        # Get real prediction from loaded data
        congestion_raw = _get_future_prediction(location_data, base_time, h)
        
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
        "service": "Traffic Congestion ML API (Real Data)",
        "format": "6 horizons per location (t+1h through t+6h)",
        "data_source": "forecasts.json",
        "available_locations": len(FORECAST_DATA),
        "endpoints": {
            "/predictions/current": "Get predictions for current time + 6 horizons",
            "/predictions/batch": "Get batch predictions for multiple locations",
        }
    }


@app.get("/predictions/current", response_model=OutputList)
def get_current_predictions(locations: int = 20):
    """
    Get multi-horizon predictions for N locations using real forecast data.
    
    Args:
        locations: Number of locations to return (default 20, max = available locations)
    """
    now = datetime.now()
    
    # Get available locations (up to requested amount)
    available_locations = list(FORECAST_DATA.keys())
    num_to_return = min(locations, len(available_locations))
    selected_locations = available_locations[:num_to_return]
    
    predictions = [_generate_location_prediction(now, loc) for loc in selected_locations]
    return OutputList(predictions=predictions)


@app.get("/predictions/batch", response_model=OutputList)
def get_batch_predictions(
    locations: int = 50,
    base_hour: int = None
):
    """
    Get batch predictions for many locations using real forecast data.
    
    Args:
        locations: Number of locations (default 50, max = available locations)
        base_hour: Starting hour (0-23), defaults to current
    """
    if base_hour is None:
        base_time = datetime.now()
    else:
        base_time = datetime.now().replace(hour=base_hour, minute=0, second=0)
    
    # Get available locations (up to requested amount)
    available_locations = list(FORECAST_DATA.keys())
    num_to_return = min(locations, len(available_locations))
    selected_locations = available_locations[:num_to_return]
    
    predictions = [_generate_location_prediction(base_time, loc) for loc in selected_locations]
    return OutputList(predictions=predictions)


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/grid-frames")
def get_grid_frames():
    """
    Get all grid frames with congestion data.
    Returns frames with pre-computed grid positions and congestion values.
    """
    if not GRID_DATA or not GRID_DATA.get('frames'):
        return {
            "error": "Grid data not available",
            "metadata": {"grid_size": 25},
            "frames": []
        }
    
    return {
        "metadata": GRID_DATA.get('metadata', {}),
        "frames": GRID_DATA.get('frames', []),
        "total_frames": len(GRID_DATA.get('frames', []))
    }


@app.get("/grid-frame/{frame_index}")
def get_grid_frame(frame_index: int):
    """
    Get a specific grid frame by index.
    """
    frames = GRID_DATA.get('frames', [])
    
    if frame_index < 0 or frame_index >= len(frames):
        raise HTTPException(
            status_code=404,
            detail=f"Frame index {frame_index} not found. Available frames: 0-{len(frames)-1}"
        )
    
    return {
        "metadata": GRID_DATA.get('metadata', {}),
        "frame": frames[frame_index],
        "frame_index": frame_index,
        "total_frames": len(frames)
    }


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file for ML model processing.
    File will be saved to backend/uploads/ directory.
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed"
        )
    
    # Create uploads directory if it doesn't exist
    uploads_dir = Path(__file__).parent / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    
    # Save file with timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = uploads_dir / safe_filename
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "status": "success",
            "message": "CSV file uploaded successfully",
            "filename": safe_filename,
            "path": str(file_path),
            "size_bytes": file_path.stat().st_size,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving file: {str(e)}"
        )
    finally:
        file.file.close()


@app.post("/reload")
def reload_forecasts():
    """
    Reload forecast data from forecasts.json and forecasts_grid.json.
    Use this endpoint when your ML model updates the JSON files.
    """
    try:
        load_forecasts()
        load_grid_data()
        return {
            "status": "success",
            "message": "Forecasts and grid data reloaded",
            "locations": len(FORECAST_DATA),
            "grid_frames": len(GRID_DATA.get('frames', [])),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


# ── Run Instructions ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Traffic Congestion ML API (Real Data)...")
    print("📍 API docs: http://localhost:8000/docs")
    print("📍 Frontend CORS: enabled for all origins")
    print(f"📊 Loaded {len(FORECAST_DATA)} locations from forecasts.json")
    print("\n💡 Run with: uvicorn api:app --reload (for hot reload)")
    print("   Or just:  python api.py\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
