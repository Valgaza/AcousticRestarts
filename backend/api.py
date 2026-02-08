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
import subprocess
import sys

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


class RouteRequest(BaseModel):
    start_node: int
    end_node: int
    timestamp: str  # ISO format: "2026-02-08T03:00:00"


class RouteOptimizationInput(BaseModel):
    requests: List[RouteRequest]


class RouteAssignment(BaseModel):
    request_idx: int
    chosen_route: List[int]  # Edge IDs
    route_nodes: List[int]  # Node IDs
    total_cost: float
    alternatives_considered: int


class RouteOptimizationOutput(BaseModel):
    assignments: List[RouteAssignment]
    output_csv_path: str


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
    Automatically triggers the full pipeline:
    1. Save CSV to backend/uploads/
    2. Run ML inference (pipeline.py)
    3. Process forecasts (process_forecasts.py)
    4. Generate grid data (forecasts_grid.json)
    5. Reload data in API
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed"
        )
    
    backend_dir = Path(__file__).parent
    workspace_root = backend_dir.parent
    uploads_dir = backend_dir / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    
    # Clear old CSV files to avoid conflicts
    for old_csv in uploads_dir.glob("*.csv"):
        old_csv.unlink()
    
    # Save file without timestamp (scripts expect single CSV)
    file_path = uploads_dir / file.filename
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        pipeline_steps = []
        
        # Step 1: Run ML inference pipeline
        print("\n🚀 Step 1: Running ML inference pipeline...")
        pipeline_py = workspace_root / "ml" / "pipeline.py"
        result = subprocess.run(
            [sys.executable, str(pipeline_py), "--csv", str(file_path)],
            capture_output=True,
            text=True,
            cwd=str(workspace_root / "ml")
        )
        if result.returncode != 0:
            raise Exception(f"Pipeline failed: {result.stderr}")
        pipeline_steps.append({"step": "ML Inference", "status": "success", "output": result.stdout.strip()})
        
        # Step 2: Process forecasts into grid format
        print("\n🔄 Step 2: Processing forecasts into grid...")
        process_py = backend_dir / "outputs" / "process_forecasts.py"
        result = subprocess.run(
            [sys.executable, str(process_py)],
            capture_output=True,
            text=True,
            cwd=str(backend_dir / "outputs")
        )
        if result.returncode != 0:
            raise Exception(f"Process forecasts failed: {result.stderr}")
        pipeline_steps.append({"step": "Grid Generation", "status": "success", "output": result.stdout.strip()})
        
        # Step 3: Reload data in API
        print("\n♻️  Step 3: Reloading forecast data in API...")
        load_forecasts()
        load_grid_data()
        pipeline_steps.append({"step": "API Reload", "status": "success", "locations": len(FORECAST_DATA), "frames": len(GRID_DATA.get('frames', []))})
        
        print("\n✅ Pipeline completed successfully!")
        
        return {
            "status": "success",
            "message": "CSV processed and forecasts generated",
            "filename": file.filename,
            "path": str(file_path),
            "size_bytes": file_path.stat().st_size,
            "pipeline_steps": pipeline_steps,
            "forecast_locations": len(FORECAST_DATA),
            "grid_frames": len(GRID_DATA.get('frames', [])),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"\n❌ Pipeline error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
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


@app.post("/optimize-routes", response_model=RouteOptimizationOutput)
async def optimize_routes(input_data: RouteOptimizationInput):
    """
    Optimize multiple route requests using the route_optimizer.py script.
    
    This endpoint:
    1. Processes each route request through the optimizer
    2. Returns the chosen routes with cost information
    3. Generates a modified CSV with updated traffic predictions
    """
    try:
        # Get workspace root
        ml_dir = Path(__file__).parent.parent / "ml"
        route_optimizer_script = ml_dir / "route_optimizer.py"
        
        if not route_optimizer_script.exists():
            raise HTTPException(status_code=500, detail="route_optimizer.py not found in ml/ directory")
        
        assignments = []
        output_csv_path = None
        
        # Process each route request
        for idx, request in enumerate(input_data.requests):
            print(f"\n🔄 Processing route {idx + 1}/{len(input_data.requests)}: "
                  f"Node {request.start_node} → {request.end_node} @ {request.timestamp}")
            
            # Run route optimizer
            cmd = [
                sys.executable,
                str(route_optimizer_script),
                "--start", str(request.start_node),
                "--end", str(request.end_node),
                "--timestamp", request.timestamp
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(ml_dir),
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per route
            )
            
            if result.returncode != 0:
                print(f"❌ Route optimizer failed: {result.stderr}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Route optimization failed for request {idx + 1}: {result.stderr}"
                )
            
            # Parse JSON output from stdout
            try:
                # Find the JSON output (last line usually contains the result)
                output_lines = result.stdout.strip().split('\n')
                json_output = None
                for line in reversed(output_lines):
                    if line.strip().startswith('{'):
                        json_output = json.loads(line)
                        break
                
                if not json_output:
                    raise ValueError("No JSON output found from route optimizer")
                
                # Store the output CSV path (same for all requests)
                if not output_csv_path:
                    output_csv_path = json_output.get("output_csv", "backend/outputs/optimized_edges.csv")
                
                # Create assignment
                assignments.append(RouteAssignment(
                    request_idx=idx,
                    chosen_route=json_output.get("chosen_edges", []),
                    route_nodes=json_output.get("chosen_route", []),
                    total_cost=json_output.get("total_cost", 0.0),
                    alternatives_considered=len(json_output.get("all_routes", []))
                ))
                
                print(f"✅ Route {idx + 1} optimized: {len(json_output.get('chosen_route', []))} nodes, "
                      f"cost={json_output.get('total_cost', 0):.2f}")
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"❌ Failed to parse route optimizer output: {e}")
                print(f"   stdout: {result.stdout}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse route optimizer output for request {idx + 1}"
                )
        
        print(f"\n🎉 All {len(assignments)} routes optimized successfully")
        
        return RouteOptimizationOutput(
            assignments=assignments,
            output_csv_path=output_csv_path or "backend/outputs/optimized_edges.csv"
        )
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Route optimization timed out")
    except Exception as e:
        print(f"❌ Error in route optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _merge_gt_forecasts_into_forecasts():
    """
    Merge gt_forecasts.json into forecasts.json.
    
    gt_forecasts.json has decimal lat/lon (e.g. 19.085912).
    forecasts.json has integer lat/lon (×10^6, e.g. 19085912).
    
    For each entry in gt_forecasts:
    - Convert lat/lon to integer (×10^6)
    - Find matching entries in forecasts.json by (lat_int, lon_int)
    - Update the congestion values for matching timestamps
    - If no match found, add as new entry
    """
    backend_dir = Path(__file__).parent
    gt_path = backend_dir / "outputs" / "gt_forecasts.json"
    forecasts_path = backend_dir / "outputs" / "forecasts.json"
    
    if not gt_path.exists():
        raise FileNotFoundError("gt_forecasts.json not found")
    
    # Load gt_forecasts
    with open(gt_path) as f:
        gt_data = json.load(f)
    
    # Load or initialize forecasts.json
    if forecasts_path.exists():
        with open(forecasts_path) as f:
            forecasts_data = json.load(f)
    else:
        forecasts_data = {"outputs": []}
    
    # Build lookup: (lat_int, lon_int, datetime) → index in forecasts outputs
    existing_lookup = {}
    for idx, entry in enumerate(forecasts_data["outputs"]):
        key = (entry["latitude"], entry["longitude"], entry["DateTime"])
        existing_lookup[key] = idx
    
    updated_count = 0
    added_count = 0
    
    for gt_entry in gt_data["outputs"]:
        # Convert decimal lat/lon to integer (×10^6)
        lat_int = int(round(gt_entry["latitude"] * 1e6))
        lon_int = int(round(gt_entry["longitude"] * 1e6))
        dt = gt_entry["DateTime"]
        congestion = gt_entry["predicted_congestion_level"]
        
        key = (lat_int, lon_int, dt)
        
        if key in existing_lookup:
            # Update existing entry
            idx = existing_lookup[key]
            forecasts_data["outputs"][idx]["predicted_congestion_level"] = congestion
            updated_count += 1
        else:
            # Add new entry with integer lat/lon
            new_entry = {
                "DateTime": dt,
                "latitude": lat_int,
                "longitude": lon_int,
                "predicted_congestion_level": congestion,
            }
            forecasts_data["outputs"].append(new_entry)
            added_count += 1
    
    # Save updated forecasts.json
    with open(forecasts_path, "w") as f:
        json.dump(forecasts_data, f, indent=2)
    
    print(f"✅ Merged gt_forecasts: {updated_count} updated, {added_count} added")
    return updated_count, added_count


@app.post("/optimize-and-forecast")
async def optimize_and_forecast(input_data: RouteOptimizationInput):
    """
    Full pipeline: Optimize routes → Run GT forecasting → Update grid.
    
    Steps:
    1. Run route_optimizer.py for each request → optimized_edges.csv
    2. Run gt_forecasting.py on the optimized edges → gt_forecasts.json
    3. Merge gt_forecasts.json into forecasts.json (lat/lon ×10^6)
    4. Run process_forecasts.py → forecasts_grid.json
    5. Reload API data so frontend gets updated grid
    """
    backend_dir = Path(__file__).parent
    workspace_root = backend_dir.parent
    ml_dir = workspace_root / "ml"
    
    pipeline_steps = []
    assignments = []
    
    try:
        # ── Step 1: Route Optimization ───────────────────────────────────
        print("\n🚀 Step 1: Running route optimizer...")
        route_optimizer_script = ml_dir / "route_optimizer.py"
        
        if not route_optimizer_script.exists():
            raise HTTPException(status_code=500, detail="route_optimizer.py not found")
        
        output_csv_path = None
        
        for idx, request in enumerate(input_data.requests):
            print(f"  Route {idx + 1}/{len(input_data.requests)}: "
                  f"Node {request.start_node} → {request.end_node} @ {request.timestamp}")
            
            cmd = [
                sys.executable,
                str(route_optimizer_script),
                "--start", str(request.start_node),
                "--end", str(request.end_node),
                "--timestamp", request.timestamp
            ]
            
            result = subprocess.run(
                cmd, cwd=str(ml_dir),
                capture_output=True, text=True, timeout=120
            )
            
            if result.returncode != 0:
                raise Exception(f"Route optimizer failed for request {idx + 1}: {result.stderr}")
            
            # Parse JSON output
            output_lines = result.stdout.strip().split('\n')
            json_output = None
            for line in reversed(output_lines):
                if line.strip().startswith('{'):
                    json_output = json.loads(line)
                    break
            
            if not json_output:
                raise ValueError(f"No JSON output from route optimizer for request {idx + 1}")
            
            if not output_csv_path:
                output_csv_path = json_output.get("output_csv", str(backend_dir / "outputs" / "optimized_edges.csv"))
            
            assignments.append({
                "request_idx": idx,
                "chosen_route": json_output.get("chosen_edges", []),
                "route_nodes": json_output.get("chosen_route", []),
                "total_cost": json_output.get("total_cost", 0.0),
                "alternatives_considered": len(json_output.get("all_routes", []))
            })
            
            print(f"  ✅ Route {idx + 1} optimized: cost={json_output.get('total_cost', 0):.2f}")
        
        pipeline_steps.append({
            "step": "Route Optimization",
            "status": "success",
            "routes_optimized": len(assignments),
            "output_csv": output_csv_path
        })
        
        # ── Step 2: GT Forecasting ───────────────────────────────────────
        print("\n🔮 Step 2: Running GT forecasting on optimized edges...")
        gt_forecasting_script = ml_dir / "gt_forecasting.py"
        
        if not gt_forecasting_script.exists():
            raise Exception("gt_forecasting.py not found in ml/ directory")
        
        result = subprocess.run(
            [sys.executable, str(gt_forecasting_script)],
            cwd=str(ml_dir),
            capture_output=True, text=True, timeout=300  # 5 min timeout
        )
        
        if result.returncode != 0:
            raise Exception(f"GT forecasting failed: {result.stderr}")
        
        print(f"  ✅ GT forecasting completed")
        pipeline_steps.append({
            "step": "GT Forecasting",
            "status": "success",
            "output": result.stdout.strip()[-200:]  # Last 200 chars
        })
        
        # ── Step 3: Merge gt_forecasts into forecasts.json ──────────────
        print("\n📊 Step 3: Merging GT forecasts into forecasts.json...")
        updated, added = _merge_gt_forecasts_into_forecasts()
        
        pipeline_steps.append({
            "step": "Merge Forecasts",
            "status": "success",
            "entries_updated": updated,
            "entries_added": added
        })
        
        # ── Step 4: Run process_forecasts.py ────────────────────────────
        print("\n🔄 Step 4: Processing forecasts into grid...")
        process_py = backend_dir / "outputs" / "process_forecasts.py"
        
        result = subprocess.run(
            [sys.executable, str(process_py)],
            cwd=str(workspace_root),
            capture_output=True, text=True, timeout=60
        )
        
        if result.returncode != 0:
            raise Exception(f"Process forecasts failed: {result.stderr}")
        
        print(f"  ✅ Grid data regenerated")
        pipeline_steps.append({
            "step": "Grid Generation",
            "status": "success",
            "output": result.stdout.strip()
        })
        
        # ── Step 5: Reload API data ─────────────────────────────────────
        print("\n♻️  Step 5: Reloading forecast data...")
        load_forecasts()
        load_grid_data()
        
        pipeline_steps.append({
            "step": "API Reload",
            "status": "success",
            "locations": len(FORECAST_DATA),
            "grid_frames": len(GRID_DATA.get('frames', []))
        })
        
        print("\n✅ Full pipeline completed successfully!")
        
        return {
            "status": "success",
            "message": "Route optimization and forecasting completed",
            "pipeline_steps": pipeline_steps,
            "assignments": assignments,
            "forecast_locations": len(FORECAST_DATA),
            "grid_frames": len(GRID_DATA.get('frames', [])),
            "timestamp": datetime.now().isoformat()
        }
    
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Pipeline timed out")
    except Exception as e:
        print(f"\n❌ Pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


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
