# ML Pipeline Automation

Automated end-to-end pipeline for traffic congestion forecasting.

## 🔄 Pipeline Flow

```
CSV Upload → ML Inference → Process Forecasts → Grid Generation → Frontend Display
```

### Pipeline Steps:
1. **CSV Upload**: Upload traffic data CSV to `backend/uploads/`
2. **ML Inference**: Run TFT model to generate predictions (`ml/pipeline.py`)
3. **Process Forecasts**: Convert predictions to structured format (`backend/outputs/process_forecasts.py`)
4. **Grid Generation**: Create 25x25 grid data (`backend/outputs/forecasts_grid.json`)
5. **Frontend Update**: UI automatically displays new predictions

## 🚀 Usage Methods

### Method 1: Via Web UI (Recommended)
1. Start the backend: `cd backend && uv run python api.py`
2. Start the frontend: `cd frontend && npm run dev`
3. Open http://localhost:3000
4. Click "Select CSV File" in the Analytics panel
5. Upload your CSV file
6. **Pipeline runs automatically** ✨
7. Grid updates in real-time

### Method 2: Command Line (Manual)
```bash
# From project root
python run_pipeline.py                    # Use latest CSV from uploads/
python run_pipeline.py path/to/data.csv   # Use specific CSV
```

Or using bash:
```bash
./run_pipeline.sh                         # Use latest CSV
./run_pipeline.sh path/to/data.csv        # Use specific CSV
```

### Method 3: Individual Steps (Debugging)
```bash
# Step 1: Run inference
cd ml
uv run python pipeline.py --csv ../backend/uploads/your_file.csv

# Step 2: Generate grid
cd ../backend/outputs
uv run python process_forecasts.py

# Step 3: Reload API (if running)
curl -X POST http://localhost:8000/reload
```

## 📊 Input CSV Format

Your CSV must include these columns:
- `time_idx`: Time index
- `edge_id`: Road segment ID
- `road_type`: Type of road (Highway, Arterial, etc.)
- `latitude_midroad`, `longitude_midroad`: Coordinates
- `hour_of_day`, `day_of_week`: Temporal features
- `average_speed_kph`: Average speed
- `vehicle_count`: Number of vehicles
- `travel_time_seconds`: Travel time
- `is_weekend`, `is_holiday`: Binary flags
- `weather_condition`: Weather type
- `visibility`: Visibility value
- `event_impact_score`: Event impact
- Additional features as needed

## 📁 Output Files

After pipeline runs:
- `backend/outputs/forecasts.json`: Raw ML predictions with coordinates
- `backend/outputs/forecasts_grid.json`: 25x25 grid with time frames

## 🔧 API Endpoints

### POST /upload-csv
Upload CSV and trigger full pipeline
```bash
curl -X POST -F "file=@traffic_data.csv" http://localhost:8000/upload-csv
```

### POST /reload
Reload forecast data without re-running pipeline
```bash
curl -X POST http://localhost:8000/reload
```

### GET /grid-frames
Get all grid frames
```bash
curl http://localhost:8000/grid-frames
```

## 🐛 Troubleshooting

### Pipeline fails at Step 1 (ML Inference)
- Check if model checkpoint exists: `ml/checkpoints/best_model.pth`
- Verify CSV has required columns
- Check CSV has enough rows for encoder length (default: 20)

### Pipeline fails at Step 2 (Grid Generation)
- Ensure `backend/outputs/forecasts.json` exists
- Check forecasts.json has valid data structure

### Frontend not updating
- Check browser console for errors
- Verify API is running on port 8000
- Try manual reload: POST http://localhost:8000/reload

### "No CSV found" error
- Ensure CSV is in `backend/uploads/` directory
- Check file has `.csv` extension

## 📝 Configuration

### Model Settings (ml/pipeline.py)
- `encoder_length`: Number of historical time steps (default: 20)
- `prediction_length`: Number of future time steps (default: 20)
- `forecast_interval_minutes`: Minutes between predictions (default: 20)

### Grid Settings (backend/outputs/process_forecasts.py)
- `GRID_SIZE`: Grid dimensions (default: 25x25)

## 🎯 Example Workflow

```bash
# 1. Start services
cd backend && uv run python api.py &
cd frontend && npm run dev &

# 2. Upload CSV via UI or CLI
python run_pipeline.py data/my_traffic_data.csv

# 3. Check results
cat backend/outputs/forecasts_grid.json | jq '.frames | length'

# 4. View in browser
open http://localhost:3000
```

## ⚡ Quick Test

```bash
# Use existing sample CSV (if available)
python run_pipeline.py

# Check output
ls -lh backend/outputs/
```

## 🔄 Auto-Refresh

The frontend polls the API every 30 seconds for updates. When new grid data is available:
- Grid automatically updates
- Frame counter resets
- Analytics panel shows new metrics

## 📚 Related Files

- `ml/pipeline.py`: Main ML inference script
- `backend/api.py`: FastAPI backend with upload endpoint
- `backend/outputs/process_forecasts.py`: Grid generation script
- `frontend/hooks/useGridTraffic.ts`: Frontend data management
- `run_pipeline.py`: Automated pipeline runner (Python)
- `run_pipeline.sh`: Automated pipeline runner (Bash)
