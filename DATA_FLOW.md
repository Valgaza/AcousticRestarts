# Traffic Prediction Data Flow

## Overview
The system loads real ML model predictions from `forecasts.json` and serves them through a FastAPI backend to the Next.js frontend.

## Architecture

```
ML Model → forecasts.json → Backend API → Frontend
```

## Data Flow Steps

### 1. ML Model Output
- Your ML model generates predictions and saves them to `backend/forecasts.json`
- Format: 
```json
{
  "outputs": [
    {
      "DateTime": "2024-01-01 00:00:00",
      "latitude": 19085912,
      "longitude": 72866783,
      "predicted_congestion_level": 0.3452484607696533
    },
    ...
  ]
}
```

### 2. Backend API Loads Data
- On startup, `backend/api.py` automatically loads `forecasts.json`
- Data is structured by location: `{(lat, lon): {datetime: congestion_value}}`
- Currently loaded: **6 unique locations** with **1000 total predictions**

### 3. Backend Serves Predictions
- API transforms the data into 6 time horizons (t+1h through t+6h)
- Normalizes congestion values to [-1, 1] range
- Endpoints:
  - `GET /predictions/current?locations=N` - Get N locations with 6 hourly predictions
  - `GET /predictions/batch?locations=N&base_hour=H` - Get batch predictions
  - `POST /reload` - Reload forecasts.json when model updates it
  - `GET /health` - Health check

### 4. Frontend Fetches & Displays
- Frontend calls API every 30 seconds (configurable)
- Maps lat/lon coordinates to 25x25 grid
- Displays color-coded heatmap with road network overlay
- Shows predictions for selected time horizons

## Updating Predictions

### When Your Model Generates New Predictions:

1. **Model saves to forecasts.json**
   ```bash
   # Your model outputs to backend/forecasts.json
   ```

2. **Reload the backend** (choose one method):
   
   **Option A: API Reload Endpoint** (Recommended)
   ```bash
   curl -X POST http://localhost:8000/reload
   ```
   
   **Option B: Restart Backend**
   ```bash
   # Kill and restart
   lsof -ti:8000 | xargs kill -9
   cd backend && uv run python api.py
   ```

3. **Frontend automatically picks up changes** on next refresh (30 seconds)

## Current Configuration

- **Data Source**: `backend/forecasts.json`
- **Locations**: 6 unique coordinates
- **Total Predictions**: 1000 time-series records
- **API Port**: 8000
- **Frontend Port**: 3000
- **Auto-refresh**: 30 seconds

## Endpoints Reference

### GET /
Returns API info and available locations count

### GET /predictions/current?locations={N}
- **Parameters**: 
  - `locations` (default: 20) - Number of locations to return
- **Returns**: Predictions with 6 time horizons per location
- **Example**:
```bash
curl 'http://localhost:8000/predictions/current?locations=6'
```

### POST /reload
- **Purpose**: Reload forecasts.json without restarting server
- **Use case**: After your ML model updates the JSON file
- **Example**:
```bash
curl -X POST http://localhost:8000/reload
```

### GET /health
- **Purpose**: Check if API is running
- **Returns**: `{"status": "healthy", "timestamp": "..."}`

## Coordinate Format

- **Input**: Integer format (e.g., 19085912 = 19.085912°)
- **Grid**: 25x25 cells mapped to geographic bounds
- **Mapping**: See `frontend/lib/geoUtils.ts` for lat/lon → grid conversion

## Development Workflow

1. **Start Backend**:
   ```bash
   cd backend
   uv run python api.py
   # Loads forecasts.json automatically
   ```

2. **Start Frontend**:
   ```bash
   cd frontend
   npm run dev
   # Connects to backend on localhost:8000
   ```

3. **Update Predictions**:
   - Model updates `forecasts.json`
   - Call reload endpoint or restart backend
   - Frontend auto-refreshes data

## Files Involved

- `backend/forecasts.json` - ML model output (your data source)
- `backend/api.py` - FastAPI server loading and serving data
- `frontend/lib/trafficApi.ts` - API client functions
- `frontend/hooks/useLiveTraffic.ts` - Frontend data fetching hook
- `frontend/lib/geoUtils.ts` - Coordinate to grid mapping
- `frontend/components/Viewport.tsx` - Heatmap visualization

## Notes

✅ Backend loads forecasts.json on startup
✅ Frontend fetches from backend API (not directly from JSON)
✅ Reload endpoint allows updating without restart
✅ All 6 locations are currently being served
✅ Real predictions replace dummy data
