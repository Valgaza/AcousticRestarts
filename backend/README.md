# Backend API Simulator

FastAPI backend that simulates ML model predictions for traffic congestion with multi-horizon forecasts.

## Features

- **Multi-horizon predictions**: Each location predicts 6 hours ahead (t+1h through t+6h)
- **Integer coordinates**: Latitude/longitude as integers (e.g., 40742511 = 40.742511°)
- **Normalized congestion**: Values range approximately -1 to +1 (can be negative)
- **Realistic time patterns**: Rush hour effects, time-of-day variations
- **CORS enabled**: Ready for frontend integration

## Data Format

### New Multi-Horizon Format

Each prediction includes:
- **Location**: Integer lat/lon (e.g., latitude: 40742511, longitude: -73949134)
- **6 Horizons**: Predictions for t+1h, t+2h, ..., t+6h
- **Normalized Congestion**: Float values (approximately -1 to +1)

Example:
```json
{
  "predictions": [
    {
      "latitude": 40742511,
      "longitude": -73949134,
      "horizons": [
        {
          "horizon": "t+1h",
          "DateTime": "2026-02-08T00:22:57.603573",
          "predicted_congestion": -0.9281
        },
        {
          "horizon": "t+2h",
          "DateTime": "2026-02-08T01:22:57.603573",
          "predicted_congestion": -0.6372
        },
        ...
      ]
    }
  ]
}
```

## Installation

Dependencies are managed by the parent project. From the project root:

```bash
uv sync
```

## Running the API

```bash
# From project root
uv run python backend/api.py
```

API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs (Swagger UI)

## API Endpoints

### 1. Current Predictions
```bash
GET /predictions/current?locations=20
```
Returns multi-horizon predictions for N locations.

**Response**:
```json
{
  "predictions": [
    {
      "latitude": 40758123,
      "longitude": -73985456,
      "horizons": [
        {
          "horizon": "t+1h",
          "DateTime": "2026-02-07T15:30:00",
          "predicted_congestion": 0.7241
        },
        ...
      ]
    }
  ]
}
```

### 2. Batch Predictions
```bash
GET /predictions/batch?locations=50&base_hour=8
```
Returns batch predictions for many locations.

### 3. Health Check
```bash
GET /health
```

## Testing

```bash
# From project root
uv run python backend/test_api.py
```

This will:
- Test all endpoints
- Show sample output format
- Display congestion statistics
- Save a `sample_predictions.json` file

## Data Model

```python
class HorizonPrediction(BaseModel):
    horizon: str                # "t+1h", "t+2h", etc.
    DateTime: str               # ISO format timestamp
    predicted_congestion: float # Normalized value (~-1 to +1)

class LocationPrediction(BaseModel):
    latitude: int               # Integer format (40742511)
    longitude: int              # Integer format (-73949134)
    horizons: List[HorizonPrediction]  # 6 horizons

class OutputList(BaseModel):
    predictions: List[LocationPrediction]
```

## Coordinate Format

Coordinates are stored as integers for efficiency:
- **Multiply by 1,000,000** to store: `40.742511° → 40742511`
- **Divide by 1,000,000** to display: `40742511 → 40.742511°`

## Congestion Values

Unlike the old format (0-1), congestion is now normalized:
- **Negative values**: Below-average congestion
- **Zero**: Average congestion
- **Positive values**: Above-average congestion
- **Range**: Approximately -1.5 to +1.5 (most values between -1 and +1)

This mimics standardized ML model outputs (z-scores).

## Simulation Logic

- **Hotspots**: Pre-defined high-traffic areas
- **Time patterns**: Rush hours get 1.4-1.5x multiplier
- **Evolution**: Congestion changes over the 6-hour horizon
- **Noise**: Gaussian randomness for realism

## Frontend Integration

```javascript
// Fetch predictions
fetch('http://localhost:8000/predictions/current?locations=30')
  .then(res => res.json())
  .then(data => {
    data.predictions.forEach(pred => {
      const lat = pred.latitude / 1_000_000
      const lon = pred.longitude / 1_000_000
      
      // Process each horizon
      pred.horizons.forEach(h => {
        console.log(`${h.horizon}: congestion = ${h.predicted_congestion}`)
      })
    })
  })
```
