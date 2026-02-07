# Traffic Congestion Frontend - Live API Integration

## ✨ What Changed

Your frontend is now **dynamic** and can fetch real-time traffic predictions from the backend API!

## 🎯 How It Works

### 1. **Geographic Mapping**
[lib/geoUtils.ts](lib/geoUtils.ts) - Converts lat/lon to grid coordinates:
- Takes latitude/longitude from API
- Maps to 25×25 grid positions
- Handles Manhattan bounds (configurable for any city)
- Auto-calculates bounds from data

### 2. **API Integration**  
[lib/trafficApi.ts](lib/trafficApi.ts) - Fetches predictions:
- `/predictions/current` - Current traffic
- `/predictions/next-hours` - Future predictions
- `/predictions/custom` - Custom time/count

### 3. **Live Traffic Hook**
[hooks/useLiveTraffic.ts](hooks/useLiveTraffic.ts) - Manages state:
- Fetches from API every 30 seconds
- Maps predictions to grid cells
- Averages multiple predictions per cell
- Auto-refresh toggle

### 4. **Dynamic Viewport**
[components/Viewport.tsx](components/Viewport.tsx) - Updated to:
- Switch between simulation vs live API data
- Show loading states
- Display connection status
- Manual refresh button

## 🚀 How to Use

### Start the Backend
```bash
# Terminal 1
cd backend
uv run python api.py
```

### Start the Frontend
```bash
# Terminal 2
cd frontend
npm run dev
```

### Toggle Live Data
1. Open http://localhost:3000
2. Click the **"Live API Data"** toggle in the left sidebar
3. Grid colors update from real API predictions!

## 🎨 Color Mapping

The grid cells color based on `predicted_congestion_level` (0.0-1.0):

| Congestion Level | Color | Meaning |
|---|---|---|
| 0.0 - 0.3 | 🟢 Green | Low / Free flow |
| 0.3 - 0.6 | 🟡 Yellow | Moderate |
| 0.6 - 0.8 | 🔴 Red | Heavy |
| 0.8 - 1.0 | 🟣 Purple | Gridlock |

## ⚙️ Configuration

Edit [lib/geoUtils.ts](lib/geoUtils.ts) to change city bounds:

```typescript
const DEFAULT_BOUNDS: GeoBounds = {
  minLat: 40.700,  // Your city's south edge
  maxLat: 40.800,  // Your city's north edge
  minLon: -74.020, // West edge
  maxLon: -73.940, // East edge
}
```

## 🔄 Auto-Refresh

- **Default**: Refreshes every 30 seconds
- **Pause**: Click "Pause Auto" button
- **Manual**: Click "Refresh" button anytime

## 🐛 Error Handling

If the API is down, you'll see:
- ⚠️ Error banner at the top
- Simulation mode still works
- Can retry with refresh button

## 📊 Data Flow

```
Backend API (port 8000)
    ↓ Predictions with lat/lon
useLiveTraffic Hook
    ↓ Map to grid cells
    ↓ Average multiple points per cell
Viewport Component
    ↓ Color cells by congestion
25×25 Grid Display 🗺️
```

That's it! Your frontend is now fully dynamic. 🎉
