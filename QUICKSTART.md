# 🚀 Quick Start: Upload CSV → See Results

## Fastest Way (Web UI - 3 clicks!)

1. **Start servers:**
   ```bash
   cd backend && uv run python api.py &
   cd frontend && npm run dev
   ```

2. **Open browser:** http://localhost:3000

3. **Upload CSV:**
   - Click "Select CSV File" button (right panel)
   - Choose your traffic data CSV
   - ✨ **Done!** Pipeline runs automatically

4. **Watch results:**
   - Grid updates in real-time
   - "Active Bottlenecks" counter updates
   - Frame slider cycles through predictions

---

## Command Line (1 command!)

```bash
# From project root
python run_pipeline.py path/to/your_data.csv
```

That's it! The pipeline will:
- ✅ Run ML inference
- ✅ Generate grid data
- ✅ Output to backend/outputs/
- ✅ Frontend auto-picks up changes

---

## What You'll See

**Before Upload:**
- Empty grid or sample data
- 0 active bottlenecks

**After Upload (30-60 seconds):**
- Colored heatmap grid (green → yellow → red → purple)
- Active bottlenecks count (red + purple cells)
- Frame counter: "Frame 1/18" (cycles automatically)
- Timeline scrubber at bottom

---

## Troubleshooting

**Grid not updating?**
```bash
# Manually reload API
curl -X POST http://localhost:8000/reload
```

**Pipeline stuck?**
```bash
# Check logs
tail -f backend/outputs/pipeline.log
```

**Need fresh start?**
```bash
# Clear old data
rm backend/uploads/*.csv
rm backend/outputs/forecasts*.json
```

---

## CSV Requirements (Minimum)

Your CSV needs these columns:
- `time_idx`, `edge_id`, `latitude_midroad`, `longitude_midroad`
- `hour_of_day`, `average_speed_kph`, `vehicle_count`
- `road_type`, `is_weekend`, `weather_condition`

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for full format.

---

## Expected Processing Times

- **Small CSV** (<1000 rows): ~10 seconds
- **Medium CSV** (1000-5000 rows): ~30 seconds  
- **Large CSV** (>5000 rows): ~60 seconds

Progress shown in:
- Backend terminal output
- Upload response JSON
- Browser console (F12)

---

## Output Files Location

```
backend/outputs/
├── forecasts.json          # Raw ML predictions
└── forecasts_grid.json     # Grid data (used by frontend)
```

---

## Need Help?

1. Check [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for detailed docs
2. Look at `backend/api.py` upload endpoint for pipeline code
3. Run with `--verbose` flag: `python run_pipeline.py --csv data.csv --verbose`
