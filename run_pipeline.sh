#!/bin/bash
# Automated ML Pipeline Runner
# Usage: ./run_pipeline.sh <path_to_csv>
# Or: ./run_pipeline.sh (uses latest CSV from backend/uploads)

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting ML Pipeline...${NC}\n"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Run ML Inference
echo -e "${BLUE}📊 Step 1: Running ML Inference Pipeline${NC}"
if [ -z "$1" ]; then
    echo "No CSV specified, using latest from backend/uploads/"
    cd ml && uv run python pipeline.py && cd ..
else
    echo "Using CSV: $1"
    cd ml && uv run python pipeline.py --csv "$1" && cd ..
fi

echo -e "${GREEN}✅ ML Inference Complete${NC}\n"

# Step 2: Process Forecasts into Grid
echo -e "${BLUE}🗺️  Step 2: Generating Grid Data${NC}"
cd backend/outputs && uv run python process_forecasts.py && cd ../..
echo -e "${GREEN}✅ Grid Data Generated${NC}\n"

# Step 3: Show Summary
echo -e "${BLUE}📈 Pipeline Summary${NC}"
FORECAST_COUNT=$(python3 -c "import json; data=json.load(open('backend/outputs/forecasts.json')); print(len(data['outputs']))")
GRID_FRAMES=$(python3 -c "import json; data=json.load(open('backend/outputs/forecasts_grid.json')); print(len(data['frames']))")
echo "  • Forecast records: $FORECAST_COUNT"
echo "  • Grid frames: $GRID_FRAMES"
echo "  • Output files:"
echo "    - backend/outputs/forecasts.json"
echo "    - backend/outputs/forecasts_grid.json"

echo -e "\n${GREEN}✅ Pipeline Complete! Frontend will auto-update.${NC}"
