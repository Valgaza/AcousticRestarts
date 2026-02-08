"""
Process forecasts.json into a 25x25 grid, restructured by timestamp.

Reads the raw forecasts (grouped by coordinate), and outputs a new JSON file where:
- Data is grouped by timestamp (time-first)
- Each coordinate is mapped to a grid cell (i, j) based on bounding rectangle
- Multiple coordinates in the same cell are averaged
- Empty cells default to congestion level 0
"""

import json
import os
from collections import defaultdict

GRID_SIZE = 25
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "forecasts.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "forecasts_grid.json")


def load_forecasts(path):
    with open(path) as f:
        return json.load(f)["outputs"]


def compute_bounds(entries):
    lats = [e["latitude"] for e in entries]
    lons = [e["longitude"] for e in entries]
    return {
        "lat_min": min(lats),
        "lat_max": max(lats),
        "lon_min": min(lons),
        "lon_max": max(lons),
    }


def coord_to_cell(lat, lon, bounds):
    """Map a lat/lon to a grid cell (row, col) in the 25x25 grid."""
    lat_range = bounds["lat_max"] - bounds["lat_min"]
    lon_range = bounds["lon_max"] - bounds["lon_min"]

    # Avoid division by zero if all points share a coordinate
    if lat_range == 0:
        lat_range = 1
    if lon_range == 0:
        lon_range = 1

    # Normalize to [0, 1]
    lat_norm = (lat - bounds["lat_min"]) / lat_range
    lon_norm = (lon - bounds["lon_min"]) / lon_range

    # Map to grid index [0, 24]. Clamp to handle exact max values.
    row = min(int(lat_norm * GRID_SIZE), GRID_SIZE - 1)
    col = min(int(lon_norm * GRID_SIZE), GRID_SIZE - 1)

    # Invert row so that higher latitudes (north) are at the top (row 0)
    row = (GRID_SIZE - 1) - row

    return row, col


def process(entries, bounds):
    """
    Group entries by timestamp, map coords to grid cells,
    average congestion for cells with multiple points,
    and fill empty cells with 0.
    """
    # Intermediate: timestamp -> (row, col) -> list of congestion values
    ts_cell_values = defaultdict(lambda: defaultdict(list))
    # Intermediate: timestamp -> (row, col) -> list of speed values
    ts_cell_speeds = defaultdict(lambda: defaultdict(list))

    for entry in entries:
        ts = entry["DateTime"]
        row, col = coord_to_cell(entry["latitude"], entry["longitude"], bounds)
        ts_cell_values[ts][(row, col)].append(entry["predicted_congestion_level"])
        speed = entry.get("average_speed_kph")
        if speed is not None:
            ts_cell_speeds[ts][(row, col)].append(speed)

    # Build final structure
    timestamps_sorted = sorted(ts_cell_values.keys())
    result = []

    for ts in timestamps_sorted:
        cells = []
        cell_map = ts_cell_values[ts]
        speed_map = ts_cell_speeds[ts]
        all_speeds = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                values = cell_map.get((i, j))
                if values:
                    congestion = sum(values) / len(values)
                else:
                    congestion = 0.0
                cells.append({
                    "row": i,
                    "col": j,
                    "predicted_congestion_level": round(congestion, 6),
                })
                speeds = speed_map.get((i, j))
                if speeds:
                    all_speeds.extend(speeds)

        avg_speed = round(sum(all_speeds) / len(all_speeds), 2) if all_speeds else 0.0
        result.append({
            "DateTime": ts,
            "AvgSpeed": avg_speed,
            "cells": cells,
        })

    return result


def main():
    entries = load_forecasts(INPUT_PATH)
    bounds = compute_bounds(entries)

    print(f"Loaded {len(entries)} entries")
    print(f"Bounds: lat [{bounds['lat_min']} - {bounds['lat_max']}], "
          f"lon [{bounds['lon_min']} - {bounds['lon_max']}]")
    print(f"Lat span: {(bounds['lat_max'] - bounds['lat_min']) / 1e6:.6f}°")
    print(f"Lon span: {(bounds['lon_max'] - bounds['lon_min']) / 1e6:.6f}°")

    result = process(entries, bounds)

    unique_ts = len(result)
    cells_per_ts = len(result[0]["cells"]) if result else 0
    populated = sum(
        1 for c in result[0]["cells"] if c["predicted_congestion_level"] > 0
    ) if result else 0

    print(f"\nOutput: {unique_ts} timestamps x {cells_per_ts} cells each")
    print(f"Populated cells per timestamp: ~{populated} / {GRID_SIZE * GRID_SIZE}")

    output = {
        "metadata": {
            "grid_size": GRID_SIZE,
            "bounds": {
                "lat_min": bounds["lat_min"] / 1e6,
                "lat_max": bounds["lat_max"] / 1e6,
                "lon_min": bounds["lon_min"] / 1e6,
                "lon_max": bounds["lon_max"] / 1e6,
            },
            "total_timestamps": unique_ts,
            "cells_per_timestamp": cells_per_ts,
        },
        "frames": result,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
