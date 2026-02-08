"""
Enrich forecasts.json with average_speed_kph from the CSV in backend/uploads.

Cross-references by matching:
- latitude/longitude (CSV decimal -> forecast integer format via *1e6)
- DateTime -> time_idx (using hour_of_day and position within hour from CSV)

Writes the enriched data back to forecasts.json.
"""

import csv
import glob
import json
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(SCRIPT_DIR, "..", "uploads")
FORECASTS_PATH = os.path.join(SCRIPT_DIR, "forecasts.json")


def find_csv():
    """Find the single CSV file in backend/uploads."""
    pattern = os.path.join(UPLOADS_DIR, "*.csv")
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 CSV in {UPLOADS_DIR}, found {len(matches)}"
        )
    return matches[0]


def build_csv_lookup(csv_path):
    """
    Build a lookup: (time_idx, lat_int, lon_int) -> average_speed_kph
    Also build a mapping: (hour, position_in_hour) -> time_idx
    """
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Build speed lookup keyed by (time_idx, lat_int, lon_int)
    speed_lookup = {}
    for r in rows:
        tidx = int(r["time_idx"])
        lat_int = int(round(float(r["latitude_midroad"]) * 1e6))
        lon_int = int(round(float(r["longitude_midroad"]) * 1e6))
        speed_lookup[(tidx, lat_int, lon_int)] = float(r["average_speed_kph"])

    # Build hour -> sorted list of time_idx values for that hour
    hour_to_tidxs = defaultdict(set)
    for r in rows:
        hour_to_tidxs[int(r["hour_of_day"])].add(int(r["time_idx"]))

    # (hour, position) -> time_idx
    hour_pos_to_tidx = {}
    for hour, tidxs in hour_to_tidxs.items():
        for pos, tidx in enumerate(sorted(tidxs)):
            hour_pos_to_tidx[(hour, pos)] = tidx

    return speed_lookup, hour_pos_to_tidx


def datetime_to_hour_pos(dt_str):
    """Extract (hour, position_in_hour) from a DateTime string."""
    time_part = dt_str.split(" ")[1]
    parts = time_part.split(":")
    hour = int(parts[0])
    minute = int(parts[1])
    # Minutes are 9, 29, 49 -> positions 0, 1, 2
    position = round((minute - 9) / 20)
    return hour, position


def main():
    csv_path = find_csv()
    print(f"Using CSV: {csv_path}")

    speed_lookup, hour_pos_to_tidx = build_csv_lookup(csv_path)
    print(f"Speed lookup entries: {len(speed_lookup)}")
    print(f"Hour-position mappings: {len(hour_pos_to_tidx)}")

    with open(FORECASTS_PATH) as f:
        data = json.load(f)

    entries = data["outputs"]
    matched = 0
    missed = 0

    for entry in entries:
        hour, pos = datetime_to_hour_pos(entry["DateTime"])
        tidx = hour_pos_to_tidx.get((hour, pos))
        if tidx is None:
            entry["average_speed_kph"] = None
            missed += 1
            continue

        key = (tidx, entry["latitude"], entry["longitude"])
        speed = speed_lookup.get(key)
        if speed is not None:
            entry["average_speed_kph"] = round(speed, 2)
            matched += 1
        else:
            entry["average_speed_kph"] = None
            missed += 1

    print(f"\nMatched: {matched}, Missed: {missed}")

    with open(FORECASTS_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {FORECASTS_PATH}")


if __name__ == "__main__":
    main()
