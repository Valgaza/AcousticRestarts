"""
Synthetic Traffic Dataset Generator for Mumbai (Small Road Network)
===================================================================
Generates realistic traffic data over a hand-crafted 25-edge road network
centered on Mumbai, spanning 120 days at 20-minute intervals. The network
has 16 intersection nodes connected by highways, arterials, residential
streets, and ramps — forming a coherent, connected graph.

Output: mumbai_traffic_synthetic.csv
Target: ~216,000 rows (25 edges × 8,640 timestamps)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import time as _time

# ============================================================================
# 0. SEED FOR REPRODUCIBILITY
# ============================================================================
np.random.seed(42)

# ============================================================================
# 1. ROAD NETWORK — 16 nodes, 25 edges, centered on Mumbai
# ============================================================================
# Node layout (schematic — not to scale):
#
#       N0 -------- N1 -------- N2
#       |           |           |
#       N3 ---N4----N5----N6--- N7
#       |     |     |     |     |
#       N8    N9    N10   N11   N12
#       |           |           |
#       N13 -------N14---------N15
#
# N5/N10 are the CBD center. The highway runs N1→N5→N10→N14 (north-south).
# Arterials run east-west through N3–N7 and N13–N15.
# Residential streets branch off arterials.
# Ramps connect highway to residential/arterial junctions.

CENTER_LAT = 19.0760
CENTER_LNG = 72.8777

# Approximate degree offsets at Mumbai's latitude (~19°N)
METERS_PER_DEG_LAT = 111_320.0
METERS_PER_DEG_LNG = 111_320.0 * np.cos(np.radians(CENTER_LAT))

# Node definitions: (node_id, lat_offset_meters, lng_offset_meters)
# Offsets are from the city center (N5/N10 area)
NODE_POSITIONS = {
    0:  (-1200,  -800),   # NW suburb
    1:  (-1200,     0),   # N highway junction
    2:  (-1200,   800),   # NE suburb
    3:  ( -500,  -900),   # W arterial junction
    4:  ( -500,  -450),   # NW residential node
    5:  ( -500,     0),   # CBD north (highway)
    6:  ( -500,   450),   # NE residential node
    7:  ( -500,   900),   # E arterial junction
    8:  ( -100, -1100),   # W suburb branch
    9:  ( -100,  -450),   # W residential branch
    10: (    0,     0),   # CBD CENTER
    11: ( -100,   450),   # E residential branch
    12: ( -100,  1100),   # E suburb branch
    13: (  600,  -800),   # SW junction
    14: (  600,     0),   # S highway junction (CBD south)
    15: (  600,   800),   # SE junction
}

NUM_NODES = len(NODE_POSITIONS)

node_lats = np.array([
    CENTER_LAT + off[0] / METERS_PER_DEG_LAT for off in NODE_POSITIONS.values()
])
node_lngs = np.array([
    CENTER_LNG + off[1] / METERS_PER_DEG_LNG for off in NODE_POSITIONS.values()
])

# Edge definitions: (edge_id, start_node, end_node, road_type)
# Each edge connects two intersections and has a designated road type.
# The graph is connected and the node references are semantically meaningful.
EDGE_DEFS = [
    # --- Highway corridor (N-S spine): N1 → N5 → N10 → N14 ---
    ( 0,  1,  5, "Highway"),
    ( 1,  5, 10, "Highway"),
    ( 2, 10, 14, "Highway"),
    # --- Northern arterial (E-W): N0 → N1 → N2 ---
    ( 3,  0,  1, "Arterial"),
    ( 4,  1,  2, "Arterial"),
    # --- Central arterial (E-W): N3 → N4 → N5 → N6 → N7 ---
    ( 5,  3,  4, "Arterial"),
    ( 6,  4,  5, "Arterial"),
    ( 7,  5,  6, "Arterial"),
    ( 8,  6,  7, "Arterial"),
    # --- Southern arterial (E-W): N13 → N14 → N15 ---
    ( 9, 13, 14, "Arterial"),
    (10, 14, 15, "Arterial"),
    # --- N-S residential connectors ---
    (11,  0,  3, "Residential"),   # NW vertical
    (12,  4,  9, "Residential"),   # inner west vertical
    (13,  6, 11, "Residential"),   # inner east vertical
    (14,  2,  7, "Residential"),   # NE vertical
    (15,  3,  8, "Residential"),   # W suburb branch
    (16,  7, 12, "Residential"),   # E suburb branch
    (17,  8, 13, "Residential"),   # SW vertical
    (18, 12, 15, "Residential"),   # SE vertical
    # --- Residential in CBD area ---
    (19,  9, 10, "Residential"),   # W approach to center
    (20, 10, 11, "Residential"),   # E approach from center
    # --- Ramps (highway on/off connections) ---
    (21,  1,  4, "Ramp"),          # N highway ↔ NW residential
    (22,  1,  6, "Ramp"),          # N highway ↔ NE residential
    (23,  5,  9, "Ramp"),          # CBD highway ↔ W residential
    (24, 14, 13, "Ramp"),          # S highway ↔ SW junction
]

NUM_EDGES = len(EDGE_DEFS)  # 25

# Extract arrays from edge definitions
edge_ids = np.array([e[0] for e in EDGE_DEFS])
start_nodes = np.array([e[1] for e in EDGE_DEFS])
end_nodes = np.array([e[2] for e in EDGE_DEFS])
road_type_arr = np.array([e[3] for e in EDGE_DEFS])

# Compute edge midpoint lat/lng (for spatial features) and road length
edge_lats = (node_lats[start_nodes] + node_lats[end_nodes]) / 2.0
edge_lngs = (node_lngs[start_nodes] + node_lngs[end_nodes]) / 2.0

# Road length = Euclidean distance between start and end nodes (in meters)
dlat_m = (node_lats[end_nodes] - node_lats[start_nodes]) * METERS_PER_DEG_LAT
dlng_m = (node_lngs[end_nodes] - node_lngs[start_nodes]) * METERS_PER_DEG_LNG
road_lengths = np.sqrt(dlat_m**2 + dlng_m**2)

# Properties per road type
ROAD_PROPS = {
    #                  lane_count, speed_limit, free_flow, capacity
    "Highway":      (6, 80, 90, 4000),
    "Arterial":     (4, 60, 70, 2500),
    "Residential":  (2, 30, 35, 800),
    "Ramp":         (2, 40, 50, 1200),
}

lane_counts = np.array([ROAD_PROPS[rt][0] for rt in road_type_arr])
speed_limits = np.array([ROAD_PROPS[rt][1] for rt in road_type_arr])
free_flow_speeds = np.array([ROAD_PROPS[rt][2] for rt in road_type_arr])
road_capacities = np.array([ROAD_PROPS[rt][3] for rt in road_type_arr])

# Distance of each edge midpoint from the city center (in meters)
dist_from_center = np.sqrt(
    ((edge_lats - CENTER_LAT) * METERS_PER_DEG_LAT) ** 2
    + ((edge_lngs - CENTER_LNG) * METERS_PER_DEG_LNG) ** 2
)

# ============================================================================
# 2. TIME RANGE — 120 days × 72 timestamps/day = 8640 timestamps
# ============================================================================
START_DATE = datetime(2024, 1, 1)
NUM_DAYS = 120
INTERVAL_MINUTES = 20
TIMESTAMPS_PER_DAY = 72
TOTAL_TIMESTAMPS = NUM_DAYS * TIMESTAMPS_PER_DAY  # 8640

timestamps = pd.date_range(start=START_DATE, periods=TOTAL_TIMESTAMPS, freq="20min")
hours = timestamps.hour + timestamps.minute / 60.0  # fractional hour
day_of_week = timestamps.dayofweek  # 0=Mon .. 6=Sun
is_weekend = (day_of_week >= 5).astype(int)

# Cyclic encoding of hour
hour_sin = np.sin(2 * np.pi * hours / 24.0)
hour_cos = np.cos(2 * np.pi * hours / 24.0)

# Indian national holidays falling in Jan–Apr 2024
HOLIDAYS = {
    datetime(2024, 1, 26),  # Republic Day
    datetime(2024, 3, 25),  # Holi
    datetime(2024, 3, 29),  # Good Friday
    datetime(2024, 4, 11),  # Eid-ul-Fitr (approx.)
    datetime(2024, 4, 14),  # Ambedkar Jayanti
    datetime(2024, 4, 17),  # Ram Navami
    datetime(2024, 4, 21),  # Mahavir Jayanti
}
is_holiday = np.array([ts.date() in {h.date() for h in HOLIDAYS} for ts in timestamps], dtype=int)

# ============================================================================
# 3. SPATIAL MULTIPLIER — CBD / Suburban
# ============================================================================
# CBD: edge midpoint within 500m of center → 1.5× occupancy
# Middle: 500–1000m → 1.0×
# Suburban: beyond 1000m → 0.7×
# (Thresholds scaled down from the original 2km/4km to match the smaller network)

spatial_mult = np.ones(NUM_EDGES)
spatial_mult[dist_from_center <= 500] = 1.5
spatial_mult[dist_from_center > 1000] = 0.7

# ============================================================================
# 4. BASE OCCUPANCY PROFILE (time-of-day pattern)
# ============================================================================
def base_occupancy_profile(hour_frac: np.ndarray) -> np.ndarray:
    """
    Return base occupancy (0–100) for each fractional hour.
    Weekday pattern (weekend adjustment applied separately).
      AM rush  8–10:  70–90%
      PM rush  17–19: 70–90%
      Midday   11–16: 40–60%
      Night    22–6:  10–30%
      Transitions smoothed linearly.
    """
    occ = np.empty_like(hour_frac, dtype=float)
    h = hour_frac

    # Night: 22–6 → 10–30 (use 20 as midpoint)
    night_mask = (h >= 22) | (h < 6)
    occ[night_mask] = 20.0

    # Morning ramp 6–8
    ramp_am = (h >= 6) & (h < 8)
    occ[ramp_am] = 20 + (h[ramp_am] - 6) / 2.0 * 60  # 20 → 80

    # AM rush 8–10
    am_rush = (h >= 8) & (h < 10)
    occ[am_rush] = 80.0

    # Post-AM 10–11
    post_am = (h >= 10) & (h < 11)
    occ[post_am] = 80 - (h[post_am] - 10) * 30  # 80 → 50

    # Midday 11–16
    midday = (h >= 11) & (h < 16)
    occ[midday] = 50.0

    # Pre-PM ramp 16–17
    pre_pm = (h >= 16) & (h < 17)
    occ[pre_pm] = 50 + (h[pre_pm] - 16) * 30  # 50 → 80

    # PM rush 17–19
    pm_rush = (h >= 17) & (h < 19)
    occ[pm_rush] = 80.0

    # Evening wind-down 19–22
    evening = (h >= 19) & (h < 22)
    occ[evening] = 80 - (h[evening] - 19) / 3.0 * 60  # 80 → 20

    return occ

base_occ_time = base_occupancy_profile(hours.values)

# ============================================================================
# 5. WEATHER GENERATION (per timestamp, city-wide)
# ============================================================================
WEATHER_OPTIONS = ["Clear", "Rain", "Fog"]
WEATHER_PROBS = [0.70, 0.20, 0.10]
weather_ts = np.random.choice(WEATHER_OPTIONS, size=TOTAL_TIMESTAMPS, p=WEATHER_PROBS)

# Weather occupancy multiplier
weather_mult_map = {"Clear": 1.0, "Rain": 1.15, "Fog": 1.05}
weather_mult_ts = np.array([weather_mult_map[w] for w in weather_ts])

# Temperature: seasonal + diurnal cycle (Mumbai Jan–Apr: ~20°C–35°C)
day_index = np.arange(TOTAL_TIMESTAMPS) // TIMESTAMPS_PER_DAY
seasonal_temp = 24 + 6 * np.sin(2 * np.pi * day_index / 365)  # slow seasonal
diurnal_temp = 4 * np.cos(2 * np.pi * (hours.values - 14) / 24)  # peak at 2 PM
temperature = seasonal_temp + diurnal_temp + np.random.normal(0, 1.5, TOTAL_TIMESTAMPS)

# Visibility: higher during day, lower at night; reduced in rain/fog
base_vis = 8000 + 4000 * np.cos(2 * np.pi * (hours.values - 12) / 24)  # 4k–12k m
vis_weather_factor = np.where(weather_ts == "Rain", 0.5,
                              np.where(weather_ts == "Fog", 0.3, 1.0))
visibility = base_vis * vis_weather_factor + np.random.normal(0, 500, TOTAL_TIMESTAMPS)
visibility = np.clip(visibility, 200, 15000)

# ============================================================================
# 6. RANDOM EVENTS — Accidents, Special Events
# ============================================================================
# --- 6a. Accidents: 3–5 per day, each affects 1 edge for 30–60 min (2–3 intervals)
accident_mask = np.zeros((TOTAL_TIMESTAMPS, NUM_EDGES), dtype=float)

for day in range(NUM_DAYS):
    n_accidents = np.random.randint(3, 6)
    for _ in range(n_accidents):
        edge = np.random.randint(NUM_EDGES)
        start_slot = day * TIMESTAMPS_PER_DAY + np.random.randint(TIMESTAMPS_PER_DAY)
        duration_slots = np.random.randint(2, 4)  # 2–3 slots = 40–60 min
        end_slot = min(start_slot + duration_slots, TOTAL_TIMESTAMPS)
        accident_mask[start_slot:end_slot, edge] = 0.25  # +25% occupancy

# --- 6b. Special events: 1–2 per week, affect nearby edges for 2–4 hours
# Build graph adjacency: edges that share a node are "neighbors"
edge_neighbors = [[] for _ in range(NUM_EDGES)]
for i in range(NUM_EDGES):
    for j in range(NUM_EDGES):
        if i == j:
            continue
        # Two edges are neighbors if they share at least one node
        nodes_i = {start_nodes[i], end_nodes[i]}
        nodes_j = {start_nodes[j], end_nodes[j]}
        if nodes_i & nodes_j:
            edge_neighbors[i].append(j)

# Also build 2-hop neighbors (neighbors of neighbors) for event spread
edge_neighbors_2hop = [set() for _ in range(NUM_EDGES)]
for i in range(NUM_EDGES):
    hop1 = set(edge_neighbors[i])
    hop2 = set()
    for j in hop1:
        hop2.update(edge_neighbors[j])
    hop2.discard(i)
    edge_neighbors_2hop[i] = hop2

event_mask = np.zeros((TOTAL_TIMESTAMPS, NUM_EDGES), dtype=float)
event_impact = np.zeros((TOTAL_TIMESTAMPS, NUM_EDGES), dtype=float)

num_weeks = NUM_DAYS // 7
for week in range(num_weeks + 1):
    n_events = np.random.randint(1, 3)
    for _ in range(n_events):
        event_day = week * 7 + np.random.randint(min(7, NUM_DAYS - week * 7))
        if event_day >= NUM_DAYS:
            continue
        event_hour = np.random.randint(16, 21)
        event_slot = event_day * TIMESTAMPS_PER_DAY + event_hour * 3
        duration_slots = np.random.randint(6, 13)  # 2–4 hours
        end_slot = min(event_slot + duration_slots, TOTAL_TIMESTAMPS)

        # Pick a random center edge for the event
        center_edge = np.random.randint(NUM_EDGES)

        # Affected edges: center (impact 1.0), 1-hop neighbors (0.8), 2-hop (0.6)
        event_mask[event_slot:end_slot, center_edge] = 0.40
        event_impact[event_slot:end_slot, center_edge] = 1.0

        for nb in edge_neighbors[center_edge]:
            event_mask[event_slot:end_slot, nb] = max(event_mask[event_slot, nb], 0.30)
            event_impact[event_slot:end_slot, nb] = max(event_impact[event_slot, nb], 0.8)

        for nb2 in edge_neighbors_2hop[center_edge]:
            event_mask[event_slot:end_slot, nb2] = max(event_mask[event_slot, nb2], 0.20)
            event_impact[event_slot:end_slot, nb2] = max(event_impact[event_slot, nb2], 0.6)

# ============================================================================
# 7. NEIGHBOR INFLUENCE (graph-based spatial smoothing)
# ============================================================================
# Already built in edge_neighbors above. 70% self + 30% avg of graph neighbors.

# ============================================================================
# 8. MAIN GENERATION LOOP — compute occupancy per (timestamp, edge)
# ============================================================================
print("=" * 60)
print("Mumbai Synthetic Traffic Dataset Generator (Small Network)")
print("=" * 60)
print(f"Network: {NUM_NODES} nodes, {NUM_EDGES} edges")
print(f"Time: {NUM_DAYS} days x {TIMESTAMPS_PER_DAY} intervals = {TOTAL_TIMESTAMPS} timestamps")
print(f"Total rows: {NUM_EDGES * TOTAL_TIMESTAMPS:,}")
print("-" * 60)
print("Generating occupancy matrix...")

CELL_CAPACITY = 100  # vehicles per edge segment

# Pre-allocate occupancy matrix (timestamps × edges)
occupancy = np.zeros((TOTAL_TIMESTAMPS, NUM_EDGES), dtype=np.float64)

gen_start = _time.time()

for t in range(TOTAL_TIMESTAMPS):
    # --- Progress bar ---
    if t % 200 == 0 or t == TOTAL_TIMESTAMPS - 1:
        pct = (t + 1) / TOTAL_TIMESTAMPS * 100
        bar_len = 40
        filled = int(bar_len * (t + 1) / TOTAL_TIMESTAMPS)
        bar = "█" * filled + "░" * (bar_len - filled)
        elapsed = _time.time() - gen_start
        eta = (elapsed / (t + 1)) * (TOTAL_TIMESTAMPS - t - 1) if t > 0 else 0
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}%  |  "
            f"Elapsed: {elapsed:.0f}s  |  ETA: {eta:.0f}s  |  "
            f"Timestep {t+1}/{TOTAL_TIMESTAMPS}"
        )
        sys.stdout.flush()

    # a) Base time-of-day occupancy
    occ = np.full(NUM_EDGES, base_occ_time[t])

    # b) Weekend reduction (-40%)
    if is_weekend[t]:
        occ *= 0.60

    # c) Holiday treated like weekend
    if is_holiday[t]:
        occ *= 0.65

    # d) Spatial multiplier (CBD / suburban)
    occ *= spatial_mult

    # e) Weather multiplier
    occ *= weather_mult_ts[t]

    # f) Accident overlay
    occ *= (1.0 + accident_mask[t])

    # g) Special event overlay
    occ *= (1.0 + event_mask[t])

    # h) Gaussian noise (std = 5%)
    noise = np.random.normal(0, 5, NUM_EDGES)
    occ += noise

    # i) Graph-based spatial smoothing: 70% self + 30% avg of neighboring edges
    occ_smoothed = occ.copy()
    for i in range(NUM_EDGES):
        nbrs = edge_neighbors[i]
        if nbrs:
            occ_smoothed[i] = 0.70 * occ[i] + 0.30 * np.mean(occ[nbrs])

    # j) Temporal smoothing: 60% current + 40% previous
    if t > 0:
        occ_smoothed = 0.60 * occ_smoothed + 0.40 * occupancy[t - 1]

    # k) Clip to valid range
    occ_smoothed = np.clip(occ_smoothed, 0, 100)

    occupancy[t] = occ_smoothed

print()  # newline after progress bar
print(f"Occupancy matrix generated in {_time.time() - gen_start:.1f}s")

# ============================================================================
# 9. DERIVE TRAFFIC METRICS FROM OCCUPANCY
# ============================================================================
print("Building final DataFrame...")

n_rows = TOTAL_TIMESTAMPS * NUM_EDGES

# Repeat edge-level arrays for each timestamp
edge_ids_full = np.tile(edge_ids, TOTAL_TIMESTAMPS)
start_nodes_full = np.tile(start_nodes, TOTAL_TIMESTAMPS)
end_nodes_full = np.tile(end_nodes, TOTAL_TIMESTAMPS)
road_lengths_full = np.tile(road_lengths, TOTAL_TIMESTAMPS)
road_type_full = np.tile(road_type_arr, TOTAL_TIMESTAMPS)
lane_counts_full = np.tile(lane_counts, TOTAL_TIMESTAMPS)
speed_limits_full = np.tile(speed_limits, TOTAL_TIMESTAMPS)
free_flow_full = np.tile(free_flow_speeds, TOTAL_TIMESTAMPS)
road_cap_full = np.tile(road_capacities, TOTAL_TIMESTAMPS)
lat_full = np.tile(edge_lats, TOTAL_TIMESTAMPS)
lng_full = np.tile(edge_lngs, TOTAL_TIMESTAMPS)

# Repeat timestamp-level arrays for each edge
ts_indices = np.repeat(np.arange(TOTAL_TIMESTAMPS), NUM_EDGES)
timestamps_full = timestamps[ts_indices]
hour_sin_full = np.repeat(hour_sin, NUM_EDGES)
hour_cos_full = np.repeat(hour_cos, NUM_EDGES)
dow_full = np.repeat(day_of_week, NUM_EDGES)
is_weekend_full = np.repeat(is_weekend, NUM_EDGES)
is_holiday_full = np.repeat(is_holiday, NUM_EDGES)
weather_full = np.repeat(weather_ts, NUM_EDGES)
temp_full = np.repeat(temperature, NUM_EDGES)
vis_full = np.repeat(visibility, NUM_EDGES)

# Flatten occupancy and event impact
occ_flat = occupancy.ravel()
event_impact_flat = event_impact.ravel()

# --- Derived fields ---
# Vehicle count: (occupancy / 100) * cell_capacity
vehicle_count = (occ_flat / 100.0) * CELL_CAPACITY
vehicle_count = np.clip(vehicle_count, 0, None).astype(int)

# Average speed: inverse relationship with occupancy
# At 0% occupancy → free_flow_speed; at 100% → ~5 kph (residual crawl)
avg_speed = free_flow_full * (1.0 - 0.95 * occ_flat / 100.0)
avg_speed = np.clip(avg_speed, 3.0, free_flow_full.astype(float))

# Congestion level: 1 - (avg_speed / free_flow_speed)
congestion = 1.0 - (avg_speed / free_flow_full)
congestion = np.clip(congestion, 0.0, 1.0)

# Travel time: road_length / (avg_speed * 0.277)  [seconds]
travel_time = road_lengths_full / (avg_speed * 0.277)

# ============================================================================
# 10. ASSEMBLE DATAFRAME
# ============================================================================
print("Assembling DataFrame...")

df = pd.DataFrame({
    "edge_id": edge_ids_full,
    "start_node_id": start_nodes_full,
    "end_node_id": end_nodes_full,
    "latitude": np.round(lat_full, 6),
    "longitude": np.round(lng_full, 6),
    "road_length_meters": np.round(road_lengths_full, 2),
    "road_type": road_type_full,
    "lane_count": lane_counts_full,
    "speed_limit_kph": speed_limits_full,
    "free_flow_speed_kph": free_flow_full,
    "road_capacity": road_cap_full,
    "timestamp": timestamps_full,
    "hour_sin": np.round(hour_sin_full, 6),
    "hour_cos": np.round(hour_cos_full, 6),
    "day_of_week": dow_full,
    "is_weekend": is_weekend_full.astype(bool),
    "is_holiday": is_holiday_full.astype(bool),
    "weather_condition": weather_full,
    "temperature_celsius": np.round(temp_full, 1),
    "visibility": np.round(vis_full, 1),
    "event_impact_score": np.round(event_impact_flat, 2),
    "average_speed_kph": np.round(avg_speed, 2),
    "vehicle_count": vehicle_count,
    "travel_time_seconds": np.round(travel_time, 2),
    "congestion_level": np.round(congestion, 4),
})

# ============================================================================
# 11. DATA VALIDATION
# ============================================================================
print("-" * 60)
print("Running data validation checks...")

checks_passed = 0
checks_total = 0

def validate(condition: bool, description: str):
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        checks_passed += 1
    print(f"  [{status}] {description}")

ROAD_TYPES = list(ROAD_PROPS.keys())

validate(len(df) == n_rows, f"Row count = {n_rows:,} (got {len(df):,})")
validate(df["edge_id"].nunique() == NUM_EDGES, f"Unique edges = {NUM_EDGES}")
validate(df["timestamp"].nunique() == TOTAL_TIMESTAMPS, f"Unique timestamps = {TOTAL_TIMESTAMPS}")
validate(df["congestion_level"].between(0, 1).all(), "Congestion in [0, 1]")
validate(df["average_speed_kph"].min() >= 0, "Average speed >= 0")
validate(
    (df["average_speed_kph"] <= df["free_flow_speed_kph"] + 0.01).all(),
    "Average speed <= free flow speed"
)
validate(df["vehicle_count"].min() >= 0, "Vehicle count >= 0")
validate(df["travel_time_seconds"].min() > 0, "Travel time > 0")
validate(not df.isnull().any().any(), "No null values")
validate(
    set(df["weather_condition"].unique()) == {"Clear", "Rain", "Fog"},
    "Weather conditions complete"
)
validate(
    set(df["road_type"].unique()) == set(ROAD_TYPES),
    "All road types present"
)
validate(df["event_impact_score"].between(0, 1).all(), "Event impact in [0, 1]")

# Validate graph coherence: every node used as start or end actually exists
all_node_ids_in_edges = set(df["start_node_id"].unique()) | set(df["end_node_id"].unique())
validate(
    all_node_ids_in_edges.issubset(set(NODE_POSITIONS.keys())),
    f"All edge node IDs are valid ({len(all_node_ids_in_edges)} unique nodes)"
)

print(f"\nValidation: {checks_passed}/{checks_total} checks passed")

# ============================================================================
# 12. SAVE TO CSV
# ============================================================================
OUTPUT_FILE = "mumbai_traffic_synthetic.csv"
print(f"\nSaving to {OUTPUT_FILE}...")
save_start = _time.time()
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved in {_time.time() - save_start:.1f}s")

# ============================================================================
# 13. SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(f"\nDataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"File size: {__import__('os').path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

print("\n--- Road Network ---")
edges_df = df.drop_duplicates("edge_id")
print(f"  Total nodes: {len(all_node_ids_in_edges)}")
print(f"  Total edges: {edges_df.shape[0]}")
print(f"  Road types: {dict(edges_df['road_type'].value_counts())}")
print(f"  Avg road length: {edges_df['road_length_meters'].mean():.1f} m")

print("\n--- Network Topology ---")
for _, row in edges_df.iterrows():
    print(f"  Edge {int(row['edge_id']):2d}: "
          f"Node {int(row['start_node_id']):2d} → Node {int(row['end_node_id']):2d}  "
          f"({row['road_type']:12s}, {row['road_length_meters']:.0f}m)")

print("\n--- Traffic Metrics ---")
print(f"  Avg speed (kph):       mean={df['average_speed_kph'].mean():.1f}, "
      f"std={df['average_speed_kph'].std():.1f}, "
      f"min={df['average_speed_kph'].min():.1f}, "
      f"max={df['average_speed_kph'].max():.1f}")
print(f"  Vehicle count:         mean={df['vehicle_count'].mean():.1f}, "
      f"std={df['vehicle_count'].std():.1f}, "
      f"min={df['vehicle_count'].min()}, "
      f"max={df['vehicle_count'].max()}")
print(f"  Congestion level:      mean={df['congestion_level'].mean():.4f}, "
      f"std={df['congestion_level'].std():.4f}")
print(f"  Travel time (s):       mean={df['travel_time_seconds'].mean():.1f}, "
      f"std={df['travel_time_seconds'].std():.1f}")

print("\n--- Temporal ---")
print(f"  Weekday records: {(~df['is_weekend']).sum():,}")
print(f"  Weekend records: {df['is_weekend'].sum():,}")
print(f"  Holiday records: {df['is_holiday'].sum():,}")

print("\n--- Weather ---")
for w in WEATHER_OPTIONS:
    count = (df["weather_condition"] == w).sum()
    pct = count / len(df) * 100
    print(f"  {w:8s}: {count:>10,} ({pct:.1f}%)")

print("\n--- Events ---")
print(f"  Records with event impact > 0: {(df['event_impact_score'] > 0).sum():,} "
      f"({(df['event_impact_score'] > 0).mean() * 100:.2f}%)")

print("\n--- Speed by Road Type ---")
speed_by_road = df.groupby("road_type")["average_speed_kph"].agg(["mean", "std", "min", "max"])
print(speed_by_road.round(1).to_string())

print("\n--- Speed by Time Period (Weekdays) ---")
weekday_df = df[~df["is_weekend"]]
hours_col = pd.to_datetime(weekday_df["timestamp"]).dt.hour
for label in ["Rush (8-10, 17-19)", "Off-peak (11-16)", "Night (22-6)"]:
    if "Rush" in label:
        mask = ((hours_col >= 8) & (hours_col < 10)) | ((hours_col >= 17) & (hours_col < 19))
    elif "Off-peak" in label:
        mask = (hours_col >= 11) & (hours_col < 16)
    else:
        mask = (hours_col >= 22) | (hours_col < 6)
    subset = weekday_df[mask]
    print(f"  {label}: avg_speed={subset['average_speed_kph'].mean():.1f}, "
          f"avg_congestion={subset['congestion_level'].mean():.3f}")

print("\n" + "=" * 60)
print("Generation complete!")
print("=" * 60)
