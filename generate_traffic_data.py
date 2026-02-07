"""
Synthetic Traffic Dataset Generator for Mumbai (Irregular Road Network)
======================================================================
Generates realistic traffic data over an organic-looking ~210-node, 400-edge
road network centered on Mumbai, spanning 7 days at 20-minute intervals.

The network starts from a 14×15 jittered grid, then:
  - Removes ~50 peripheral edges (creating dead ends and T-junctions)
  - Adds ~59 diagonal shortcut edges (creating irregular connectivity)
  - Assigns road types by spatial position (highway corridor, arterials, ramps)

Output: mumbai_traffic_synthetic.csv
Target: ~201,600 rows (400 edges × 504 timestamps)
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
# 1. ROAD NETWORK — ~210 nodes, 400 edges, irregular layout
# ============================================================================

CENTER_LAT = 19.0760
CENTER_LNG = 72.8777

METERS_PER_DEG_LAT = 111_320.0
METERS_PER_DEG_LNG = 111_320.0 * np.cos(np.radians(CENTER_LAT))

# --- Step 1: Generate jittered base grid nodes ---
BASE_ROWS = 14
BASE_COLS = 15
ROW_SPACING_M = 170   # ~170m between rows → spans ~2.2km
COL_SPACING_M = 170   # ~170m between cols → spans ~2.4km
JITTER_STD_M = 50     # ±50m random displacement per node

NUM_BASE_NODES = BASE_ROWS * BASE_COLS  # 210

# Base grid positions (lat_offset, lng_offset) in meters from center
base_lat_offsets = np.zeros(NUM_BASE_NODES)
base_lng_offsets = np.zeros(NUM_BASE_NODES)
node_grid_row = np.zeros(NUM_BASE_NODES, dtype=int)
node_grid_col = np.zeros(NUM_BASE_NODES, dtype=int)

for r in range(BASE_ROWS):
    for c in range(BASE_COLS):
        nid = r * BASE_COLS + c
        # Center the grid: row 0 = northernmost, col 0 = westernmost
        base_lat_offsets[nid] = (BASE_ROWS // 2 - r) * ROW_SPACING_M
        base_lng_offsets[nid] = (c - BASE_COLS // 2) * COL_SPACING_M
        node_grid_row[nid] = r
        node_grid_col[nid] = c

# Apply jitter (Gaussian noise) to break the perfect grid
jitter_lat = np.random.normal(0, JITTER_STD_M, NUM_BASE_NODES)
jitter_lng = np.random.normal(0, JITTER_STD_M, NUM_BASE_NODES)

node_lat_offsets = base_lat_offsets + jitter_lat
node_lng_offsets = base_lng_offsets + jitter_lng

# Convert to actual lat/lng
node_lats = CENTER_LAT + node_lat_offsets / METERS_PER_DEG_LAT
node_lngs = CENTER_LNG + node_lng_offsets / METERS_PER_DEG_LNG

# Node positions dict for compatibility
NODE_POSITIONS = {nid: (node_lat_offsets[nid], node_lng_offsets[nid])
                  for nid in range(NUM_BASE_NODES)}

# --- Step 2: Generate base grid edges (391) ---
edge_set = set()   # set of (min_node, max_node) tuples for dedup

# Horizontal edges: connect (r, c) → (r, c+1) for all valid
for r in range(BASE_ROWS):
    for c in range(BASE_COLS - 1):
        n1 = r * BASE_COLS + c
        n2 = r * BASE_COLS + c + 1
        edge_set.add((min(n1, n2), max(n1, n2)))

# Vertical edges: connect (r, c) → (r+1, c) for all valid
for r in range(BASE_ROWS - 1):
    for c in range(BASE_COLS):
        n1 = r * BASE_COLS + c
        n2 = (r + 1) * BASE_COLS + c
        edge_set.add((min(n1, n2), max(n1, n2)))

print(f"Base grid edges: {len(edge_set)}")

# --- Step 3: Remove ~50 edges to create dead ends and T-junctions ---
# Compute node degrees
def compute_degrees(edges, n_nodes):
    deg = np.zeros(n_nodes, dtype=int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return deg

# Prefer removing peripheral edges (both endpoints far from center)
edge_list = list(edge_set)
edge_peripherality = []
for a, b in edge_list:
    dist_a = np.sqrt(node_lat_offsets[a]**2 + node_lng_offsets[a]**2)
    dist_b = np.sqrt(node_lat_offsets[b]**2 + node_lng_offsets[b]**2)
    edge_peripherality.append(min(dist_a, dist_b))  # how far the closer node is

# Sort by peripherality (most peripheral first) and try to remove
remove_order = np.argsort(edge_peripherality)[::-1]  # most peripheral first
n_to_remove = 50
n_removed = 0
degrees = compute_degrees(edge_set, NUM_BASE_NODES)

for idx in remove_order:
    if n_removed >= n_to_remove:
        break
    a, b = edge_list[idx]
    # Only remove if neither node drops to degree 0
    if degrees[a] > 1 and degrees[b] > 1:
        edge_set.discard((a, b))
        degrees[a] -= 1
        degrees[b] -= 1
        n_removed += 1

print(f"After removing {n_removed} edges: {len(edge_set)}")

# --- Step 4: Add diagonal shortcut edges to reach 400 ---
# Find candidate pairs: close nodes that aren't already connected and aren't
# on the same grid row or column (to create true diagonals)
TARGET_EDGES = 400
n_to_add = TARGET_EDGES - len(edge_set)

# Compute pairwise distances for nearby nodes (only check within 350m)
candidates = []
for i in range(NUM_BASE_NODES):
    for j in range(i + 1, NUM_BASE_NODES):
        if (i, j) in edge_set:
            continue
        # Must be on different row AND different column (true diagonal)
        if node_grid_row[i] == node_grid_row[j] or node_grid_col[i] == node_grid_col[j]:
            continue
        dlat = node_lat_offsets[i] - node_lat_offsets[j]
        dlng = node_lng_offsets[i] - node_lng_offsets[j]
        dist = np.sqrt(dlat**2 + dlng**2)
        if dist < 350:
            candidates.append((i, j, dist))

# Sort by distance (prefer shorter diagonals) and add with some randomness
np.random.shuffle(candidates)  # shuffle first for variety
candidates.sort(key=lambda x: x[2])  # then sort by distance

n_added = 0
for i, j, dist in candidates:
    if n_added >= n_to_add:
        break
    edge_set.add((i, j))
    n_added += 1

print(f"After adding {n_added} diagonals: {len(edge_set)}")

# --- Step 5: BFS connectivity check ---
def bfs_connected(edges, n_nodes):
    """Check if all nodes with degree > 0 are connected. Return unreachable set."""
    adj = [[] for _ in range(n_nodes)]
    has_edge = set()
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        has_edge.add(a)
        has_edge.add(b)

    if not has_edge:
        return set()

    start = min(has_edge)
    visited = set()
    queue = [start]
    visited.add(start)
    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    return has_edge - visited

unreachable = bfs_connected(edge_set, NUM_BASE_NODES)
if unreachable:
    print(f"WARNING: {len(unreachable)} unreachable nodes, reconnecting...")
    # Connect unreachable nodes to their nearest reachable neighbour
    reachable = set()
    for a, b in edge_set:
        reachable.add(a)
        reachable.add(b)
    reachable -= unreachable

    for node in unreachable:
        best_dist = float("inf")
        best_target = None
        for target in reachable:
            d = np.sqrt((node_lat_offsets[node] - node_lat_offsets[target])**2 +
                        (node_lng_offsets[node] - node_lng_offsets[target])**2)
            if d < best_dist:
                best_dist = d
                best_target = target
        if best_target is not None:
            edge_set.add((min(node, best_target), max(node, best_target)))
            reachable.add(node)

# Trim or pad to exactly 400 if needed
edge_list_final = sorted(edge_set)
if len(edge_list_final) > TARGET_EDGES:
    # Remove excess (from the longest diagonal edges added)
    edge_list_final = edge_list_final[:TARGET_EDGES]
elif len(edge_list_final) < TARGET_EDGES:
    # Add more diagonals
    for i, j, dist in candidates:
        if len(edge_list_final) >= TARGET_EDGES:
            break
        if (i, j) not in edge_set:
            edge_list_final.append((i, j))
            edge_set.add((i, j))
    edge_list_final.sort()

NUM_EDGES = len(edge_list_final)
print(f"Final network: {NUM_BASE_NODES} nodes, {NUM_EDGES} edges")

# --- Step 6: Assign road types based on spatial position ---
# Highway: edges along the central vertical corridor (both nodes near center lng)
# Arterial: edges on 3 horizontal corridors (base rows 3, 7, 11) or near-center edges
# Ramp: non-highway/arterial edges that share a node with a highway edge
# Residential: everything else

HIGHWAY_CORRIDOR_WIDTH_M = 150   # nodes within 150m of center longitude
ARTERIAL_CORRIDOR_ROWS = {3, 7, 11}  # base grid rows acting as arterial corridors
ARTERIAL_INNER_RADIUS_M = 500    # edges near center get arterial status

# Identify highway edges: both endpoints close to the central vertical line
highway_edges = set()
for idx, (a, b) in enumerate(edge_list_final):
    # Both nodes must be near center longitude AND edge must be roughly vertical
    lng_a = abs(node_lng_offsets[a])
    lng_b = abs(node_lng_offsets[b])
    lat_diff = abs(node_lat_offsets[a] - node_lat_offsets[b])
    lng_diff = abs(node_lng_offsets[a] - node_lng_offsets[b])

    if (lng_a < HIGHWAY_CORRIDOR_WIDTH_M and lng_b < HIGHWAY_CORRIDOR_WIDTH_M
            and lat_diff > lng_diff):  # more vertical than horizontal
        highway_edges.add(idx)

# Identify highway-connected nodes (for ramp detection)
highway_nodes = set()
for idx in highway_edges:
    a, b = edge_list_final[idx]
    highway_nodes.add(a)
    highway_nodes.add(b)

# Assign road types
road_types = []
for idx, (a, b) in enumerate(edge_list_final):
    if idx in highway_edges:
        road_types.append("Highway")
        continue

    # Arterial: on a designated corridor row or close to center
    row_a, row_b = node_grid_row[a], node_grid_row[b]
    dist_a = np.sqrt(node_lat_offsets[a]**2 + node_lng_offsets[a]**2)
    dist_b = np.sqrt(node_lat_offsets[b]**2 + node_lng_offsets[b]**2)

    is_arterial_row = (row_a in ARTERIAL_CORRIDOR_ROWS and row_b in ARTERIAL_CORRIDOR_ROWS
                       and row_a == row_b)
    is_arterial_inner = (dist_a < ARTERIAL_INNER_RADIUS_M or dist_b < ARTERIAL_INNER_RADIUS_M)

    # Ramp: shares a node with the highway but isn't highway itself
    shares_highway_node = (a in highway_nodes or b in highway_nodes)

    if is_arterial_row:
        road_types.append("Arterial")
    elif shares_highway_node and not is_arterial_inner:
        road_types.append("Ramp")
    elif is_arterial_inner:
        road_types.append("Arterial")
    else:
        road_types.append("Residential")

# Build final edge arrays
edge_ids = np.arange(NUM_EDGES)
start_nodes = np.array([e[0] for e in edge_list_final])
end_nodes = np.array([e[1] for e in edge_list_final])
road_type_arr = np.array(road_types)

# Compute edge midpoint lat/lng and road length
edge_lats = (node_lats[start_nodes] + node_lats[end_nodes]) / 2.0
edge_lngs = (node_lngs[start_nodes] + node_lngs[end_nodes]) / 2.0

dlat_m = (node_lats[end_nodes] - node_lats[start_nodes]) * METERS_PER_DEG_LAT
dlng_m = (node_lngs[end_nodes] - node_lngs[start_nodes]) * METERS_PER_DEG_LNG
road_lengths = np.sqrt(dlat_m**2 + dlng_m**2)

# Properties per road type
ROAD_PROPS = {
    "Highway":      (6, 80, 90, 4000),
    "Arterial":     (4, 60, 70, 2500),
    "Residential":  (2, 30, 35, 800),
    "Ramp":         (2, 40, 50, 1200),
}

lane_counts      = np.array([ROAD_PROPS[rt][0] for rt in road_type_arr])
speed_limits     = np.array([ROAD_PROPS[rt][1] for rt in road_type_arr])
free_flow_speeds = np.array([ROAD_PROPS[rt][2] for rt in road_type_arr])
road_capacities  = np.array([ROAD_PROPS[rt][3] for rt in road_type_arr])

# Distance of each edge midpoint from the city center (in meters)
dist_from_center = np.sqrt(
    ((edge_lats - CENTER_LAT) * METERS_PER_DEG_LAT) ** 2
    + ((edge_lngs - CENTER_LNG) * METERS_PER_DEG_LNG) ** 2
)

# Node degree distribution (for summary)
final_degrees = compute_degrees(edge_list_final, NUM_BASE_NODES)

# ============================================================================
# 2. TIME RANGE — 7 days × 72 timestamps/day = 504 timestamps
# ============================================================================
START_DATE = datetime(2024, 1, 1)
NUM_DAYS = 7
INTERVAL_MINUTES = 20
TIMESTAMPS_PER_DAY = 72
TOTAL_TIMESTAMPS = NUM_DAYS * TIMESTAMPS_PER_DAY  # 504

timestamps = pd.date_range(start=START_DATE, periods=TOTAL_TIMESTAMPS, freq="20min")
hours = timestamps.hour + timestamps.minute / 60.0
day_of_week = timestamps.dayofweek
is_weekend = (day_of_week >= 5).astype(int)

hour_sin = np.sin(2 * np.pi * hours / 24.0)
hour_cos = np.cos(2 * np.pi * hours / 24.0)

# No national holidays in Jan 1–7
HOLIDAYS = set()
is_holiday = np.zeros(TOTAL_TIMESTAMPS, dtype=int)

# ============================================================================
# 3. SPATIAL MULTIPLIER — CBD / Suburban
# ============================================================================
spatial_mult = np.ones(NUM_EDGES)
spatial_mult[dist_from_center <= 500] = 1.5
spatial_mult[dist_from_center > 1000] = 0.7

# ============================================================================
# 4. BASE OCCUPANCY PROFILE (time-of-day pattern)
# ============================================================================
def base_occupancy_profile(hour_frac: np.ndarray) -> np.ndarray:
    occ = np.empty_like(hour_frac, dtype=float)
    h = hour_frac

    night_mask = (h >= 22) | (h < 6)
    occ[night_mask] = 20.0

    ramp_am = (h >= 6) & (h < 8)
    occ[ramp_am] = 20 + (h[ramp_am] - 6) / 2.0 * 60

    am_rush = (h >= 8) & (h < 10)
    occ[am_rush] = 80.0

    post_am = (h >= 10) & (h < 11)
    occ[post_am] = 80 - (h[post_am] - 10) * 30

    midday = (h >= 11) & (h < 16)
    occ[midday] = 50.0

    pre_pm = (h >= 16) & (h < 17)
    occ[pre_pm] = 50 + (h[pre_pm] - 16) * 30

    pm_rush = (h >= 17) & (h < 19)
    occ[pm_rush] = 80.0

    evening = (h >= 19) & (h < 22)
    occ[evening] = 80 - (h[evening] - 19) / 3.0 * 60

    return occ

base_occ_time = base_occupancy_profile(hours.values)

# ============================================================================
# 5. WEATHER GENERATION (per timestamp, city-wide)
# ============================================================================
WEATHER_OPTIONS = ["Clear", "Rain", "Fog"]
WEATHER_PROBS = [0.70, 0.20, 0.10]
weather_ts = np.random.choice(WEATHER_OPTIONS, size=TOTAL_TIMESTAMPS, p=WEATHER_PROBS)

weather_mult_map = {"Clear": 1.0, "Rain": 1.15, "Fog": 1.05}
weather_mult_ts = np.array([weather_mult_map[w] for w in weather_ts])

day_index = np.arange(TOTAL_TIMESTAMPS) // TIMESTAMPS_PER_DAY
seasonal_temp = 24 + 6 * np.sin(2 * np.pi * day_index / 365)
diurnal_temp = 4 * np.cos(2 * np.pi * (hours.values - 14) / 24)
temperature = seasonal_temp + diurnal_temp + np.random.normal(0, 1.5, TOTAL_TIMESTAMPS)

base_vis = 8000 + 4000 * np.cos(2 * np.pi * (hours.values - 12) / 24)
vis_weather_factor = np.where(weather_ts == "Rain", 0.5,
                              np.where(weather_ts == "Fog", 0.3, 1.0))
visibility = base_vis * vis_weather_factor + np.random.normal(0, 500, TOTAL_TIMESTAMPS)
visibility = np.clip(visibility, 200, 15000)

# ============================================================================
# 6. RANDOM EVENTS — Accidents, Special Events
# ============================================================================
accident_mask = np.zeros((TOTAL_TIMESTAMPS, NUM_EDGES), dtype=float)

for day in range(NUM_DAYS):
    n_accidents = np.random.randint(3, 6)
    for _ in range(n_accidents):
        edge = np.random.randint(NUM_EDGES)
        start_slot = day * TIMESTAMPS_PER_DAY + np.random.randint(TIMESTAMPS_PER_DAY)
        duration_slots = np.random.randint(2, 4)
        end_slot = min(start_slot + duration_slots, TOTAL_TIMESTAMPS)
        accident_mask[start_slot:end_slot, edge] = 0.25

# Graph adjacency: edges sharing a node are neighbors
edge_neighbors = [[] for _ in range(NUM_EDGES)]
for i in range(NUM_EDGES):
    nodes_i = {start_nodes[i], end_nodes[i]}
    for j in range(i + 1, NUM_EDGES):
        nodes_j = {start_nodes[j], end_nodes[j]}
        if nodes_i & nodes_j:
            edge_neighbors[i].append(j)
            edge_neighbors[j].append(i)

# 2-hop neighbors for event spread
edge_neighbors_2hop = [set() for _ in range(NUM_EDGES)]
for i in range(NUM_EDGES):
    hop1 = set(edge_neighbors[i])
    hop2 = set()
    for j in hop1:
        hop2.update(edge_neighbors[j])
    hop2.discard(i)
    edge_neighbors_2hop[i] = hop2

# Special events: 1–2 per week
event_mask = np.zeros((TOTAL_TIMESTAMPS, NUM_EDGES), dtype=float)
event_impact = np.zeros((TOTAL_TIMESTAMPS, NUM_EDGES), dtype=float)

num_weeks = max(1, NUM_DAYS // 7)
for week in range(num_weeks + 1):
    n_events = np.random.randint(1, 3)
    for _ in range(n_events):
        remaining = min(7, NUM_DAYS - week * 7)
        if remaining <= 0:
            continue
        event_day = week * 7 + np.random.randint(remaining)
        if event_day >= NUM_DAYS:
            continue
        event_hour = np.random.randint(16, 21)
        event_slot = event_day * TIMESTAMPS_PER_DAY + event_hour * 3
        duration_slots = np.random.randint(6, 13)
        end_slot = min(event_slot + duration_slots, TOTAL_TIMESTAMPS)

        center_edge = np.random.randint(NUM_EDGES)

        event_mask[event_slot:end_slot, center_edge] = 0.40
        event_impact[event_slot:end_slot, center_edge] = 1.0

        for nb in edge_neighbors[center_edge]:
            event_mask[event_slot:end_slot, nb] = np.maximum(
                event_mask[event_slot:end_slot, nb], 0.30)
            event_impact[event_slot:end_slot, nb] = np.maximum(
                event_impact[event_slot:end_slot, nb], 0.8)

        for nb2 in edge_neighbors_2hop[center_edge]:
            event_mask[event_slot:end_slot, nb2] = np.maximum(
                event_mask[event_slot:end_slot, nb2], 0.20)
            event_impact[event_slot:end_slot, nb2] = np.maximum(
                event_impact[event_slot:end_slot, nb2], 0.6)

# ============================================================================
# 7. MAIN GENERATION LOOP — compute occupancy per (timestamp, edge)
# ============================================================================
# Degree distribution summary
unique_degs, deg_counts = np.unique(final_degrees[final_degrees > 0], return_counts=True)
deg_dist_str = ", ".join(f"deg {d}: {c} nodes" for d, c in zip(unique_degs, deg_counts))

road_type_counts = {}
for rt in road_type_arr:
    road_type_counts[rt] = road_type_counts.get(rt, 0) + 1

print("=" * 60)
print("Mumbai Synthetic Traffic Dataset Generator")
print("=" * 60)
print(f"Network: {np.sum(final_degrees > 0)} nodes, {NUM_EDGES} edges")
print(f"  Road types: {road_type_counts}")
print(f"  Node degrees: {deg_dist_str}")
print(f"Time: {NUM_DAYS} days x {TIMESTAMPS_PER_DAY} intervals = {TOTAL_TIMESTAMPS} timestamps")
print(f"Total rows: {NUM_EDGES * TOTAL_TIMESTAMPS:,}")
print("-" * 60)
print("Generating occupancy matrix...")

CELL_CAPACITY = 100

occupancy = np.zeros((TOTAL_TIMESTAMPS, NUM_EDGES), dtype=np.float64)

gen_start = _time.time()

for t in range(TOTAL_TIMESTAMPS):
    if t % 50 == 0 or t == TOTAL_TIMESTAMPS - 1:
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

    occ = np.full(NUM_EDGES, base_occ_time[t])

    if is_weekend[t]:
        occ *= 0.60
    if is_holiday[t]:
        occ *= 0.65

    occ *= spatial_mult
    occ *= weather_mult_ts[t]
    occ *= (1.0 + accident_mask[t])
    occ *= (1.0 + event_mask[t])

    noise = np.random.normal(0, 5, NUM_EDGES)
    occ += noise

    occ_smoothed = occ.copy()
    for i in range(NUM_EDGES):
        nbrs = edge_neighbors[i]
        if nbrs:
            occ_smoothed[i] = 0.70 * occ[i] + 0.30 * np.mean(occ[nbrs])

    if t > 0:
        occ_smoothed = 0.60 * occ_smoothed + 0.40 * occupancy[t - 1]

    occ_smoothed = np.clip(occ_smoothed, 0, 100)
    occupancy[t] = occ_smoothed

print()
print(f"Occupancy matrix generated in {_time.time() - gen_start:.1f}s")

# ============================================================================
# 8. DERIVE TRAFFIC METRICS FROM OCCUPANCY
# ============================================================================
print("Building final DataFrame...")

n_rows = TOTAL_TIMESTAMPS * NUM_EDGES

edge_ids_full     = np.tile(edge_ids, TOTAL_TIMESTAMPS)
start_nodes_full  = np.tile(start_nodes, TOTAL_TIMESTAMPS)
end_nodes_full    = np.tile(end_nodes, TOTAL_TIMESTAMPS)
road_lengths_full = np.tile(road_lengths, TOTAL_TIMESTAMPS)
road_type_full    = np.tile(road_type_arr, TOTAL_TIMESTAMPS)
lane_counts_full  = np.tile(lane_counts, TOTAL_TIMESTAMPS)
speed_limits_full = np.tile(speed_limits, TOTAL_TIMESTAMPS)
free_flow_full    = np.tile(free_flow_speeds, TOTAL_TIMESTAMPS)
road_cap_full     = np.tile(road_capacities, TOTAL_TIMESTAMPS)
lat_full          = np.tile(edge_lats, TOTAL_TIMESTAMPS)
lng_full          = np.tile(edge_lngs, TOTAL_TIMESTAMPS)

ts_indices        = np.repeat(np.arange(TOTAL_TIMESTAMPS), NUM_EDGES)
timestamps_full   = timestamps[ts_indices]
hour_sin_full     = np.repeat(hour_sin, NUM_EDGES)
hour_cos_full     = np.repeat(hour_cos, NUM_EDGES)
dow_full          = np.repeat(day_of_week, NUM_EDGES)
is_weekend_full   = np.repeat(is_weekend, NUM_EDGES)
is_holiday_full   = np.repeat(is_holiday, NUM_EDGES)
weather_full      = np.repeat(weather_ts, NUM_EDGES)
temp_full         = np.repeat(temperature, NUM_EDGES)
vis_full          = np.repeat(visibility, NUM_EDGES)

occ_flat = occupancy.ravel()
event_impact_flat = event_impact.ravel()

vehicle_count = (occ_flat / 100.0) * CELL_CAPACITY
vehicle_count = np.clip(vehicle_count, 0, None).astype(int)

avg_speed = free_flow_full * (1.0 - 0.95 * occ_flat / 100.0)
avg_speed = np.clip(avg_speed, 3.0, free_flow_full.astype(float))

congestion = 1.0 - (avg_speed / free_flow_full)
congestion = np.clip(congestion, 0.0, 1.0)

travel_time = road_lengths_full / (avg_speed * 0.277)

# ============================================================================
# 9. ASSEMBLE DATAFRAME
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
# 10. DATA VALIDATION
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

all_node_ids_in_edges = set(df["start_node_id"].unique()) | set(df["end_node_id"].unique())
validate(
    all_node_ids_in_edges.issubset(set(NODE_POSITIONS.keys())),
    f"All edge node IDs are valid ({len(all_node_ids_in_edges)} unique nodes)"
)

print(f"\nValidation: {checks_passed}/{checks_total} checks passed")

# ============================================================================
# 11. SAVE TO CSV
# ============================================================================
OUTPUT_FILE = "mumbai_traffic_synthetic.csv"
print(f"\nSaving to {OUTPUT_FILE}...")
save_start = _time.time()
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved in {_time.time() - save_start:.1f}s")

# ============================================================================
# 12. SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(f"\nDataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"File size: {__import__('os').path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

print("\n--- Road Network ---")
edges_df = df.drop_duplicates("edge_id")
n_active_nodes = len(all_node_ids_in_edges)
print(f"  Total nodes: {n_active_nodes}")
print(f"  Total edges: {edges_df.shape[0]}")
print(f"  Road types: {dict(edges_df['road_type'].value_counts())}")
print(f"  Avg road length: {edges_df['road_length_meters'].mean():.1f} m")

print("\n--- Node Degree Distribution ---")
for d, c in zip(unique_degs, deg_counts):
    label = {1: "dead end", 2: "through", 3: "T-junction", 4: "crossroads"}.get(d, f"{d}-way")
    print(f"  Degree {d} ({label}): {c} nodes")

print("\n--- Spatial Zones ---")
n_cbd = int(np.sum(dist_from_center <= 500))
n_mid = int(np.sum((dist_from_center > 500) & (dist_from_center <= 1000)))
n_sub = int(np.sum(dist_from_center > 1000))
print(f"  CBD (<500m):      {n_cbd} edges")
print(f"  Middle (500-1km): {n_mid} edges")
print(f"  Suburban (>1km):  {n_sub} edges")

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
