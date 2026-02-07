"""
Plot the Mumbai synthetic road network (~210 nodes, 400 edges, irregular layout).

Produces a figure with:
  - ~210 nodes (intersections) at their jittered lat/lng coordinates
  - 400 edges colored and styled by road type
  - Sparse node/edge labels to avoid clutter
  - CBD / Suburban zone rings overlaid
  - 350m reference grid in the background

Saves to: mumbai_road_network.png
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ============================================================================
# 1. NETWORK DEFINITION (mirrors generate_traffic_data.py exactly)
# ============================================================================
np.random.seed(42)

CENTER_LAT = 19.0760
CENTER_LNG = 72.8777

METERS_PER_DEG_LAT = 111_320.0
METERS_PER_DEG_LNG = 111_320.0 * np.cos(np.radians(CENTER_LAT))

# --- Step 1: Generate jittered base grid nodes ---
BASE_ROWS = 14
BASE_COLS = 15
ROW_SPACING_M = 170
COL_SPACING_M = 170
JITTER_STD_M = 50

NUM_BASE_NODES = BASE_ROWS * BASE_COLS  # 210

base_lat_offsets = np.zeros(NUM_BASE_NODES)
base_lng_offsets = np.zeros(NUM_BASE_NODES)
node_grid_row = np.zeros(NUM_BASE_NODES, dtype=int)
node_grid_col = np.zeros(NUM_BASE_NODES, dtype=int)

for r in range(BASE_ROWS):
    for c in range(BASE_COLS):
        nid = r * BASE_COLS + c
        base_lat_offsets[nid] = (BASE_ROWS // 2 - r) * ROW_SPACING_M
        base_lng_offsets[nid] = (c - BASE_COLS // 2) * COL_SPACING_M
        node_grid_row[nid] = r
        node_grid_col[nid] = c

jitter_lat = np.random.normal(0, JITTER_STD_M, NUM_BASE_NODES)
jitter_lng = np.random.normal(0, JITTER_STD_M, NUM_BASE_NODES)

node_lat_offsets = base_lat_offsets + jitter_lat
node_lng_offsets = base_lng_offsets + jitter_lng

node_lats = CENTER_LAT + node_lat_offsets / METERS_PER_DEG_LAT
node_lngs = CENTER_LNG + node_lng_offsets / METERS_PER_DEG_LNG

NODE_POSITIONS = {nid: (node_lat_offsets[nid], node_lng_offsets[nid])
                  for nid in range(NUM_BASE_NODES)}

# --- Step 2: Generate base grid edges ---
edge_set = set()

for r in range(BASE_ROWS):
    for c in range(BASE_COLS - 1):
        n1 = r * BASE_COLS + c
        n2 = r * BASE_COLS + c + 1
        edge_set.add((min(n1, n2), max(n1, n2)))

for r in range(BASE_ROWS - 1):
    for c in range(BASE_COLS):
        n1 = r * BASE_COLS + c
        n2 = (r + 1) * BASE_COLS + c
        edge_set.add((min(n1, n2), max(n1, n2)))

# --- Step 3: Remove ~50 peripheral edges ---
def compute_degrees(edges, n_nodes):
    deg = np.zeros(n_nodes, dtype=int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return deg

edge_list = list(edge_set)
edge_peripherality = []
for a, b in edge_list:
    dist_a = np.sqrt(node_lat_offsets[a]**2 + node_lng_offsets[a]**2)
    dist_b = np.sqrt(node_lat_offsets[b]**2 + node_lng_offsets[b]**2)
    edge_peripherality.append(min(dist_a, dist_b))

remove_order = np.argsort(edge_peripherality)[::-1]
n_to_remove = 50
n_removed = 0
degrees = compute_degrees(edge_set, NUM_BASE_NODES)

for idx in remove_order:
    if n_removed >= n_to_remove:
        break
    a, b = edge_list[idx]
    if degrees[a] > 1 and degrees[b] > 1:
        edge_set.discard((a, b))
        degrees[a] -= 1
        degrees[b] -= 1
        n_removed += 1

# --- Step 4: Add diagonal shortcut edges to reach 400 ---
TARGET_EDGES = 400
n_to_add = TARGET_EDGES - len(edge_set)

candidates = []
for i in range(NUM_BASE_NODES):
    for j in range(i + 1, NUM_BASE_NODES):
        if (i, j) in edge_set:
            continue
        if node_grid_row[i] == node_grid_row[j] or node_grid_col[i] == node_grid_col[j]:
            continue
        dlat = node_lat_offsets[i] - node_lat_offsets[j]
        dlng = node_lng_offsets[i] - node_lng_offsets[j]
        dist = np.sqrt(dlat**2 + dlng**2)
        if dist < 350:
            candidates.append((i, j, dist))

np.random.shuffle(candidates)
candidates.sort(key=lambda x: x[2])

n_added = 0
for i, j, dist in candidates:
    if n_added >= n_to_add:
        break
    edge_set.add((i, j))
    n_added += 1

# --- Step 5: BFS connectivity check ---
def bfs_connected(edges, n_nodes):
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

# Trim or pad to exactly 400
edge_list_final = sorted(edge_set)
if len(edge_list_final) > TARGET_EDGES:
    edge_list_final = edge_list_final[:TARGET_EDGES]
elif len(edge_list_final) < TARGET_EDGES:
    for i, j, dist in candidates:
        if len(edge_list_final) >= TARGET_EDGES:
            break
        if (i, j) not in edge_set:
            edge_list_final.append((i, j))
            edge_set.add((i, j))
    edge_list_final.sort()

NUM_EDGES = len(edge_list_final)
start_nodes = np.array([e[0] for e in edge_list_final])
end_nodes = np.array([e[1] for e in edge_list_final])

# --- Step 6: Assign road types ---
HIGHWAY_CORRIDOR_WIDTH_M = 150
ARTERIAL_CORRIDOR_ROWS = {3, 7, 11}
ARTERIAL_INNER_RADIUS_M = 500

highway_edges = set()
for idx, (a, b) in enumerate(edge_list_final):
    lng_a = abs(node_lng_offsets[a])
    lng_b = abs(node_lng_offsets[b])
    lat_diff = abs(node_lat_offsets[a] - node_lat_offsets[b])
    lng_diff = abs(node_lng_offsets[a] - node_lng_offsets[b])
    if (lng_a < HIGHWAY_CORRIDOR_WIDTH_M and lng_b < HIGHWAY_CORRIDOR_WIDTH_M
            and lat_diff > lng_diff):
        highway_edges.add(idx)

highway_nodes = set()
for idx in highway_edges:
    a, b = edge_list_final[idx]
    highway_nodes.add(a)
    highway_nodes.add(b)

road_types = []
for idx, (a, b) in enumerate(edge_list_final):
    if idx in highway_edges:
        road_types.append("Highway")
        continue
    row_a, row_b = node_grid_row[a], node_grid_row[b]
    dist_a = np.sqrt(node_lat_offsets[a]**2 + node_lng_offsets[a]**2)
    dist_b = np.sqrt(node_lat_offsets[b]**2 + node_lng_offsets[b]**2)
    is_arterial_row = (row_a in ARTERIAL_CORRIDOR_ROWS and row_b in ARTERIAL_CORRIDOR_ROWS
                       and row_a == row_b)
    is_arterial_inner = (dist_a < ARTERIAL_INNER_RADIUS_M or dist_b < ARTERIAL_INNER_RADIUS_M)
    shares_highway_node = (a in highway_nodes or b in highway_nodes)
    if is_arterial_row:
        road_types.append("Arterial")
    elif shares_highway_node and not is_arterial_inner:
        road_types.append("Ramp")
    elif is_arterial_inner:
        road_types.append("Arterial")
    else:
        road_types.append("Residential")

# Build EDGE_DEFS for plotting
EDGE_DEFS = [(idx, edge_list_final[idx][0], edge_list_final[idx][1], road_types[idx])
             for idx in range(NUM_EDGES)]

# Compute node degrees for display
final_degrees = compute_degrees(edge_list_final, NUM_BASE_NODES)

# ============================================================================
# 2. BUILD GRAPH
# ============================================================================
G = nx.Graph()
for nid in range(NUM_BASE_NODES):
    if final_degrees[nid] > 0:
        G.add_node(nid, pos=(node_lngs[nid], node_lats[nid]))

for eid_val, src, dst, rtype in EDGE_DEFS:
    G.add_edge(src, dst, edge_id=eid_val, road_type=rtype)

pos = nx.get_node_attributes(G, "pos")

# ============================================================================
# 3. STYLING (dark theme — matches TRAFFIC.OS UI)
# ============================================================================
BG_COLOR = "#1a1f2e"          # dark navy background
GRID_COLOR = "#2a3040"        # subtle grid lines
TEXT_COLOR = "#e0e6ed"         # light gray text
TEXT_DIM = "#7a8599"           # dimmed labels

ROAD_STYLE = {
    "Highway":     ("#ef4444",  5.0,  "-"),     # bright red
    "Arterial":    ("#38bdf8",  3.0,  "-"),     # cyan / sky blue
    "Residential": ("#4ade80",  1.6,  "-"),     # teal green
    "Ramp":        ("#f59e0b",  2.2,  "--"),    # amber / gold dashed
}

# Node colors by zone (matching UI accent palette)
NODE_CBD_COLOR = "#f472b6"      # pink (matches UI congestion pink)
NODE_MID_COLOR = "#38bdf8"      # cyan
NODE_SUB_COLOR = "#94a3b8"      # slate gray

CBD_RADIUS_M = 500
SUBURBAN_RADIUS_M = 1000

cbd_radius_lat = CBD_RADIUS_M / METERS_PER_DEG_LAT
cbd_radius_lng = CBD_RADIUS_M / METERS_PER_DEG_LNG
sub_radius_lat = SUBURBAN_RADIUS_M / METERS_PER_DEG_LAT
sub_radius_lng = SUBURBAN_RADIUS_M / METERS_PER_DEG_LNG

# ============================================================================
# 4. PLOT
# ============================================================================
fig, ax = plt.subplots(figsize=(18, 15))
fig.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- 4a. Background grid (350m spacing) ---
grid_extent_m = 1500
grid_step_m = 350
grid_offsets = np.arange(-grid_extent_m, grid_extent_m + 1, grid_step_m)

for off_m in grid_offsets:
    lng = CENTER_LNG + off_m / METERS_PER_DEG_LNG
    lat_lo = CENTER_LAT - grid_extent_m / METERS_PER_DEG_LAT
    lat_hi = CENTER_LAT + grid_extent_m / METERS_PER_DEG_LAT
    ax.plot([lng, lng], [lat_lo, lat_hi], color=GRID_COLOR, linewidth=0.4, zorder=0)

    lat = CENTER_LAT + off_m / METERS_PER_DEG_LAT
    lng_lo = CENTER_LNG - grid_extent_m / METERS_PER_DEG_LNG
    lng_hi = CENTER_LNG + grid_extent_m / METERS_PER_DEG_LNG
    ax.plot([lng_lo, lng_hi], [lat, lat], color=GRID_COLOR, linewidth=0.4, zorder=0)

# --- 4b. Zone circles ---
sub_circle = mpatches.Ellipse(
    (CENTER_LNG, CENTER_LAT), width=2 * sub_radius_lng, height=2 * sub_radius_lat,
    fill=True, facecolor="#38bdf8", edgecolor="#38bdf8", linewidth=0.8,
    alpha=0.06, linestyle=":", zorder=1)
ax.add_patch(sub_circle)

cbd_circle = mpatches.Ellipse(
    (CENTER_LNG, CENTER_LAT), width=2 * cbd_radius_lng, height=2 * cbd_radius_lat,
    fill=True, facecolor="#f472b6", edgecolor="#f472b6", linewidth=0.8,
    alpha=0.08, linestyle="--", zorder=1)
ax.add_patch(cbd_circle)

# --- 4c. Draw edges by road type ---
for eid_val, src, dst, rtype in EDGE_DEFS:
    color, lw, ls = ROAD_STYLE[rtype]
    x = [pos[src][0], pos[dst][0]]
    y = [pos[src][1], pos[dst][1]]
    ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, zorder=3,
            solid_capstyle="round", alpha=0.85)

    # Edge ID at midpoint — label every 20th edge
    if eid_val % 20 == 0:
        mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
        ax.text(mx, my, str(eid_val), fontsize=4, fontweight="bold", color=TEXT_DIM,
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                          alpha=0.85, linewidth=0.3))

# --- 4d. Draw nodes ---
node_x = [pos[n][0] for n in G.nodes()]
node_y = [pos[n][1] for n in G.nodes()]

node_colors = []
node_sizes = []
for n in G.nodes():
    dist = np.sqrt(NODE_POSITIONS[n][0]**2 + NODE_POSITIONS[n][1]**2)
    deg = final_degrees[n]
    if dist <= CBD_RADIUS_M:
        node_colors.append(NODE_CBD_COLOR)
        node_sizes.append(40)
    elif dist <= SUBURBAN_RADIUS_M:
        node_colors.append(NODE_MID_COLOR)
        node_sizes.append(25)
    else:
        node_colors.append(NODE_SUB_COLOR)
        node_sizes.append(15)
    if deg == 1:
        node_sizes[-1] = max(node_sizes[-1], 50)

ax.scatter(node_x, node_y, c=node_colors, s=node_sizes, zorder=6,
           edgecolors=BG_COLOR, linewidths=0.6)

# Node ID labels — label every 10th node
for n in G.nodes():
    if n % 10 == 0:
        x, y = pos[n]
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 5),
                    fontsize=4, ha="center", va="bottom", zorder=7, color=TEXT_DIM,
                    fontweight="bold")

# --- 4e. Center marker ---
ax.plot(CENTER_LNG, CENTER_LAT, marker="+", color="#f472b6", markersize=14,
        markeredgewidth=2, zorder=8)

# --- 4f. Legend ---
road_counts = {}
for _, _, _, rt in EDGE_DEFS:
    road_counts[rt] = road_counts.get(rt, 0) + 1

unique_degs, deg_counts = np.unique(final_degrees[final_degrees > 0], return_counts=True)

legend_elements = [
    Line2D([0], [0], color=ROAD_STYLE["Highway"][0], linewidth=ROAD_STYLE["Highway"][1],
           linestyle=ROAD_STYLE["Highway"][2], label=f"Highway ({road_counts.get('Highway',0)})"),
    Line2D([0], [0], color=ROAD_STYLE["Arterial"][0], linewidth=ROAD_STYLE["Arterial"][1],
           linestyle=ROAD_STYLE["Arterial"][2], label=f"Arterial ({road_counts.get('Arterial',0)})"),
    Line2D([0], [0], color=ROAD_STYLE["Residential"][0], linewidth=ROAD_STYLE["Residential"][1],
           linestyle=ROAD_STYLE["Residential"][2], label=f"Residential ({road_counts.get('Residential',0)})"),
    Line2D([0], [0], color=ROAD_STYLE["Ramp"][0], linewidth=ROAD_STYLE["Ramp"][1],
           linestyle=ROAD_STYLE["Ramp"][2], label=f"Ramp ({road_counts.get('Ramp',0)})"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=NODE_CBD_COLOR, markersize=8,
           label="CBD node (<500m)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=NODE_MID_COLOR, markersize=7,
           label="Middle node (500m-1km)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=NODE_SUB_COLOR, markersize=6,
           label="Suburban node (>1km)"),
    mpatches.Patch(facecolor="#f472b6", edgecolor="#f472b6", alpha=0.25,
                   linestyle="--", label="CBD zone (<500m)"),
    mpatches.Patch(facecolor="#38bdf8", edgecolor="#38bdf8", alpha=0.15,
                   linestyle=":", label="Suburban boundary (1km)"),
]
legend = ax.legend(handles=legend_elements, loc="lower left", fontsize=7,
                   framealpha=0.9, edgecolor=GRID_COLOR, facecolor="#232a3b",
                   labelcolor=TEXT_COLOR)

# --- 4g. Degree distribution text box ---
deg_text_lines = ["Node Degrees:"]
for d, c in zip(unique_degs, deg_counts):
    label = {1: "dead end", 2: "through", 3: "T-junction", 4: "crossroads"}.get(d, f"{d}-way")
    deg_text_lines.append(f"  deg {d} ({label}): {c}")
deg_text = "\n".join(deg_text_lines)

ax.text(0.98, 0.98, deg_text, transform=ax.transAxes, fontsize=6,
        verticalalignment="top", horizontalalignment="right", zorder=10,
        color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#232a3b", edgecolor=GRID_COLOR,
                  alpha=0.9))

# --- 4h. Axes and title ---
ax.set_xlabel("Longitude", fontsize=11, color=TEXT_COLOR)
ax.set_ylabel("Latitude", fontsize=11, color=TEXT_COLOR)

n_active = int(np.sum(final_degrees > 0))
ax.set_title(f"Mumbai Synthetic Road Network\n{n_active} nodes, {NUM_EDGES} edges (irregular jittered grid)",
             fontsize=14, fontweight="bold", color=TEXT_COLOR)
ax.set_aspect("equal")
ax.tick_params(labelsize=8, colors=TEXT_DIM)
for spine in ax.spines.values():
    spine.set_color(GRID_COLOR)

# Scale bar (500m)
scale_bar_m = 500
scale_bar_deg = scale_bar_m / METERS_PER_DEG_LNG
sb_x = ax.get_xlim()[1] - scale_bar_deg - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.03
sb_y = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04
ax.plot([sb_x, sb_x + scale_bar_deg], [sb_y, sb_y], color=TEXT_COLOR, linewidth=2.5, zorder=10)
ax.text(sb_x + scale_bar_deg / 2, sb_y, f"{scale_bar_m}m", fontsize=8,
        ha="center", va="bottom", fontweight="bold", zorder=10, color=TEXT_COLOR)

plt.tight_layout()

OUTPUT_FILE = "mumbai_road_network.png"
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved plot to {OUTPUT_FILE}")
plt.close()
