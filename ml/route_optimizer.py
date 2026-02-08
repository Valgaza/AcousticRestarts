"""
Route Optimizer with Game-Theoretic Load-Balancing
===================================================
Given (start_node, end_node, timestamp), finds the optimal route through
the traffic network while load-balancing across concurrent users.

Pipeline:
  1. Read CSV from backend/uploads/ → build adjacency list at runtime
  2. Find K=3 shortest paths using Yen's algorithm (Dijkstra-based)
  3. Score each path: congestion from CSV + forecast predictions + load penalty
  4. Choose lowest-cost route, record in persistent ledger
  5. Output modified CSV rows (affected edges only) for re-prediction

Usage:
  python route_optimizer.py --start 0 --end 150 --timestamp "2026-02-08T03:00:00"
"""

import json
import heapq
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------

def _workspace_root():
    cur = Path(__file__).resolve().parent
    while cur.parent != cur:
        if (cur / "backend").exists() and (cur / "ml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def _first_csv(uploads_dir):
    csvs = sorted(uploads_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    return str(csvs[0]) if csvs else None


# ---------------------------------------------------------------------------
# 1. Build graph from CSV
# ---------------------------------------------------------------------------

TRAFFIC_COLS = [
    "vehicle_count", "average_speed_kph", "travel_time_seconds",
    "congestion_level", "neighbor_avg_congestion_t-1",
    "neighbor_avg_speed_t-1", "upstream_congestion_t-1",
    "downstream_congestion_t-1", "hour_of_day", "day_of_week",
]


def build_graph(csv_path):
    """Read the traffic CSV and return all structures needed for routing.

    Returns
    -------
    adj : dict[int, list[tuple(int, int, float)]]
        node → [(neighbour, edge_id, distance)]  (undirected)
    edges : dict[int, dict]
        edge_id → static edge info (start_node, end_node, length, lat, lon …)
    edge_by_nodes : dict[tuple(int,int), int]
        (min_node, max_node) → edge_id
    edge_traffic : dict[int, dict[int, dict]]
        edge_id → {time_idx → {col: value}}
    time_indices : list[int]
        sorted unique time_idx values
    df : DataFrame
        the raw CSV kept for output generation
    """
    df = pd.read_csv(csv_path)

    adj = defaultdict(list)           # node  → [(nbr, eid, dist)]
    edges = {}                        # eid   → static props
    edge_by_nodes = {}                # (lo, hi) → eid
    edge_traffic = defaultdict(dict)  # eid   → {tidx → row dict}
    time_indices = sorted(df["time_idx"].unique().tolist())

    for _, row in df.iterrows():
        eid = int(row["edge_id"])
        sn  = int(row["start_node_id"])
        en  = int(row["end_node_id"])
        tidx = int(row["time_idx"])

        # Per-timestep traffic state
        edge_traffic[eid][tidx] = {c: row[c] for c in TRAFFIC_COLS}

        # Static topology (only once per edge)
        if eid not in edges:
            length = float(row["road_length_meters"])
            edges[eid] = {
                "start_node": sn,
                "end_node": en,
                "length": length,
                "road_type": row["road_type"],
                "lane_count": int(row["lane_count"]),
                "speed_limit": float(row["speed_limit_kph"]),
                "free_flow_speed": float(row["free_flow_speed_kph"]),
                "capacity": int(row["road_capacity"]),
                "lat": float(row["latitude_midroad"]),
                "lon": float(row["longitude_midroad"]),
            }
            pair = (min(sn, en), max(sn, en))
            edge_by_nodes[pair] = eid
            adj[sn].append((en, eid, length))
            adj[en].append((sn, eid, length))

    return adj, edges, edge_by_nodes, edge_traffic, time_indices, df


def _edge_between(edge_by_nodes, a, b):
    return edge_by_nodes.get((min(a, b), max(a, b)))


def _find_time_idx(edge_traffic, edges, time_indices, timestamp):
    """Map a datetime to the closest time_idx by hour-of-day + day-of-week."""
    target_h = timestamp.hour + timestamp.minute / 60.0
    target_dow = timestamp.weekday()

    ref_eid = next(iter(edges))
    ref = edge_traffic[ref_eid]

    best, best_score = time_indices[-1], float("inf")
    for tidx in time_indices:
        if tidx not in ref:
            continue
        h = float(ref[tidx]["hour_of_day"])
        d = int(ref[tidx]["day_of_week"])
        score = min(abs(h - target_h), 24 - abs(h - target_h))
        if d != target_dow:
            score += 48
        if score < best_score:
            best_score = score
            best = tidx
    return best


def _neighbor_edges(adj, edges, eid):
    """Return the set of edge-ids that share a node with *eid*."""
    info = edges[eid]
    nbrs = set()
    for _, neid, _ in adj[info["start_node"]]:
        if neid != eid:
            nbrs.add(neid)
    for _, neid, _ in adj[info["end_node"]]:
        if neid != eid:
            nbrs.add(neid)
    return nbrs


# ---------------------------------------------------------------------------
# 2. Shortest-path algorithms (Dijkstra + Yen's K-shortest)
# ---------------------------------------------------------------------------

def _dijkstra(adj, source, target, excl_edges=None, excl_nodes=None):
    """Standard Dijkstra. Returns (node_path, total_dist) or None."""
    excl_e = excl_edges or set()
    excl_n = excl_nodes or set()

    if source in excl_n or target in excl_n:
        return None

    dist = {source: 0.0}
    prev = {source: None}
    visited = set()
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            path = []
            n = target
            while n is not None:
                path.append(n)
                n = prev[n]
            return path[::-1], d
        for nb, eid, w in adj[u]:
            if nb in excl_n or eid in excl_e or nb in visited:
                continue
            nd = d + w
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                prev[nb] = u
                heapq.heappush(heap, (nd, nb))
    return None


def _path_edges(edge_by_nodes, path):
    return [
        _edge_between(edge_by_nodes, path[i], path[i + 1])
        for i in range(len(path) - 1)
    ]


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def yen_k_shortest(adj, edge_by_nodes, edges, source, target,
                   K=3, min_diversity=0.3):
    """Yen's algorithm for K shortest loopless paths with diversity filter.

    Finds up to K+4 candidates, then picks the K most edge-diverse paths so
    the returned routes are meaningfully different (Jaccard < 1-min_diversity).
    """
    first = _dijkstra(adj, source, target)
    if not first:
        return []

    A = [first]   # accepted paths: list of (node_path, cost)
    B = []        # candidate heap: (cost, counter, node_path)
    ctr = 0

    for k in range(1, K + 4):
        if k > len(A):
            break
        prev_path = A[k - 1][0]

        for i in range(len(prev_path) - 1):
            root = prev_path[: i + 1]
            root_cost = 0.0
            for j in range(len(root) - 1):
                e = _edge_between(edge_by_nodes, root[j], root[j + 1])
                if e is not None:
                    root_cost += edges[e]["length"]

            # Exclude edges already used at this spur point by accepted paths
            excl_e = set()
            for p, _ in A:
                if len(p) > i and p[: i + 1] == root:
                    e = _edge_between(edge_by_nodes, p[i], p[i + 1])
                    if e is not None:
                        excl_e.add(e)

            excl_n = set(root[:-1])  # avoid loops through root nodes

            spur = _dijkstra(adj, root[-1], target, excl_e, excl_n)
            if not spur:
                continue

            full = root[:-1] + spur[0]
            cost = root_cost + spur[1]
            ft = tuple(full)

            exist_A = {tuple(p) for p, _ in A}
            exist_B = {tuple(p) for _, _, p in B}
            if ft not in exist_A and ft not in exist_B:
                heapq.heappush(B, (cost, ctr, full))
                ctr += 1

        if not B:
            break
        c, _, p = heapq.heappop(B)
        A.append((p, c))

    # -- Diversity filter: keep K paths with sufficient edge differences -----
    diverse = [A[0]]
    for path, cost in A[1:]:
        if len(diverse) >= K:
            break
        pe = _path_edges(edge_by_nodes, path)
        if all(
            _jaccard(pe, _path_edges(edge_by_nodes, sp)) < (1 - min_diversity)
            for sp, _ in diverse
        ):
            diverse.append((path, cost))

    # Back-fill if diversity filtering was too strict
    for path, cost in A[1:]:
        if len(diverse) >= K:
            break
        if not any(tuple(path) == tuple(p) for p, _ in diverse):
            diverse.append((path, cost))

    return diverse[:K]


# ---------------------------------------------------------------------------
# 3. Forecast lookup
# ---------------------------------------------------------------------------

def _load_forecasts(forecast_path):
    """Load forecasts.json → dict[(lat_int, lon_int)] → sorted [(dt, cong)]."""
    lookup = defaultdict(list)
    p = Path(forecast_path)
    if not p.exists():
        return lookup

    with open(p) as f:
        raw = json.load(f)

    for e in raw.get("outputs", raw.get("predictions", [])):
        lat = int(e.get("latitude", 0))
        lon = int(e.get("longitude", 0))
        dt_str = e.get("DateTime", e.get("datetime", ""))
        cong = float(
            e.get("predicted_congestion_level",
                   e.get("predicted_congestion", 0))
        )
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                dt = datetime.fromisoformat(dt_str)
            except ValueError:
                continue
        lookup[(lat, lon)].append((dt, cong))

    for k in lookup:
        lookup[k].sort(key=lambda x: x[0])
    return lookup


def _forecast_congestion(lookup, lat, lon, ts):
    """Get predicted congestion for an edge's lat/lon at timestamp *ts*."""
    key = (int(round(lat * 1e6)), int(round(lon * 1e6)))
    entries = lookup.get(key, [])
    if not entries:
        return 0.0
    return min(entries, key=lambda x: abs((x[0] - ts).total_seconds()))[1]


# ---------------------------------------------------------------------------
# 4. Assignment ledger (persistent JSON for load-balancing across calls)
# ---------------------------------------------------------------------------

def _load_ledger(path):
    p = Path(path)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass
    return {"assignments": []}


def _save_ledger(data, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


def _cleanup_ledger(data, ref_time, decay_minutes=60):
    cutoff = ref_time - timedelta(minutes=decay_minutes)
    data["assignments"] = [
        a for a in data["assignments"]
        if datetime.fromisoformat(a["recorded_at"]) > cutoff
    ]


def _edge_load(data, eid):
    """Count how many recent assignments include this edge."""
    return sum(1 for a in data["assignments"] if eid in a["edges"])


def _record_assignment(data, edge_ids, ts, path):
    data["assignments"].append({
        "edges": edge_ids,
        "timestamp": ts.isoformat(),
        "recorded_at": datetime.now().isoformat(),
    })
    _save_ledger(data, path)


# ---------------------------------------------------------------------------
# 5. Route cost scoring
# ---------------------------------------------------------------------------

CONGESTION_SCALE = 100.0   # weight for predicted congestion
LOAD_PENALTY     = 50.0    # penalty per concurrent user on an edge


def _score_route(path_nodes, edge_by_nodes, edges, edge_traffic, time_idx,
                 forecast_lookup, ledger_data, timestamp):
    """Lower score = better route."""
    total = 0.0
    for i in range(len(path_nodes) - 1):
        eid = _edge_between(edge_by_nodes, path_nodes[i], path_nodes[i + 1])
        if eid is None:
            continue
        info = edges[eid]
        traffic = edge_traffic[eid].get(time_idx, {})

        vc = float(traffic.get("vehicle_count", 0))
        fc = _forecast_congestion(
            forecast_lookup, info["lat"], info["lon"], timestamp
        )
        load = _edge_load(ledger_data, eid)

        total += vc + fc * CONGESTION_SCALE + LOAD_PENALTY * load
    return total


# ---------------------------------------------------------------------------
# 6. Generate modified CSV
# ---------------------------------------------------------------------------

def _update_neighbor_features(modified, adj, edges, edge_traffic, time_idx):
    """Recompute spatial neighbour features for the modified rows."""
    idx_map = {int(row["edge_id"]): idx for idx, row in modified.iterrows()}

    for idx, row in modified.iterrows():
        eid = int(row["edge_id"])
        info = edges[eid]

        nbr_eids = _neighbor_edges(adj, edges, eid)
        congs, speeds = [], []
        for neid in nbr_eids:
            if neid in idx_map:
                congs.append(float(modified.at[idx_map[neid], "congestion_level"]))
                speeds.append(float(modified.at[idx_map[neid], "average_speed_kph"]))
            elif neid in edge_traffic and time_idx in edge_traffic[neid]:
                t = edge_traffic[neid][time_idx]
                congs.append(float(t["congestion_level"]))
                speeds.append(float(t["average_speed_kph"]))

        if congs:
            modified.at[idx, "neighbor_avg_congestion_t-1"] = round(
                np.mean(congs), 4
            )
        if speeds:
            modified.at[idx, "neighbor_avg_speed_t-1"] = round(
                np.mean(speeds), 2
            )

        # Upstream: edges whose end_node == this edge's start_node
        up = []
        for _, neid, _ in adj[info["start_node"]]:
            if neid != eid and edges[neid]["end_node"] == info["start_node"]:
                if neid in idx_map:
                    up.append(float(modified.at[idx_map[neid], "congestion_level"]))
                elif neid in edge_traffic and time_idx in edge_traffic[neid]:
                    up.append(float(edge_traffic[neid][time_idx]["congestion_level"]))
        if up:
            modified.at[idx, "upstream_congestion_t-1"] = round(np.mean(up), 4)

        # Downstream: edges whose start_node == this edge's end_node
        dn = []
        for _, neid, _ in adj[info["end_node"]]:
            if neid != eid and edges[neid]["start_node"] == info["end_node"]:
                if neid in idx_map:
                    dn.append(float(modified.at[idx_map[neid], "congestion_level"]))
                elif neid in edge_traffic and time_idx in edge_traffic[neid]:
                    dn.append(float(edge_traffic[neid][time_idx]["congestion_level"]))
        if dn:
            modified.at[idx, "downstream_congestion_t-1"] = round(
                np.mean(dn), 4
            )


def _generate_modified_csv(df, adj, edges, edge_by_nodes, edge_traffic,
                           chosen_edges, shortest_edges, time_idx,
                           output_path):
    """Write a CSV of only the modified edge-rows for re-prediction."""
    chosen_set   = set(chosen_edges)
    shortest_set = set(shortest_edges)
    diverted_from = shortest_set - chosen_set   # user was sent away from these
    diverted_to   = chosen_set - shortest_set   # user was sent here instead
    shared        = chosen_set & shortest_set
    all_affected  = chosen_set | shortest_set

    mask = (df["edge_id"].isin(all_affected)) & (df["time_idx"] == time_idx)
    mod = df.loc[mask].copy()

    if mod.empty:
        avail = df.loc[df["edge_id"].isin(all_affected), "time_idx"].unique()
        if len(avail):
            closest = int(min(avail, key=lambda t: abs(t - time_idx)))
            mask = (df["edge_id"].isin(all_affected)) & (df["time_idx"] == closest)
            mod = df.loc[mask].copy()

    for idx in mod.index:
        eid       = int(mod.at[idx, "edge_id"])
        free_flow = float(mod.at[idx, "free_flow_speed_kph"])
        vc        = int(mod.at[idx, "vehicle_count"])
        speed     = float(mod.at[idx, "average_speed_kph"])
        length    = float(mod.at[idx, "road_length_meters"])

        if eid in diverted_from:
            # Traffic diverted away → fewer vehicles, speed recovers
            new_vc = max(0, vc - 1)
            recovery = (free_flow - speed) / max(vc, 1)
            new_speed = min(free_flow, speed + recovery)

        elif eid in diverted_to:
            # User routed here → one extra vehicle, small speed decrease
            new_vc = vc + 1
            impact = (speed - free_flow * 0.5) * 0.02 / max(vc + 1, 1)
            new_speed = max(free_flow * 0.3, speed - impact)

        else:
            # Shared edge: user passes through regardless, +1 vehicle
            new_vc = vc + 1
            new_speed = speed  # negligible change on shared segments

        mod.at[idx, "vehicle_count"]     = new_vc
        mod.at[idx, "average_speed_kph"] = round(new_speed, 2)
        mod.at[idx, "congestion_level"]  = round(max(0.0, 1 - new_speed / free_flow), 4)
        if new_speed > 0:
            mod.at[idx, "travel_time_seconds"] = round(
                length / (new_speed / 3.6), 2
            )

    # Recompute spatial features for consistency
    _update_neighbor_features(mod, adj, edges, edge_traffic, time_idx)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mod.to_csv(output_path, index=False)
    return mod


# ---------------------------------------------------------------------------
# 7. Main entry point
# ---------------------------------------------------------------------------

def optimize_route(start_node, end_node, timestamp_str,
                   csv_path=None, forecast_path=None,
                   output_path=None, ledger_path=None):
    """Run the full route-optimization pipeline for a single request.

    Parameters
    ----------
    start_node : int
    end_node : int
    timestamp_str : str   ISO-format, e.g. "2026-02-08T03:00:00"
    csv_path, forecast_path, output_path, ledger_path : str | None
        Override default file locations (auto-detected from workspace root).

    Returns
    -------
    dict with keys:
        chosen_route, chosen_edges, total_cost, total_distance,
        all_routes, output_csv, affected_edges_count
    """
    ws = _workspace_root()
    csv_path      = csv_path      or _first_csv(ws / "backend" / "uploads")
    forecast_path = forecast_path or str(ws / "backend" / "outputs" / "forecasts.json")
    output_path   = output_path   or str(ws / "backend" / "outputs" / "optimized_edges.csv")
    ledger_path   = ledger_path   or str(ws / "backend" / "outputs" / "route_assignments.json")

    if csv_path is None:
        raise FileNotFoundError("No CSV found in backend/uploads/")

    # ── 1. Build graph from CSV ──────────────────────────────────────────
    print(f"[1/6] Building graph from {Path(csv_path).name} …")
    adj, edges, edge_by_nodes, edge_traffic, time_indices, df = build_graph(csv_path)
    print(f"       {len(edges)} edges, {len(set().union(*([set()] + [set(n for n,_,_ in adj[k]) for k in adj])))} nodes")

    # Validate nodes
    all_nodes = set()
    for eid, info in edges.items():
        all_nodes.add(info["start_node"])
        all_nodes.add(info["end_node"])
    if start_node not in all_nodes:
        raise ValueError(f"start_node {start_node} not in graph (valid: 0–{max(all_nodes)})")
    if end_node not in all_nodes:
        raise ValueError(f"end_node {end_node} not in graph (valid: 0–{max(all_nodes)})")

    # ── 2. Parse timestamp → time_idx ────────────────────────────────────
    timestamp = datetime.fromisoformat(timestamp_str)
    time_idx = _find_time_idx(edge_traffic, edges, time_indices, timestamp)
    print(f"[2/6] Timestamp {timestamp_str} → time_idx {time_idx}")

    # ── 3. K-shortest paths ──────────────────────────────────────────────
    print(f"[3/6] Finding K=3 shortest paths  {start_node} → {end_node} …")
    paths = yen_k_shortest(adj, edge_by_nodes, edges, start_node, end_node, K=3)
    if not paths:
        raise ValueError(f"No path exists between {start_node} and {end_node}")
    for i, (p, c) in enumerate(paths):
        print(f"       Route {i+1}: {len(p)} nodes, {c:.1f} m,  edges={_path_edges(edge_by_nodes, p)}")

    # ── 4. Load forecasts + ledger ───────────────────────────────────────
    print(f"[4/6] Loading forecasts & assignment ledger …")
    fc_lookup = _load_forecasts(forecast_path)
    ledger = _load_ledger(ledger_path)
    _cleanup_ledger(ledger, timestamp)

    # ── 5. Score & choose best route ─────────────────────────────────────
    scored = []
    for path_nodes, base_dist in paths:
        cost = _score_route(
            path_nodes, edge_by_nodes, edges, edge_traffic, time_idx,
            fc_lookup, ledger, timestamp,
        )
        scored.append((cost, path_nodes, base_dist))
    scored.sort(key=lambda x: x[0])

    best_cost, best_path, best_dist = scored[0]
    best_edges = _path_edges(edge_by_nodes, best_path)
    shortest_edges = _path_edges(edge_by_nodes, paths[0][0])

    print(f"[5/6] Best route: cost={best_cost:.2f}  dist={best_dist:.1f} m  "
          f"edges={best_edges}")

    # Record in ledger
    _record_assignment(ledger, best_edges, timestamp, ledger_path)

    # ── 6. Generate modified CSV ─────────────────────────────────────────
    print(f"[6/6] Writing modified CSV → {output_path}")
    mod_df = _generate_modified_csv(
        df, adj, edges, edge_by_nodes, edge_traffic,
        best_edges, shortest_edges, time_idx, output_path,
    )
    print(f"       {len(mod_df)} rows written ({len(set(best_edges) | set(shortest_edges))} edges)")

    return {
        "chosen_route": best_path,
        "chosen_edges": best_edges,
        "total_cost": best_cost,
        "total_distance_m": best_dist,
        "all_routes": [
            {"route": p, "edges": _path_edges(edge_by_nodes, p),
             "cost": c, "distance_m": d}
            for c, p, d in scored
        ],
        "output_csv": output_path,
        "affected_edges_count": len(mod_df),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Route optimizer with load-balancing"
    )
    parser.add_argument("--start", type=int, required=True,
                        help="Start node ID")
    parser.add_argument("--end", type=int, required=True,
                        help="End node ID")
    parser.add_argument("--timestamp", type=str, required=True,
                        help='ISO timestamp, e.g. "2026-02-08T03:00:00"')
    args = parser.parse_args()

    result = optimize_route(args.start, args.end, args.timestamp)
    print("\n" + json.dumps(result, indent=2))
