import h5py
import pandas as pd
import numpy as np

# Load sensor locations
sensor_url = 'https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/graph_sensor_locations.csv'
sensors = pd.read_csv(sensor_url)
sensors['sensor_id'] = sensors['sensor_id'].astype(str)

# Load human-readable time index (sample first 50 time steps for performance)
time_index = pd.read_csv('metrla_time_index_human.csv').head(50)

# Load traffic speed data for congestion calculation at each time step
h5_path = '/Users/ark/.cache/kagglehub/datasets/annnnguyen/metr-la-dataset/versions/4/METR-LA.h5'
CONGESTION_THRESHOLD = 20

with h5py.File(h5_path, 'r') as f:
    speeds = f['df']['block0_values'][:50, :]  # First 50 time steps
    sensor_ids = [x.decode() for x in f['df']['block0_items'][:]]

# Create final comprehensive table
final_rows = []
for t_idx in range(speeds.shape[0]):
    datetime_str = time_index.iloc[t_idx]['datetime']
    for s_idx, sensor_id in enumerate(sensor_ids):
        # Get current speed and calculate if congested
        current_speed = speeds[t_idx, s_idx]
        is_congested = current_speed < CONGESTION_THRESHOLD
        congestion_pct = 100.0 if is_congested else 0.0
        
        # Get sensor location
        sensor_row = sensors[sensors['sensor_id'] == sensor_id]
        if not sensor_row.empty:
            lat = sensor_row.iloc[0]['latitude']
            lon = sensor_row.iloc[0]['longitude']
        else:
            lat = np.nan
            lon = np.nan
        
        # Add road infrastructure data (placeholders - would need external data source)
        # These would typically come from OpenStreetMap, PeMS, or local DOT databases
        speed_limit = "Unknown"  # Would need mapping API or OSM query
        lanes = "Unknown"        # Would need mapping API or OSM query
        
        final_rows.append({
            'sensor_id': sensor_id,
            'latitude': lat,
            'longitude': lon,
            'datetime': datetime_str,
            'congestion_percentage': congestion_pct,
            'speed_limit_mph': speed_limit,
            'number_of_lanes': lanes
        })

# Save comprehensive table
final_df = pd.DataFrame(final_rows)
final_df.to_csv('metrla_comprehensive_traffic_data.csv', index=False)

print(f'Comprehensive traffic data saved with {len(final_df)} rows')
print("\nSample data:")
print(final_df.head())
print(f"\nNote: Speed limit and lane data require external sources like OpenStreetMap or PeMS.")
print("These columns are currently marked as 'Unknown' and would need additional data collection.")