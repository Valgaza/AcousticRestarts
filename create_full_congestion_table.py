import h5py
import pandas as pd
import numpy as np

# Load sensor locations
sensor_url = 'https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/graph_sensor_locations.csv'
sensors = pd.read_csv(sensor_url)
sensors['sensor_id'] = sensors['sensor_id'].astype(str)

# Load congestion percentages that were already calculated
congestion_data = pd.read_csv('metrla_congestion_per_sensor.csv')
congestion_data['sensor_id'] = congestion_data['sensor_id'].astype(str)

# Load human-readable time index (just first 100 time steps for demo)
time_index = pd.read_csv('metrla_time_index_human.csv').head(100)

# Merge congestion with location data
merged = pd.merge(congestion_data, sensors, on='sensor_id', how='left')

# Create full table by cross-joining sensors with time steps
full_rows = []
for _, time_row in time_index.iterrows():
    for _, sensor_row in merged.iterrows():
        full_rows.append({
            'sensor_id': sensor_row['sensor_id'],
            'latitude': sensor_row['latitude'],
            'longitude': sensor_row['longitude'],
            'datetime': time_row['datetime'],
            'congestion_pct': sensor_row['congestion_pct']
        })

# Save full table
full_df = pd.DataFrame(full_rows)
full_df.to_csv('metrla_full_congestion_table.csv', index=False)
print(f'Full table saved with {len(full_df)} rows (first 100 time steps)')
print(full_df.head())
