import h5py
import numpy as np
import pandas as pd

# Path to METR-LA dataset
h5_path = '/Users/ark/.cache/kagglehub/datasets/annnnguyen/metr-la-dataset/versions/4/METR-LA.h5'

# Load sensor locations
sensor_url = 'https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/graph_sensor_locations.csv'
sensors = pd.read_csv(sensor_url)
sensors['sensor_id'] = sensors['sensor_id'].astype(str)

# Load time index 
time_index = pd.read_csv('metrla_time_index_human.csv')

print("Loading METR-LA speed data...")
with h5py.File(h5_path, 'r') as f:
    df_group = f['df']
    speeds = df_group['block0_values'][:]  # Shape: (time_steps, sensors)
    sensor_ids = [x.decode() for x in df_group['block0_items'][:]]

print(f"Speed data shape: {speeds.shape}")
print(f"Time steps: {speeds.shape[0]}, Sensors: {speeds.shape[1]}")

# Calculate FREE FLOW SPEED for each sensor (95th percentile of all speeds)
# This represents normal, uncongested speed for each location
free_flow_speeds = np.nanpercentile(speeds, 95, axis=0)

print("Calculating real-time congestion for each timestamp...")

# Calculate REAL-TIME CONGESTION for each time step and sensor
# Congestion = (free_flow_speed - current_speed) / free_flow_speed * 100
# If congestion < 0, set to 0 (no congestion, actually faster than normal)

realtime_data = []
num_time_steps = min(10, speeds.shape[0])  # First 10 time steps for demo

for t in range(num_time_steps):
    current_speeds = speeds[t, :]
    
    for s, sensor_id in enumerate(sensor_ids):
        if not np.isnan(current_speeds[s]) and not np.isnan(free_flow_speeds[s]):
            # Calculate congestion percentage
            if free_flow_speeds[s] > 0:
                congestion_pct = ((free_flow_speeds[s] - current_speeds[s]) / free_flow_speeds[s]) * 100
                congestion_pct = max(0, congestion_pct)  # No negative congestion
                
                # Get sensor coordinates
                sensor_info = sensors[sensors['sensor_id'] == sensor_id]
                if not sensor_info.empty:
                    lat = sensor_info.iloc[0]['latitude']
                    lon = sensor_info.iloc[0]['longitude']
                    
                    realtime_data.append({
                        'sensor_id': sensor_id,
                        'latitude': lat,
                        'longitude': lon,
                        'datetime': time_index.iloc[t]['datetime'],
                        'current_speed_mph': round(current_speeds[s], 2),
                        'free_flow_speed_mph': round(free_flow_speeds[s], 2),
                        'congestion_pct': round(congestion_pct, 2)
                    })

# Create DataFrame
df = pd.DataFrame(realtime_data)
df = df.sort_values(['datetime', 'congestion_pct'], ascending=[True, False])

print(f"\n✅ SUCCESS: Generated real-time congestion data")
print(f"✅ Total records: {len(df)}")
print(f"✅ Time range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"✅ Congestion range: {df['congestion_pct'].min()}% to {df['congestion_pct'].max()}%")

# Show sample data
print(f"\n📊 Sample of REAL-TIME congestion data:")
print(df[['sensor_id', 'datetime', 'current_speed_mph', 'free_flow_speed_mph', 'congestion_pct']].head(10).to_string(index=False))

# Save results
df.to_csv('metrla_realtime_congestion.csv', index=False)
print(f"\n💾 Saved as: metrla_realtime_congestion.csv")

# Show some specific examples
print(f"\n🚗 Real-time congestion examples:")
for timestamp in df['datetime'].unique()[:2]:
    print(f"\nAt {timestamp}:")
    sample = df[df['datetime'] == timestamp].head(5)
    for _, row in sample.iterrows():
        print(f"  Sensor {row['sensor_id']}: {row['current_speed_mph']} mph (normal: {row['free_flow_speed_mph']} mph) → {row['congestion_pct']}% congestion")