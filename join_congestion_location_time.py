import pandas as pd

# Load congestion percentage per sensor
congestion = pd.read_csv('metrla_congestion_per_sensor.csv')

# Load sensor locations
sensor_url = 'https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/graph_sensor_locations.csv'
sensors = pd.read_csv(sensor_url)

# Merge on sensor_id (convert both to string for safety)
congestion['sensor_id'] = congestion['sensor_id'].astype(str)
sensors['sensor_id'] = sensors['sensor_id'].astype(str)

# Merge
merged = pd.merge(congestion, sensors, on='sensor_id', how='left')

# Load time index (human readable)
time_index = pd.read_csv('metrla_time_index_human.csv')

# Save merged data (one row per sensor, with congestion %, location, and time index as separate file)
merged.to_csv('metrla_congestion_location.csv', index=False)
# Save time index as is (already human readable)
# Optionally, you can join time index to each row if you want a full table (sensor x time), but that's a huge file

print(merged.head())
print('Merged congestion/location data saved as metrla_congestion_location.csv')
print('Time index (human) is in metrla_time_index_human.csv')
