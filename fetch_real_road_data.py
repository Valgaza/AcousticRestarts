import requests
import pandas as pd
import h5py
import numpy as np
import time
from geopy.distance import geodesic

def get_osm_road_data(lat, lon, radius=50):
    """Get road data from OpenStreetMap using Overpass API"""
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    query = f"""
    [out:json][timeout:25];
    (
      way["highway"]["maxspeed"]["lanes"](around:{radius},{lat},{lon});
      way["highway"]["maxspeed"](around:{radius},{lat},{lon});
      way["highway"]["lanes"](around:{radius},{lat},{lon});
    );
    out geom;
    """
    
    try:
        response = requests.get(overpass_url, params={'data': query}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'elements' in data and len(data['elements']) > 0:
                # Find closest road with the most complete data
                best_road = None
                best_score = 0
                sensor_point = (lat, lon)
                
                for element in data['elements']:
                    tags = element.get('tags', {})
                    score = 0
                    
                    # Calculate distance to road
                    if 'geometry' in element:
                        min_dist = float('inf')
                        for coord in element['geometry']:
                            road_point = (coord['lat'], coord['lon'])
                            dist = geodesic(sensor_point, road_point).meters
                            min_dist = min(min_dist, dist)
                        
                        if min_dist > radius:
                            continue
                            
                        # Score based on data completeness
                        if 'maxspeed' in tags:
                            score += 2
                        if 'lanes' in tags:
                            score += 2
                        if 'highway' in tags:
                            score += 1
                            
                        # Prefer closer roads
                        score += max(0, (radius - min_dist) / radius)
                        
                        if score > best_score:
                            best_score = score
                            best_road = tags
                
                if best_road:
                    speed_limit = best_road.get('maxspeed', None)
                    lanes = best_road.get('lanes', None)
                    highway_type = best_road.get('highway', None)
                    
                    # Parse speed limit (handle "30 mph", "50", etc.)
                    if speed_limit:
                        try:
                            speed_limit = int(''.join(filter(str.isdigit, str(speed_limit))))
                        except:
                            speed_limit = None
                    
                    # Parse lanes
                    if lanes:
                        try:
                            lanes = int(lanes)
                        except:
                            lanes = None
                    
                    return speed_limit, lanes, highway_type
        
    except Exception as e:
        print(f"Error querying OSM for {lat}, {lon}: {e}")
    
    return None, None, None

# Load sensor locations
sensor_url = 'https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/graph_sensor_locations.csv'
sensors = pd.read_csv(sensor_url)
sensors['sensor_id'] = sensors['sensor_id'].astype(str)

# Load time index (first 10 time steps for real data verification)
time_index = pd.read_csv('metrla_time_index_human.csv').head(10)

# Load actual speed data for congestion calculation
h5_path = '/Users/ark/.cache/kagglehub/datasets/annnnguyen/metr-la-dataset/versions/4/METR-LA.h5'
CONGESTION_THRESHOLD = 20

with h5py.File(h5_path, 'r') as f:
    speeds = f['df']['block0_values'][:10, :]  # First 10 time steps
    sensor_ids = [x.decode() for x in f['df']['block0_items'][:]]

print(f"Fetching real road data for {len(sensors)} sensors...")
print("This may take several minutes due to API rate limits...")

# Get real road data for each sensor (limit to first 20 sensors for demo)
real_data_rows = []
successful_sensors = 0

for idx, (_, sensor_row) in enumerate(sensors.head(20).iterrows()):
    sensor_id = sensor_row['sensor_id']
    lat = sensor_row['latitude']  
    lon = sensor_row['longitude']
    
    print(f"Processing sensor {idx+1}/20: {sensor_id}")
    
    # Get real road data from OSM
    speed_limit, lanes, highway_type = get_osm_road_data(lat, lon)
    
    # Only include if we got real road data
    if speed_limit is not None and lanes is not None:
        successful_sensors += 1
        
        # Find sensor index in speed data
        try:
            s_idx = sensor_ids.index(sensor_id)
            
            # Add rows for each time step with real congestion data
            for t_idx in range(speeds.shape[0]):
                current_speed = speeds[t_idx, s_idx]
                
                # Skip if speed data is NaN
                if not np.isnan(current_speed):
                    # Calculate actual congestion percentage at this moment
                    is_congested = current_speed < CONGESTION_THRESHOLD
                    congestion_pct = 100.0 if is_congested else 0.0
                    
                    datetime_str = time_index.iloc[t_idx]['datetime']
                    
                    real_data_rows.append({
                        'sensor_id': sensor_id,
                        'latitude': lat,
                        'longitude': lon,
                        'datetime': datetime_str,
                        'speed_limit_mph': speed_limit,
                        'congestion_pct': congestion_pct,
                        'number_of_lanes': lanes
                    })
        except ValueError:
            print(f"  Warning: Sensor {sensor_id} not found in speed data")
    else:
        print(f"  No real road data found for sensor {sensor_id}")
    
    # Rate limiting - be respectful to OSM servers
    time.sleep(1)

# Create final verified dataset
if real_data_rows:
    real_df = pd.DataFrame(real_data_rows)
    real_df.to_csv('metrla_verified_real_traffic_data.csv', index=False)
    
    print(f"\n✅ SUCCESS: Created verified dataset with {len(real_data_rows)} rows")
    print(f"✅ Real road data found for {successful_sensors} sensors")
    print(f"✅ File saved as: metrla_verified_real_traffic_data.csv")
    print("\nSample of REAL data:")
    print(real_df.head())
    print(f"\nSpeed limits range: {real_df['speed_limit_mph'].min()}-{real_df['speed_limit_mph'].max()} mph")
    print(f"Lane counts range: {real_df['number_of_lanes'].min()}-{real_df['number_of_lanes'].max()} lanes")
else:
    print("❌ No real road data could be retrieved. Check network connection and try again.")