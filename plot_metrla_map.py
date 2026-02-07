import pandas as pd
import folium

# Download the sensor locations CSV from the official DCRNN repo
import pandas as pd
import folium

# Download the sensor locations CSV from the official DCRNN repo
sensor_csv_url = 'https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/graph_sensor_locations.csv'
sensors = pd.read_csv(sensor_csv_url)

# Plot all sensors on a map
center_lat = sensors['latitude'].mean()
center_lon = sensors['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

for _, row in sensors.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color='blue',
        fill=True,
        fill_opacity=0.7,
        popup=f"Sensor ID: {row['sensor_id']}"
    ).add_to(m)

m.save('metrla_sensor_map.html')
print('Map saved as metrla_sensor_map.html')
