import pandas as pd
import folium

# Load dataset
csv_path = '/Users/ark/.cache/kagglehub/datasets/ziya07/smart-mobility-traffic-dataset/versions/1/smart_mobility_dataset.csv'
df = pd.read_csv(csv_path)

# Get center of map
center_lat = df['Latitude'].mean()
center_lon = df['Longitude'].mean()

# Create map
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Add every point
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,
        color='blue',
        fill=True,
        fill_opacity=0.6,
        popup=f"Vehicle Count: {row['Vehicle_Count']}\nSpeed: {row['Traffic_Speed_kmh']}\nCondition: {row['Traffic_Condition']}"
    ).add_to(m)

# Save map
m.save('traffic_map_all.html')
print('Map saved as traffic_map_all.html')