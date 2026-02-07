import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
csv_path = '/Users/ark/.cache/kagglehub/datasets/ziya07/smart-mobility-traffic-dataset/versions/1/smart_mobility_dataset.csv'
df = pd.read_csv(csv_path)

# Show basic info and first few rows
df.info()
print(df.head())

# Plot some key columns
df.plot(x='Timestamp', y=['Vehicle_Count', 'Traffic_Speed_kmh', 'Emission_Levels_g_km'], subplots=True, figsize=(12, 8))
plt.tight_layout()
plt.show()