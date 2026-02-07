import h5py
import numpy as np
import pandas as pd

# Path to METR-LA dataset
h5_path = '/Users/ark/.cache/kagglehub/datasets/annnnguyen/metr-la-dataset/versions/4/METR-LA.h5'

# Threshold for congestion (mph)
CONGESTION_THRESHOLD = 20

with h5py.File(h5_path, 'r') as f:
    df_group = f['df']
    speeds = df_group['block0_values'][:]
    # Calculate congestion mask (time, sensors)
    congestion = speeds < CONGESTION_THRESHOLD
    # Congestion percentage for each sensor (node)
    congestion_pct_per_sensor = np.sum(congestion, axis=0) / congestion.shape[0] * 100
    # Get sensor IDs
    sensor_ids = [x.decode() for x in df_group['block0_items'][:]]
    # Create DataFrame for easy viewing
    df = pd.DataFrame({'sensor_id': sensor_ids, 'congestion_pct': congestion_pct_per_sensor})
    df = df.sort_values('congestion_pct', ascending=False)
    df.to_csv('metrla_congestion_per_sensor.csv', index=False)
    print(df.head(10))
    print('Full results saved as metrla_congestion_per_sensor.csv')
