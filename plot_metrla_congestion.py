import h5py
import numpy as np
import matplotlib.pyplot as plt

# Path to METR-LA dataset
h5_path = '/Users/ark/.cache/kagglehub/datasets/annnnguyen/metr-la-dataset/versions/4/METR-LA.h5'

# Threshold for congestion (mph)
CONGESTION_THRESHOLD = 20

with h5py.File(h5_path, 'r') as f:
    df_group = f['df']
    # Extract the main data matrix (time, sensors)
    speeds = df_group['block0_values'][:]
    # Shape: (time_steps, num_sensors)
    print('Data shape:', speeds.shape)
    # Calculate congestion mask
    congestion = speeds < CONGESTION_THRESHOLD
    # Percentage of sensors congested at each time step
    congestion_pct = np.sum(congestion, axis=1) / congestion.shape[1] * 100
    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(congestion_pct)
    plt.xlabel('Time step')
    plt.ylabel('% Sensors Congested (<20 mph)')
    plt.title('Network-wide Congestion Over Time (METR-LA)')
    plt.tight_layout()
    plt.savefig('metrla_congestion_over_time.png')
    plt.show()
    print('Plot saved as metrla_congestion_over_time.png')
