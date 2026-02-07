import h5py
import pandas as pd

# Path to METR-LA dataset
h5_path = '/Users/ark/.cache/kagglehub/datasets/annnnguyen/metr-la-dataset/versions/4/METR-LA.h5'

with h5py.File(h5_path, 'r') as f:
    df_group = f['df']
    # axis1 is the time index
    axis1 = df_group['axis1'][:]
    print('axis1 shape:', axis1.shape)
    print('First 10 axis1 values:', axis1[:10])
    print('Last 10 axis1 values:', axis1[-10:])
    print('dtype:', axis1.dtype)
    # Save to CSV for inspection
    pd.Series(axis1).to_csv('metrla_time_index.csv', index=False)
    print('Time index saved as metrla_time_index.csv')
