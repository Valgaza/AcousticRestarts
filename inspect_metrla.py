import h5py
import numpy as np

# Path to METR-LA dataset
h5_path = '/Users/ark/.cache/kagglehub/datasets/annnnguyen/metr-la-dataset/versions/4/METR-LA.h5'

# Open HDF5 file and inspect the 'df' group
with h5py.File(h5_path, 'r') as f:
    df_group = f['df']
    print('df keys:', list(df_group.keys()))
    for key in df_group.keys():
        print(f"{key}: shape {df_group[key].shape}, dtype {df_group[key].dtype}")
    # Try to extract station coordinates if present
    if 'columns' in df_group:
        columns = [x.decode() for x in df_group['columns'][:]]
        print('Columns:', columns)
    if 'values' in df_group:
        print('Sample values:', df_group['values'][:5])
