import pandas as pd
from datetime import datetime

# Load the time index (Unix timestamps)
time_index = pd.read_csv('metrla_time_index.csv', header=None).iloc[:, 0]

# Convert to human-readable datetime (UTC)
# The timestamps appear to be in nanoseconds, convert to seconds
timestamps = time_index / 1e9
dates = pd.to_datetime(timestamps, unit='s')

# Save as CSV
out = pd.DataFrame({'timestamp': time_index, 'datetime': dates})
out.to_csv('metrla_time_index_human.csv', index=False)
print(out.head())
print('Human-readable time index saved as metrla_time_index_human.csv')
