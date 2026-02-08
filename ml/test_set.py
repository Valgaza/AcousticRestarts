"""
Extract test set data and save as CSV.

This script extracts the test portion of the traffic data (time_idx 429-503)
and saves it to the backend/uploads directory.
"""

import pandas as pd
from pathlib import Path


def extract_test_set():
    """Extract test set data and save as CSV."""
    
    # Configuration
    SOURCE_CSV = 'data/traffic_synthetic.csv'
    OUTPUT_DIR = r'C:\Users\Arnav Waghdhare\Desktop\Arnav20\Coding\Python\AcousticRestarts\backend\uploads'
    OUTPUT_FILE = 'traffic_test_set.csv'
    
    # Test set time indices
    TEST_TIME_MIN = 429
    TEST_TIME_MAX = 503
    
    print(f"Loading data from: {SOURCE_CSV}")
    
    # Load full dataset
    df = pd.read_csv(SOURCE_CSV)
    print(f"Total records: {len(df)}")
    print(f"Time index range: {df['time_idx'].min()} to {df['time_idx'].max()}")
    
    # Filter for test set
    test_df = df[(df['time_idx'] >= TEST_TIME_MIN) & (df['time_idx'] <= TEST_TIME_MAX)]
    print(f"\nTest set records: {len(test_df)}")
    print(f"Test set time index range: {test_df['time_idx'].min()} to {test_df['time_idx'].max()}")
    print(f"Unique edges in test set: {test_df['edge_id'].nunique()}")
    print(f"Time steps in test set: {test_df['time_idx'].nunique()}")
    
    # Create output directory if it doesn't exist
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = output_path / OUTPUT_FILE
    test_df.to_csv(output_file, index=False)
    
    print(f"\nTest set saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
    
    # Display sample data
    print("\nSample of test set (first 5 rows):")
    print(test_df.head())
    
    return test_df


if __name__ == "__main__":
    extract_test_set()
