"""
Example usage of the Traffic Congestion Forecasting Model

This script demonstrates how to:
1. Load and prepare data
2. Train the Prophet model
3. Generate forecasts
4. Evaluate model performance
5. Visualize predictions

Author: Arnav Waghdhare
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet_forecaster import TrafficCongestionForecaster, create_sample_data


def main():
    """
    Main function demonstrating the usage of TrafficCongestionForecaster.
    """
    print("=" * 70)
    print("Traffic Congestion Forecasting with Prophet")
    print("=" * 70)
    
    # ========================================================================
    # Step 1: Create or Load Data
    # ========================================================================
    print("\n[Step 1] Creating sample data...")
    
    # For demonstration, we'll create sample data
    # In practice, you would load your actual data here
    road_data, time_series_data = create_sample_data(n_samples=2000)
    
    print(f"Road data shape: {road_data.shape}")
    print(f"Time series data shape: {time_series_data.shape}")
    print("\nRoad data columns:", list(road_data.columns))
    print("Time series data columns:", list(time_series_data.columns))
    
    # Display sample rows
    print("\nSample road data:")
    print(road_data.head())
    print("\nSample time series data:")
    print(time_series_data.head())
    
    # ========================================================================
    # Step 2: Split Data into Train and Test Sets
    # ========================================================================
    print("\n[Step 2] Splitting data into train and test sets...")
    
    # Sort by timestamp
    time_series_data = time_series_data.sort_values('timestamp')
    
    # Split: 80% train, 20% test
    split_idx = int(len(time_series_data) * 0.8)
    train_data = time_series_data.iloc[:split_idx].copy()
    test_data = time_series_data.iloc[split_idx:].copy()
    
    print(f"Train data: {len(train_data)} samples")
    print(f"Test data: {len(test_data)} samples")
    print(f"Train date range: {train_data['timestamp'].min()} to {train_data['timestamp'].max()}")
    print(f"Test date range: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
    
    # ========================================================================
    # Step 3: Initialize and Train the Model
    # ========================================================================
    print("\n[Step 3] Initializing and training the Prophet model...")
    
    # Initialize the forecaster
    forecaster = TrafficCongestionForecaster(
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    
    # Fit the model
    print("Training model... This may take a few moments.")
    forecaster.fit(
        road_data=road_data,
        time_series_data=train_data,
        target_column='congestion_level',
        merge_on='edge_id'
    )
    
    print("✓ Model training complete!")
    print(f"Number of features used: {len(forecaster.feature_columns)}")
    print("Features:", forecaster.feature_columns)
    
    # ========================================================================
    # Step 4: Make Predictions on Test Data
    # ========================================================================
    print("\n[Step 4] Making predictions on test data...")
    
    # Prepare test data for prediction (need to merge with road data)
    test_data_with_features = test_data.merge(road_data, on='edge_id', how='left')
    
    # Generate predictions
    predictions = forecaster.predict(test_data_with_features)
    
    print(f"Generated {len(predictions)} predictions")
    print("\nSample predictions:")
    print(predictions.head(10))
    
    # ========================================================================
    # Step 5: Evaluate Model Performance
    # ========================================================================
    print("\n[Step 5] Evaluating model performance...")
    
    metrics = forecaster.evaluate(
        road_data=road_data,
        time_series_data=test_data,
        target_column='congestion_level',
        merge_on='edge_id'
    )
    
    print("\nModel Performance Metrics:")
    print(f"  MAE (Mean Absolute Error):       {metrics['MAE']:.4f}")
    print(f"  RMSE (Root Mean Squared Error):  {metrics['RMSE']:.4f}")
    print(f"  MAPE (Mean Absolute % Error):    {metrics['MAPE']:.2f}%")
    
    # ========================================================================
    # Step 6: Visualize Predictions vs Actual
    # ========================================================================
    print("\n[Step 6] Visualizing predictions vs actual values...")
    
    # Merge predictions with actual values for visualization
    test_results = predictions.merge(
        test_data[['timestamp', 'congestion_level', 'edge_id']],
        on='timestamp',
        how='inner'
    )
    
    # Plot for each road (edge_id)
    unique_edges = test_results['edge_id'].unique()
    
    fig, axes = plt.subplots(len(unique_edges), 1, figsize=(14, 4 * len(unique_edges)))
    if len(unique_edges) == 1:
        axes = [axes]
    
    for idx, edge_id in enumerate(unique_edges):
        edge_data = test_results[test_results['edge_id'] == edge_id].sort_values('timestamp')
        
        ax = axes[idx]
        ax.plot(edge_data['timestamp'], edge_data['congestion_level'], 
                label='Actual', color='blue', linewidth=2)
        ax.plot(edge_data['timestamp'], edge_data['congestion_level_predicted'], 
                label='Predicted', color='red', linestyle='--', linewidth=2)
        ax.fill_between(edge_data['timestamp'],
                        edge_data['congestion_level_lower'],
                        edge_data['congestion_level_upper'],
                        alpha=0.3, color='red', label='Confidence Interval')
        
        ax.set_xlabel('Timestamp', fontsize=12)
        ax.set_ylabel('Congestion Level', fontsize=12)
        ax.set_title(f'Traffic Congestion Forecast - Road {edge_id}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('congestion_forecast.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'congestion_forecast.png'")

if __name__ == "__main__":
    main()
