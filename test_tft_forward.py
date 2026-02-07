"""
Test script for Temporal Fusion Transformer forward pass.

Generates dummy traffic dataset and validates model execution.
"""

import torch
from datetime import datetime, timedelta
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.temporal_transformer import (
    TemporalFusionTransformer,
    QuantileLoss,
    create_tft_model
)
from outputs import Output, OutputList


def generate_dummy_traffic_data(
    batch_size: int = 8,
    encoder_length: int = 48,
    prediction_length: int = 6,
    num_edges: int = 100,
    num_road_types: int = 10,
    num_nodes: int = 200,
    device: torch.device = torch.device('cpu')
) -> Dict[str, torch.Tensor]:
    """
    Generate dummy traffic data matching the dataset schema.
    
    Features:
    - Static categorical: edge_id, road_type, start_node_id, end_node_id
    - Static real: road_length_meters, lane_count, speed_limit_kph, 
                   free_flow_speed_kph, road_capacity, latitude_midroad, longitude_midroad
    - Time-varying known categorical: is_weekend, is_holiday, weather_condition
    - Time-varying known real: hour_of_day, day_of_week, visibility, event_impact_score
    - Time-varying unknown real: average_speed_kph, vehicle_count, travel_time_seconds, 
                                  congestion_level, neighbor_avg_congestion_t-1,
                                  neighbor_avg_speed_t-1, upstream_congestion_t-1, 
                                  downstream_congestion_t-1
    """
    total_length = encoder_length + prediction_length
    
    # ============== Static Categorical Features ==============
    static_categorical = {
        'edge_id': torch.randint(0, num_edges, (batch_size,), device=device),
        'road_type': torch.randint(0, num_road_types, (batch_size,), device=device),
        'start_node_id': torch.randint(0, num_nodes, (batch_size,), device=device),
        'end_node_id': torch.randint(0, num_nodes, (batch_size,), device=device)
    }
    
    # ============== Static Real Features ==============
    # Shape: (batch_size, 7)
    static_real = torch.cat([
        torch.rand(batch_size, 1, device=device) * 1000 + 100,     # road_length_meters (100-1100)
        torch.randint(1, 5, (batch_size, 1), device=device).float(),  # lane_count (1-4)
        torch.rand(batch_size, 1, device=device) * 80 + 30,        # speed_limit_kph (30-110)
        torch.rand(batch_size, 1, device=device) * 70 + 40,        # free_flow_speed_kph (40-110)
        torch.rand(batch_size, 1, device=device) * 2000 + 500,     # road_capacity (500-2500)
        torch.rand(batch_size, 1, device=device) * 0.1 + 40.7,     # latitude_midroad (~40.7-40.8)
        torch.rand(batch_size, 1, device=device) * 0.1 - 74.0      # longitude_midroad (~-74.0 to -73.9)
    ], dim=1)
    
    # ============== Time-varying Known Categorical (Encoder) ==============
    encoder_known_categorical = {
        'is_weekend': torch.randint(0, 2, (batch_size, encoder_length), device=device),
        'is_holiday': torch.randint(0, 2, (batch_size, encoder_length), device=device),
        'weather_condition': torch.randint(0, 10, (batch_size, encoder_length), device=device)
    }
    
    # ============== Time-varying Known Real (Encoder) ==============
    # Shape: (batch_size, encoder_length, 4)
    encoder_known_real = torch.cat([
        torch.randint(0, 24, (batch_size, encoder_length, 1), device=device).float(),  # hour_of_day
        torch.randint(0, 7, (batch_size, encoder_length, 1), device=device).float(),   # day_of_week
        torch.rand(batch_size, encoder_length, 1, device=device) * 10,                # visibility (0-10)
        torch.rand(batch_size, encoder_length, 1, device=device)                      # event_impact_score (0-1)
    ], dim=-1)
    
    # ============== Time-varying Unknown Real (Encoder) ==============
    # Includes target (congestion_level) and graph-derived features
    # Shape: (batch_size, encoder_length, 8)
    encoder_unknown_real = torch.cat([
        torch.rand(batch_size, encoder_length, 1, device=device) * 80 + 10,   # average_speed_kph
        torch.randint(0, 500, (batch_size, encoder_length, 1), device=device).float(),  # vehicle_count
        torch.rand(batch_size, encoder_length, 1, device=device) * 300 + 60,  # travel_time_seconds
        torch.rand(batch_size, encoder_length, 1, device=device),             # congestion_level (0-1)
        torch.rand(batch_size, encoder_length, 1, device=device),             # neighbor_avg_congestion_t-1
        torch.rand(batch_size, encoder_length, 1, device=device) * 80 + 10,   # neighbor_avg_speed_t-1
        torch.rand(batch_size, encoder_length, 1, device=device),             # upstream_congestion_t-1
        torch.rand(batch_size, encoder_length, 1, device=device)              # downstream_congestion_t-1
    ], dim=-1)
    
    # ============== Time-varying Known Categorical (Decoder) ==============
    decoder_known_categorical = {
        'is_weekend': torch.randint(0, 2, (batch_size, prediction_length), device=device),
        'is_holiday': torch.randint(0, 2, (batch_size, prediction_length), device=device),
        'weather_condition': torch.randint(0, 10, (batch_size, prediction_length), device=device)
    }
    
    # ============== Time-varying Known Real (Decoder) ==============
    # Shape: (batch_size, prediction_length, 4)
    decoder_known_real = torch.cat([
        torch.randint(0, 24, (batch_size, prediction_length, 1), device=device).float(),  # hour_of_day
        torch.randint(0, 7, (batch_size, prediction_length, 1), device=device).float(),   # day_of_week
        torch.rand(batch_size, prediction_length, 1, device=device) * 10,                # visibility
        torch.rand(batch_size, prediction_length, 1, device=device)                      # event_impact_score
    ], dim=-1)
    
    # ============== Target for loss computation ==============
    # Future congestion levels (ground truth)
    target = torch.rand(batch_size, prediction_length, device=device)
    
    # ============== Metadata for output formatting ==============
    # Generate timestamps starting from current time
    base_time = datetime.now()
    timestamps = []
    for i in range(prediction_length):
        timestamps.append((base_time + timedelta(hours=i+1)).isoformat())
    
    return {
        'static_categorical': static_categorical,
        'static_real': static_real,
        'encoder_known_categorical': encoder_known_categorical,
        'encoder_known_real': encoder_known_real,
        'encoder_unknown_real': encoder_unknown_real,
        'decoder_known_categorical': decoder_known_categorical,
        'decoder_known_real': decoder_known_real,
        'target': target,
        'timestamps': timestamps,
        'latitudes': static_real[:, 5],   # latitude_midroad
        'longitudes': static_real[:, 6]   # longitude_midroad
    }


def format_output(
    predictions: torch.Tensor,
    timestamps: List[str],
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    batch_idx: int = 0
) -> OutputList:
    """
    Format model predictions into OutputList format.
    
    Args:
        predictions: Quantile predictions (batch, pred_len, num_quantiles)
        timestamps: List of datetime strings
        latitudes: Tensor of latitudes (batch,)
        longitudes: Tensor of longitudes (batch,)
        batch_idx: Which batch sample to format
    
    Returns:
        OutputList with predictions for each horizon
    """
    outputs = []
    
    # Use median quantile (index 1) as point prediction
    point_predictions = predictions[batch_idx, :, 1]  # (pred_len,)
    
    for t in range(len(timestamps)):
        output = Output(
            DateTime=timestamps[t],
            latitude=int(latitudes[batch_idx].item() * 1e6),  # Convert to integer (microdegrees)
            longitude=int(longitudes[batch_idx].item() * 1e6),
            predicted_congestion_level=float(point_predictions[t].item())
        )
        outputs.append(output)
    
    return OutputList(outputs=outputs)


def main():
    """Run TFT forward pass test."""
    print("=" * 60)
    print("Temporal Fusion Transformer - Forward Pass Test")
    print("=" * 60)
    
    # Configuration
    batch_size = 8
    encoder_length = 48
    prediction_length = 6
    num_edges = 100
    num_road_types = 10
    num_nodes = 200
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # ============== Generate Dummy Data ==============
    print("\n[1] Generating dummy traffic dataset...")
    data = generate_dummy_traffic_data(
        batch_size=batch_size,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        num_edges=num_edges,
        num_road_types=num_road_types,
        num_nodes=num_nodes,
        device=device
    )
    
    print(f"    Static categorical features: {list(data['static_categorical'].keys())}")
    print(f"    Static real shape: {data['static_real'].shape}")
    print(f"    Encoder known real shape: {data['encoder_known_real'].shape}")
    print(f"    Encoder unknown real shape: {data['encoder_unknown_real'].shape}")
    print(f"    Decoder known real shape: {data['decoder_known_real'].shape}")
    
    # ============== Create Model ==============
    print("\n[2] Creating TFT model...")
    model = create_tft_model(
        num_edges=num_edges,
        num_road_types=num_road_types,
        num_nodes=num_nodes,
        hidden_dim=64,
        encoder_length=encoder_length,
        prediction_length=prediction_length
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    
    # ============== Forward Pass ==============
    print("\n[3] Running forward pass...")
    model.eval()
    
    with torch.no_grad():
        output = model(
            static_categorical=data['static_categorical'],
            static_real=data['static_real'],
            encoder_known_categorical=data['encoder_known_categorical'],
            encoder_known_real=data['encoder_known_real'],
            encoder_unknown_real=data['encoder_unknown_real'],
            decoder_known_categorical=data['decoder_known_categorical'],
            decoder_known_real=data['decoder_known_real']
        )
    
    print(f"    ✓ Forward pass completed successfully!")
    print(f"    Output shape (quantile_predictions): {output['quantile_predictions'].shape}")
    print(f"    Expected shape: (batch={batch_size}, pred_len={prediction_length}, quantiles=3)")
    print(f"    Attention weights shape: {output['attention_weights'].shape}")
    
    # ============== Multi-Horizon Predictions ==============
    print("\n[4] Extracting multi-horizon predictions (t+2h, t+4h, t+6h)...")
    horizon_preds = model.get_multi_horizon_predictions(output, horizons=[2, 4, 6])
    
    for h, pred in horizon_preds.items():
        print(f"    t+{h}h predictions shape: {pred.shape}")
        print(f"           Sample quantiles [0.1, 0.5, 0.9]: {pred[0].tolist()}")
    
    # ============== Loss Computation ==============
    print("\n[5] Computing Quantile Loss...")
    loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    loss = loss_fn(output['quantile_predictions'], data['target'])
    print(f"    Quantile Loss: {loss.item():.4f}")
    
    # ============== Format Output ==============
    print("\n[6] Formatting predictions to OutputList...")
    formatted_output = format_output(
        predictions=output['quantile_predictions'],
        timestamps=data['timestamps'],
        latitudes=data['latitudes'],
        longitudes=data['longitudes'],
        batch_idx=0
    )
    
    print(f"\n    Sample OutputList (batch_idx=0):")
    print("-" * 50)
    for i, out in enumerate(formatted_output.outputs):
        print(f"    Horizon t+{i+1}h:")
        print(f"      DateTime: {out.DateTime}")
        print(f"      Latitude: {out.latitude}")
        print(f"      Longitude: {out.longitude}")
        print(f"      Predicted Congestion: {out.predicted_congestion_level:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
