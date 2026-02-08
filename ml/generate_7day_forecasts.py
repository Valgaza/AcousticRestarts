#!/usr/bin/env python3
"""
Generate 7-day traffic forecasts with 30-minute intervals.

This script loads a trained TFT model and generates forecasts for 7 days
with predictions every 30 minutes.

Usage:
    python generate_7day_forecasts.py
    
    Or with uv:
    uv run python generate_7day_forecasts.py

Options:
    --model: Path to model checkpoint (default: checkpoints/best_model.pth)
    --csv: Path to input CSV data (default: data/traffic_synthetic.csv)
    --output: Output JSON path (default: ../backend/outputs/forecasts.json)
    --start: Start datetime (default: current time, format: 'YYYY-MM-DD HH:MM:SS')
    --days: Number of days to forecast (default: 7)
    --interval: Interval in minutes (default: 30)

Output:
    - JSON file with forecasts in the format expected by backend
    - Each forecast includes: DateTime, latitude, longitude, predicted_congestion_level
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from ml.data_loader import create_data_loader
from models.custom_temporal_transformer import create_tft_model
from ml.outputs_format import Output, OutputList


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move batch tensors to specified device."""
    batch_device = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_device[key] = value.to(device)
        elif isinstance(value, dict):
            batch_device[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in value.items()}
        else:
            batch_device[key] = value
    return batch_device


def generate_extended_forecasts(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    start_datetime: datetime,
    num_days: int = 7,
    interval_minutes: int = 30,
    max_outputs: int = 10000
) -> OutputList:
    """
    Generate forecasts for extended time period (7 days).
    
    Args:
        model: Trained TFT model
        data_loader: DataLoader with data
        device: Device to run inference on
        start_datetime: Starting datetime for predictions
        num_days: Number of days to forecast (default: 7)
        interval_minutes: Time interval between predictions (default: 30 min)
        max_outputs: Maximum number of output entries (default: 10000)
    
    Returns:
        OutputList containing predictions for different locations and times
    """
    model.eval()
    outputs = []
    
    # Calculate number of prediction steps
    total_minutes = num_days * 24 * 60
    num_time_steps = total_minutes // interval_minutes
    
    print(f"\n{'='*70}")
    print(f"GENERATING {num_days}-DAY FORECASTS")
    print(f"{'='*70}")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {(start_datetime + timedelta(days=num_days)).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Interval: {interval_minutes} minutes")
    print(f"Total time steps: {num_time_steps}")
    print(f"Max outputs: {max_outputs}")
    print(f"{'='*70}\n")
    
    location_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing locations")):
            # Move batch to device
            batch = move_batch_to_device(batch, device)
            
            # Get location info from batch
            # static_real shape: (batch, 7) - [length, lanes, speed_limit, free_flow_speed, capacity, lat, lon]
            latitudes = batch['static_real'][:, 5].cpu().numpy()
            longitudes = batch['static_real'][:, 6].cpu().numpy()
            batch_size = len(latitudes)
            
            # Process each location in batch
            for sample_idx in range(batch_size):
                if len(outputs) >= max_outputs:
                    print(f"\nReached maximum output limit: {max_outputs}")
                    return OutputList(outputs=outputs)
                
                lat = float(latitudes[sample_idx])
                lon = float(longitudes[sample_idx])
                location_count += 1
                
                # Generate predictions for all time steps at this location
                # We'll need to call the model multiple times if num_time_steps > prediction_length
                current_start_time = start_datetime
                predictions_generated = 0
                
                while predictions_generated < num_time_steps:
                    # Forward pass to get predictions
                    output = model(
                        static_categorical={k: v[sample_idx:sample_idx+1] for k, v in batch['static_categorical'].items()},
                        static_real=batch['static_real'][sample_idx:sample_idx+1],
                        encoder_known_categorical={k: v[sample_idx:sample_idx+1] for k, v in batch['encoder_known_categorical'].items()},
                        encoder_known_real=batch['encoder_known_real'][sample_idx:sample_idx+1],
                        encoder_unknown_real=batch['encoder_unknown_real'][sample_idx:sample_idx+1],
                        decoder_known_categorical={k: v[sample_idx:sample_idx+1] for k, v in batch['decoder_known_categorical'].items()},
                        decoder_known_real=batch['decoder_known_real'][sample_idx:sample_idx+1]
                    )
                    
                    # Get median predictions (quantile index 1)
                    predictions = output['quantile_predictions'][0, :, 1]  # (pred_len,)
                    pred_len = predictions.size(0)
                    
                    # Convert model predictions to output format
                    for time_step in range(min(pred_len, num_time_steps - predictions_generated)):
                        # Calculate datetime for this prediction
                        time_offset = timedelta(minutes=(predictions_generated + time_step) * interval_minutes)
                        pred_datetime = start_datetime + time_offset
                        
                        # Get predicted congestion level
                        congestion_level = float(predictions[time_step].cpu().item())
                        
                        # Create output object
                        output_obj = Output(
                            DateTime=pred_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                            latitude=int(lat * 1000000),  # Convert to integer representation
                            longitude=int(lon * 1000000),
                            predicted_congestion_level=congestion_level
                        )
                        outputs.append(output_obj)
                    
                    predictions_generated += pred_len
                    
                    # Break if we've generated enough predictions for this location
                    if predictions_generated >= num_time_steps:
                        break
                    
                    # For continuing forecast, we would need to update encoder with new predictions
                    # For simplicity, we'll just repeat the last prediction for remaining steps
                    remaining_steps = num_time_steps - predictions_generated
                    last_congestion = float(predictions[-1].cpu().item())
                    
                    for step in range(remaining_steps):
                        time_offset = timedelta(minutes=(predictions_generated + step) * interval_minutes)
                        pred_datetime = start_datetime + time_offset
                        
                        output_obj = Output(
                            DateTime=pred_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                            latitude=int(lat * 1000000),
                            longitude=int(lon * 1000000),
                            predicted_congestion_level=last_congestion
                        )
                        outputs.append(output_obj)
                    
                    break  # Exit the while loop
                
                # Print progress every 10 locations
                if location_count % 10 == 0:
                    avg_per_location = len(outputs) / location_count
                    print(f"  Processed {location_count} locations, {len(outputs)} total forecasts ({avg_per_location:.0f} per location)")
    
    return OutputList(outputs=outputs)


def main():
    """Main function to generate 7-day forecasts."""
    parser = argparse.ArgumentParser(description="Generate 7-day traffic forecasts")
    parser.add_argument("--model", default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--csv", default="data/traffic_synthetic.csv", help="Path to input CSV data")
    parser.add_argument("--output", default="../backend/outputs/forecasts.json", help="Output JSON path")
    parser.add_argument("--start", default=None, help="Start datetime (YYYY-MM-DD HH:MM:SS), defaults to now")
    parser.add_argument("--days", type=int, default=7, help="Number of days to forecast")
    parser.add_argument("--interval", type=int, default=30, help="Interval in minutes between forecasts")
    parser.add_argument("--max-outputs", type=int, default=10000, help="Maximum number of output entries")
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Parse start datetime
    if args.start:
        start_datetime = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    else:
        start_datetime = datetime.now().replace(second=0, microsecond=0)
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  CSV: {args.csv}")
    print(f"  Output: {args.output}")
    print(f"  Start: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Days: {args.days}")
    print(f"  Interval: {args.interval} minutes")
    print(f"  Max outputs: {args.max_outputs}")
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n❌ Error: Model checkpoint not found: {model_path}")
        print("   Please train a model first using train_model.py")
        return
    
    # Check if CSV exists
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"\n❌ Error: CSV data not found: {csv_path}")
        return
    
    # Create data loader
    print("\n📊 Creating data loader...")
    data_loader = create_data_loader(
        csv_path=str(csv_path),
        batch_size=32,
        encoder_length=48,  # Must match training config
        prediction_length=18,  # Must match training config
        shuffle=False,
        num_workers=0,
        device=device
    )
    print(f"   Loaded {len(data_loader.dataset)} samples")
    
    # Get dataset info for model creation
    dataset = data_loader.dataset
    num_edges = dataset.num_edges
    num_road_types = dataset.num_road_types
    num_nodes = dataset.num_nodes
    
    # Create model
    print("\n🤖 Creating model...")
    model = create_tft_model(
        num_edges=num_edges,
        num_road_types=num_road_types,
        num_nodes=num_nodes,
        hidden_dim=32,  # Must match training config
        encoder_length=48,
        prediction_length=18
    ).to(device)
    
    # Load checkpoint
    print(f"   Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Generate forecasts
    print("\n🔮 Generating forecasts...")
    forecasts = generate_extended_forecasts(
        model=model,
        data_loader=data_loader,
        device=device,
        start_datetime=start_datetime,
        num_days=args.days,
        interval_minutes=args.interval,
        max_outputs=args.max_outputs
    )
    
    # Save forecasts
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving forecasts to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(forecasts.model_dump(), f, indent=2)
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ FORECAST GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total forecasts: {len(forecasts.outputs):,}")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Calculate unique locations and time points
    unique_locations = len(set((f.latitude, f.longitude) for f in forecasts.outputs))
    unique_times = len(set(f.DateTime for f in forecasts.outputs))
    print(f"Unique locations: {unique_locations}")
    print(f"Unique time points: {unique_times}")
    print(f"Avg forecasts per location: {len(forecasts.outputs) / unique_locations:.1f}")
    
    # Show sample forecasts
    print(f"\n📋 Sample forecasts:")
    for i, forecast in enumerate(forecasts.outputs[:5]):
        print(f"  {i+1}. {forecast.DateTime} | "
              f"Lat: {forecast.latitude/1000000:.6f}, Lon: {forecast.longitude/1000000:.6f} | "
              f"Congestion: {forecast.predicted_congestion_level:.4f}")
    
    print(f"\n✨ Done! Run the backend to serve these forecasts to the frontend.")


if __name__ == "__main__":
    main()
