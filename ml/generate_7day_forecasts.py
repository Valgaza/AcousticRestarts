import torch
import torch.nn as nn
from tqdm import tqdm
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

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


def generate_sliding_window_forecasts(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    start_datetime: datetime,
    prediction_interval_minutes: int = 20,
    target_steps: int = 504,  # 7 days * 24 hours * 3 (20-min intervals per hour)
    model_prediction_length: int = 18
) -> OutputList:
    """
    Generate long-term forecasts using sliding window approach.
    
    The model can only predict 18 steps ahead, but we need 504 steps for 7 days.
    We use a sliding window approach: predict 18 steps, shift forward, repeat.
    
    Args:
        model: Trained TFT model
        data_loader: DataLoader with test data
        device: Device to run inference on (GPU)
        start_datetime: Starting datetime for predictions
        prediction_interval_minutes: Time interval between predictions (default 20 min)
        target_steps: Total number of time steps to generate (default 504 for 7 days)
        model_prediction_length: Model's prediction horizon (default 18)
    
    Returns:
        OutputList containing predictions for all locations and time steps
    """
    model.eval()
    outputs = []
    
    # Calculate number of sliding windows needed
    num_windows = int(np.ceil(target_steps / model_prediction_length))
    
    print(f"\n{'='*70}")
    print(f"GENERATING 7-DAY FORECASTS WITH SLIDING WINDOW")
    print(f"{'='*70}")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {(start_datetime + timedelta(minutes=target_steps * prediction_interval_minutes)).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Interval: {prediction_interval_minutes} minutes")
    print(f"Target time steps: {target_steps}")
    print(f"Model prediction length: {model_prediction_length}")
    print(f"Number of sliding windows: {num_windows}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Collect all batches first (entire dataset)
    all_batches = []
    print("📦 Loading all batches into memory...")
    for batch in tqdm(data_loader, desc="Loading batches"):
        all_batches.append(move_batch_to_device(batch, device))
    
    total_locations = sum(batch['static_real'].size(0) for batch in all_batches)
    print(f"✓ Loaded {len(all_batches)} batches with {total_locations} locations")
    
    # For each location, we'll accumulate predictions across sliding windows
    location_forecasts = {}  # key: (lat, lon), value: list of predictions
    
    with torch.no_grad():
        # Process each sliding window
        for window_idx in range(num_windows):
            print(f"\n🔮 Processing sliding window {window_idx + 1}/{num_windows}")
            
            # For each batch in the dataset
            for batch_idx, batch in enumerate(tqdm(all_batches, desc=f"Window {window_idx + 1} batches")):
                # Forward pass
                output = model(
                    static_categorical=batch['static_categorical'],
                    static_real=batch['static_real'],
                    encoder_known_categorical=batch['encoder_known_categorical'],
                    encoder_known_real=batch['encoder_known_real'],
                    encoder_unknown_real=batch['encoder_unknown_real'],
                    decoder_known_categorical=batch['decoder_known_categorical'],
                    decoder_known_real=batch['decoder_known_real']
                )
                
                # Get median predictions (quantile index 1)
                quantile_predictions = output['quantile_predictions'][:, :, 1]  # (batch, pred_len)
                
                # Get location info from batch
                latitudes = batch['static_real'][:, 5].cpu().numpy()
                longitudes = batch['static_real'][:, 6].cpu().numpy()
                
                batch_size = quantile_predictions.size(0)
                pred_len = quantile_predictions.size(1)
                
                # Store predictions for each location in this batch
                for sample_idx in range(batch_size):
                    lat = float(latitudes[sample_idx])
                    lon = float(longitudes[sample_idx])
                    location_key = (lat, lon)
                    
                    # Initialize location if first time seeing it
                    if location_key not in location_forecasts:
                        location_forecasts[location_key] = []
                    
                    # Extract predictions for this window
                    window_predictions = quantile_predictions[sample_idx, :].cpu().numpy()
                    
                    # Append to location's forecast list
                    location_forecasts[location_key].extend(window_predictions.tolist())
            
            # TODO: Update encoder/decoder inputs for next window
            # This is simplified - in production, you'd shift the time indices
            # and update the encoder_unknown_real with previous predictions
    
    # Convert accumulated forecasts to Output objects
    print(f"\n📝 Converting {len(location_forecasts)} locations to output format...")
    
    for (lat, lon), predictions in tqdm(location_forecasts.items(), desc="Formatting outputs"):
        # Trim predictions to target_steps
        predictions = predictions[:target_steps]
        
        # Create output for each time step
        for time_step, congestion_level in enumerate(predictions):
            # Calculate datetime for this prediction
            time_offset = timedelta(minutes=time_step * prediction_interval_minutes)
            pred_datetime = start_datetime + time_offset
            
            # Create output object
            output_obj = Output(
                DateTime=pred_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                latitude=int(lat * 1000000),
                longitude=int(lon * 1000000),
                predicted_congestion_level=float(congestion_level)
            )
            outputs.append(output_obj)
    
    return OutputList(outputs=outputs)


def main():
    """Main function to generate 7-day forecasts."""
    parser = argparse.ArgumentParser(description="Generate 7-day traffic forecasts using sliding window")
    parser.add_argument("--model", default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--csv", default="data/traffic_synthetic.csv", help="Path to input CSV data")
    parser.add_argument("--output", default="checkpoints/forecasts_7d.json", help="Output JSON path")
    parser.add_argument("--start", default=None, help="Start datetime (YYYY-MM-DD HH:MM:SS), defaults to now")
    parser.add_argument("--interval", type=int, default=20, help="Interval in minutes between forecasts")
    parser.add_argument("--steps", type=int, default=504, help="Total time steps (default 504 for 7 days)")
    args = parser.parse_args()
    
    # Device configuration - Force GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"DEVICE CONFIGURATION")
    print(f"{'='*70}")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        print("⚠️  WARNING: GPU not available, using CPU")
        print("⚠️  This will be significantly slower!")
    
    # Parse start datetime
    if args.start:
        start_datetime = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    else:
        start_datetime = datetime.now().replace(second=0, microsecond=0)
    
    print(f"\n{'='*70}")
    print(f"CONFIGURATION")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"CSV: {args.csv}")
    print(f"Output: {args.output}")
    print(f"Start: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Interval: {args.interval} minutes")
    print(f"Total steps: {args.steps}")
    print(f"Duration: {args.steps * args.interval / 60:.1f} hours ({args.steps * args.interval / 60 / 24:.1f} days)")
    
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
    
    # Create data loader for ENTIRE dataset (no time filtering)
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    data_loader = create_data_loader(
        csv_path=str(csv_path),
        batch_size=64,  # Larger batch size for GPU efficiency
        encoder_length=48,  # Must match training config
        prediction_length=18,  # Must match training config
        shuffle=False,
        num_workers=0,
        device=device,
        time_idx_min=None,  # Use entire dataset
        time_idx_max=None
    )
    print(f"✓ Loaded {len(data_loader.dataset)} samples")
    print(f"✓ Batch size: {data_loader.batch_size}")
    print(f"✓ Number of batches: {len(data_loader)}")
    
    # Get dataset info for model creation
    dataset = data_loader.dataset
    num_edges = dataset.num_edges
    num_road_types = dataset.num_road_types
    num_nodes = dataset.num_nodes
    
    print(f"\n📊 Dataset Information:")
    print(f"   Edges: {num_edges}")
    print(f"   Road types: {num_road_types}")
    print(f"   Nodes: {num_nodes}")
    
    # Create model and load onto GPU
    print(f"\n{'='*70}")
    print("LOADING MODEL")
    print(f"{'='*70}")
    model = create_tft_model(
        num_edges=num_edges,
        num_road_types=num_road_types,
        num_nodes=num_nodes,
        hidden_dim=32,  # Must match training config
        encoder_length=48,
        prediction_length=18
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"✓ Model on device: {next(model.parameters()).device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    if device.type == 'cuda':
        print(f"\n📊 GPU Memory after model load:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # Generate forecasts using sliding window approach
    forecasts = generate_sliding_window_forecasts(
        model=model,
        data_loader=data_loader,
        device=device,
        start_datetime=start_datetime,
        prediction_interval_minutes=args.interval,
        target_steps=args.steps,
        model_prediction_length=18
    )
    
    # Save forecasts
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SAVING FORECASTS")
    print(f"{'='*70}")
    print(f"💾 Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(forecasts.model_dump(), f, indent=2)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("✅ FORECAST GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total forecasts: {len(forecasts.outputs):,}")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024**2:.2f} MB")
    
    # Calculate unique locations and time points
    unique_locations = len(set((f.latitude, f.longitude) for f in forecasts.outputs))
    unique_times = len(set(f.DateTime for f in forecasts.outputs))
    print(f"\n📊 Statistics:")
    print(f"   Unique locations: {unique_locations:,}")
    print(f"   Unique time points: {unique_times:,}")
    if unique_locations > 0:
        print(f"   Avg forecasts per location: {len(forecasts.outputs) / unique_locations:.1f}")
    
    # Show sample forecasts
    print(f"\n📋 Sample forecasts (first 5):")
    for i, forecast in enumerate(forecasts.outputs[:5]):
        print(f"  {i+1}. {forecast.DateTime} | "
              f"Lat: {forecast.latitude/1000000:.6f}, Lon: {forecast.longitude/1000000:.6f} | "
              f"Congestion: {forecast.predicted_congestion_level:.4f}")
        
    print(f"\n✨ Done! Run the backend to serve these forecasts to the frontend.")


if __name__ == "__main__":
    main()