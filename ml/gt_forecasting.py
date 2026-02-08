"""
Ground Truth Forecasting Script

Generates traffic congestion forecasts for specific routes and times from a CSV file
using a pretrained TFT model.

Usage:
    python gt_forecasting.py
    
    Or with uv:
    uv run python gt_forecasting.py

Features:
    - Loads pretrained model from checkpoints/best_model.pth
    - Processes optimized_edges.csv to create forecasts
    - Uses GPU if available
    - Outputs in OutputList JSON format
    - Uses training data to provide historical context
"""

import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader

from models.custom_temporal_transformer import create_tft_model
from outputs_format import Output, OutputList


class InferenceTrafficDataset(Dataset):
    """
    Dataset for inference on specific routes from CSV.
    
    Uses historical data from training dataset and forecasts for routes specified
    in the target CSV file.
    """
    
    def __init__(
        self,
        target_csv_path: str,
        training_csv_path: str,
        encoder_length: int = 48,
        prediction_length: int = 18,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            target_csv_path: Path to the optimized_edges CSV (routes to forecast)
            training_csv_path: Path to the training data CSV (for historical context)
            encoder_length: Length of encoder sequence (must match training)
            prediction_length: Length of prediction horizon (must match training)
            device: Device for tensor creation (use CPU for dataset)
        """
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.device = torch.device('cpu')  # Always use CPU for dataset
        self.total_length = encoder_length + prediction_length
        
        # Load target routes (what we want to forecast for)
        self.target_df = pd.read_csv(target_csv_path)
        print(f"Loaded {len(self.target_df)} routes from target CSV")
        
        # Load training data (for historical context)
        self.train_df = pd.read_csv(training_csv_path)
        print(f"Loaded {len(self.train_df)} records from training CSV")
        
        # Preprocess both datasets
        self._preprocess_data()
        
        # Create sequences for each target edge
        self.sequences = []
        self._create_sequences()
        
        if len(self.sequences) == 0:
            raise ValueError(
                f"No valid sequences found! Edges in target CSV may not have enough "
                f"history in training data."
            )
    
    def _preprocess_data(self):
        """Preprocess both dataframes."""
        # Process target dataframe
        self.target_df['is_weekend'] = self.target_df['is_weekend'].astype(int)
        self.target_df['is_holiday'] = self.target_df['is_holiday'].astype(int)
        
        # Process training dataframe
        self.train_df['is_weekend'] = self.train_df['is_weekend'].astype(int)
        self.train_df['is_holiday'] = self.train_df['is_holiday'].astype(int)
        
        # Create categorical encodings for both
        road_type_mapping = {
            'Residential': 0,
            'Ramp': 1,
            'Highway': 2,
            'Arterial': 3
        }
        self.target_df['road_type_encoded'] = self.target_df['road_type'].map(road_type_mapping).fillna(0).astype(int)
        self.train_df['road_type_encoded'] = self.train_df['road_type'].map(road_type_mapping).fillna(0).astype(int)
        
        weather_mapping = {
            'Clear': 0,
            'Rain': 1,
            'Fog': 2
        }
        self.target_df['weather_encoded'] = self.target_df['weather_condition'].map(weather_mapping).fillna(0).astype(int)
        self.train_df['weather_encoded'] = self.train_df['weather_condition'].map(weather_mapping).fillna(0).astype(int)
        
        # Fill missing values
        self.target_df = self.target_df.fillna(0)
        self.train_df = self.train_df.fillna(0)
        
        # Get metadata
        self.num_edges = self.train_df['edge_id'].nunique()
        self.num_road_types = self.train_df['road_type_encoded'].nunique()
        self.num_nodes = max(self.train_df['start_node_id'].max(), self.train_df['end_node_id'].max()) + 1
        self.num_weather_conditions = self.train_df['weather_encoded'].nunique()
        
        # Sort training data
        self.train_df = self.train_df.sort_values(['edge_id', 'time_idx']).reset_index(drop=True)
        
        # Group training data by edge
        self.train_edge_groups = self.train_df.groupby('edge_id')
    
    def _create_sequences(self):
        """
        Create sequences for forecasting.
        
        For each edge in target CSV, use the most recent data from training CSV
        as historical context.
        """
        target_edges = self.target_df['edge_id'].unique()
        
        for edge_id in target_edges:
            # Check if edge exists in training data
            if edge_id not in self.train_edge_groups.groups:
                print(f"Warning: Edge {edge_id} not found in training data. Skipping.")
                continue
            
            # Get training data for this edge
            train_group = self.train_edge_groups.get_group(edge_id).reset_index(drop=True)
            
            # Get target row for this edge
            target_row = self.target_df[self.target_df['edge_id'] == edge_id].iloc[0]
            
            # Need at least encoder_length timesteps in training data
            if len(train_group) < self.encoder_length:
                print(f"Warning: Edge {edge_id} has only {len(train_group)} timesteps "
                      f"in training data, need {self.encoder_length}. Skipping.")
                continue
            
            # Use the most recent encoder_length timesteps from training
            # This provides the historical context for forecasting
            self.sequences.append((edge_id, target_row))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single inference sample.
        
        Uses historical data from training set and target metadata from inference set.
        
        Returns dictionary with all features needed for TFT forward pass.
        """
        edge_id, target_row = self.sequences[idx]
        
        # Get historical data from training set
        train_group = self.train_edge_groups.get_group(edge_id).reset_index(drop=True)
        
        # Use the most recent encoder_length timesteps
        encoder_data = train_group.iloc[-self.encoder_length:].copy()
        
        # For decoder, use the target row information
        # Replicate the target row for prediction window
        decoder_data = pd.DataFrame([target_row.to_dict()] * self.prediction_length)
        
        # Update time-based features for decoder (simulate future timesteps)
        base_hour = int(target_row['hour_of_day'])
        base_day = int(target_row['day_of_week'])
        
        for i in range(self.prediction_length):
            # Increment hour_of_day (wrapping at 24)
            decoder_data.loc[i, 'hour_of_day'] = (base_hour + i) % 24
            # Update day_of_week if we cross midnight
            days_elapsed = (base_hour + i) // 24
            decoder_data.loc[i, 'day_of_week'] = (base_day + days_elapsed) % 7
            # Update is_weekend based on day_of_week
            decoder_data.loc[i, 'is_weekend'] = int(decoder_data.loc[i, 'day_of_week'] >= 5)
        
        # ============== Static Features ==============
        static_categorical = {
            'edge_id': torch.tensor(int(encoder_data['edge_id'].iloc[0]), dtype=torch.long),
            'road_type': torch.tensor(int(encoder_data['road_type_encoded'].iloc[0]), dtype=torch.long),
            'start_node_id': torch.tensor(int(encoder_data['start_node_id'].iloc[0]), dtype=torch.long),
            'end_node_id': torch.tensor(int(encoder_data['end_node_id'].iloc[0]), dtype=torch.long)
        }
        
        static_real = torch.tensor([
            float(encoder_data['road_length_meters'].iloc[0]),
            float(encoder_data['lane_count'].iloc[0]),
            float(encoder_data['speed_limit_kph'].iloc[0]),
            float(encoder_data['free_flow_speed_kph'].iloc[0]),
            float(encoder_data['road_capacity'].iloc[0]),
            float(encoder_data['latitude_midroad'].iloc[0]),
            float(encoder_data['longitude_midroad'].iloc[0])
        ], dtype=torch.float32)
        
        # ============== Encoder Features ==============
        encoder_known_categorical = {
            'is_weekend': torch.tensor(encoder_data['is_weekend'].values, dtype=torch.long),
            'is_holiday': torch.tensor(encoder_data['is_holiday'].values, dtype=torch.long),
            'weather_condition': torch.tensor(encoder_data['weather_encoded'].values, dtype=torch.long)
        }
        
        encoder_known_real = torch.tensor(
            encoder_data[['hour_of_day', 'day_of_week', 'visibility', 'event_impact_score']].values,
            dtype=torch.float32
        )
        
        # Unknown real features (what we're trying to predict)
        encoder_unknown_cols = [
            'average_speed_kph', 'vehicle_count', 'travel_time_seconds', 'congestion_level',
            'neighbor_avg_congestion_t-1', 'neighbor_avg_speed_t-1',
            'upstream_congestion_t-1', 'downstream_congestion_t-1'
        ]
        encoder_unknown_real = torch.tensor(
            encoder_data[encoder_unknown_cols].values,
            dtype=torch.float32
        )
        
        # ============== Decoder Features ==============
        decoder_known_categorical = {
            'is_weekend': torch.tensor(decoder_data['is_weekend'].values, dtype=torch.long),
            'is_holiday': torch.tensor(decoder_data['is_holiday'].values, dtype=torch.long),
            'weather_condition': torch.tensor(decoder_data['weather_encoded'].values, dtype=torch.long)
        }
        
        decoder_known_real = torch.tensor(
            decoder_data[['hour_of_day', 'day_of_week', 'visibility', 'event_impact_score']].values,
            dtype=torch.float32
        )
        
        # Store metadata for output formatting
        target_time_idx = int(target_row['time_idx'])
        latitude = float(target_row['latitude_midroad'])
        longitude = float(target_row['longitude_midroad'])
        
        return {
            'static_categorical': static_categorical,
            'static_real': static_real,
            'encoder_known_categorical': encoder_known_categorical,
            'encoder_known_real': encoder_known_real,
            'encoder_unknown_real': encoder_unknown_real,
            'decoder_known_categorical': decoder_known_categorical,
            'decoder_known_real': decoder_known_real,
            'metadata': {
                'edge_id': int(edge_id),
                'target_time_idx': target_time_idx,
                'latitude': latitude,
                'longitude': longitude
            }
        }


def collate_inference_batch(batch: List[Dict]) -> Dict:
    """Custom collate function for inference batches."""
    if not batch:
        raise ValueError("Empty batch")
    
    # Batch static features
    static_categorical = {
        key: torch.stack([sample['static_categorical'][key] for sample in batch])
        for key in batch[0]['static_categorical'].keys()
    }
    
    static_real = torch.stack([sample['static_real'] for sample in batch])
    
    # Batch encoder features
    encoder_known_categorical = {
        key: torch.stack([sample['encoder_known_categorical'][key] for sample in batch])
        for key in batch[0]['encoder_known_categorical'].keys()
    }
    
    encoder_known_real = torch.stack([sample['encoder_known_real'] for sample in batch])
    encoder_unknown_real = torch.stack([sample['encoder_unknown_real'] for sample in batch])
    
    # Batch decoder features
    decoder_known_categorical = {
        key: torch.stack([sample['decoder_known_categorical'][key] for sample in batch])
        for key in batch[0]['decoder_known_categorical'].keys()
    }
    
    decoder_known_real = torch.stack([sample['decoder_known_real'] for sample in batch])
    
    # Collect metadata
    metadata = [sample['metadata'] for sample in batch]
    
    return {
        'static_categorical': static_categorical,
        'static_real': static_real,
        'encoder_known_categorical': encoder_known_categorical,
        'encoder_known_real': encoder_known_real,
        'encoder_unknown_real': encoder_unknown_real,
        'decoder_known_categorical': decoder_known_categorical,
        'decoder_known_real': decoder_known_real,
        'metadata': metadata
    }


def load_model(
    checkpoint_path: str,
    config: Dict,
    device: torch.device
) -> nn.Module:
    """
    Load pretrained TFT model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        config: Configuration dictionary with model parameters
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    # Use training dataset dimensions (model was trained on this)
    # NOT the inference dataset dimensions
    training_csv = config.get('csv_path', './ml/data/traffic_synthetic.csv')
    temp_df = pd.read_csv(training_csv)
    
    # Calculate num_nodes properly from training data
    num_edges = temp_df['edge_id'].nunique()
    num_nodes = max(temp_df['start_node_id'].max(), temp_df['end_node_id'].max()) + 1
    num_road_types = 4  # Residential, Ramp, Highway, Arterial
    
    print(f"Model architecture: {num_edges} edges, {num_nodes} nodes, {num_road_types} road types")
    
    # Create model with same architecture as training
    model = create_tft_model(
        num_edges=num_edges,
        num_road_types=num_road_types,
        num_nodes=num_nodes,
        hidden_dim=config.get('hidden_dim', 32),
        encoder_length=config.get('encoder_length', 48),
        prediction_length=config.get('prediction_length', 18)
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    model = model.to(device)
    model.eval()
    
    return model


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move batch tensors to specified device."""
    batch_device = {}
    for key, value in batch.items():
        if key == 'metadata':
            batch_device[key] = value
        elif isinstance(value, dict):
            batch_device[key] = {k: v.to(device) for k, v in value.items()}
        elif isinstance(value, torch.Tensor):
            batch_device[key] = value.to(device)
        else:
            batch_device[key] = value
    return batch_device


def generate_forecasts(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    start_datetime: str = "2024-01-01 00:00:00",
    time_interval_minutes: int = 20
) -> OutputList:
    """
    Generate forecasts for all sequences in the dataloader.
    
    Args:
        model: Trained TFT model
        data_loader: DataLoader with inference data
        device: Device to run inference on
        start_datetime: Starting datetime for the dataset
        time_interval_minutes: Time interval between timesteps
    
    Returns:
        OutputList containing all predictions
    """
    model.eval()
    outputs = []
    
    start_dt = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
    
    print(f"\nGenerating forecasts for {len(data_loader.dataset)} sequences...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device
            batch = move_batch_to_device(batch, device)
            metadata = batch.pop('metadata')
            
            # Forward pass
            output_dict = model(
                static_categorical=batch['static_categorical'],
                static_real=batch['static_real'],
                encoder_known_categorical=batch['encoder_known_categorical'],
                encoder_known_real=batch['encoder_known_real'],
                encoder_unknown_real=batch['encoder_unknown_real'],
                decoder_known_categorical=batch['decoder_known_categorical'],
                decoder_known_real=batch['decoder_known_real']
            )
            
            # Extract predictions from dictionary
            # output_dict['quantile_predictions'] shape: (batch, pred_length, num_quantiles)
            predictions = output_dict['quantile_predictions']
            
            # Extract median prediction (quantile index 1)
            pred_median = predictions[:, :, 1].cpu().numpy()  # (batch, pred_length)
            
            # Process each sample in batch
            batch_size = pred_median.shape[0]
            for i in range(batch_size):
                sample_metadata = metadata[i]
                target_time_idx = sample_metadata['target_time_idx']
                latitude = sample_metadata['latitude']
                longitude = sample_metadata['longitude']
                
                # Generate outputs for each prediction timestep
                for t in range(pred_median.shape[1]):
                    # Calculate datetime for this prediction
                    # Start from the target time_idx
                    future_time_idx = target_time_idx + t
                    pred_datetime = start_dt + timedelta(minutes=int(future_time_idx * time_interval_minutes))
                    
                    # Create output
                    output = Output(
                        DateTime=pred_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                        latitude=int(round(latitude)),
                        longitude=int(round(longitude)),
                        predicted_congestion_level=float(pred_median[i, t])
                    )
                    outputs.append(output)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(data_loader)} batches...")
    
    print(f"\nGenerated {len(outputs)} predictions")
    return OutputList(outputs=outputs)


def main():
    """Main forecasting function."""
    # ============== Configuration ==============
    config_path = './checkpoints/config.json'
    checkpoint_path = './checkpoints/best_model.pth'
    csv_path = './ml/data/optimized_edges.csv'
    output_path = './backend/outputs/gt_forecasts.json'
    
    # Load training configuration
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}")
    else:
        print(f"Warning: Config file not found at {config_path}, using defaults")
        config = {
            'encoder_length': 48,
            'prediction_length': 18,
            'hidden_dim': 32
        }
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ============== Load Model ==============
    print("\nLoading pretrained model...")
    model = load_model(checkpoint_path, config, device)
    print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
    
    # ============== Create Dataset ==============
    print(f"\nCreating inference dataset from {csv_path}...")
    dataset = InferenceTrafficDataset(
        target_csv_path=csv_path,
        training_csv_path=config.get('csv_path', './ml/data/traffic_synthetic.csv'),
        encoder_length=config['encoder_length'],
        prediction_length=config['prediction_length']
    )
    print(f"Created {len(dataset)} inference sequences")
    
    # Create dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_inference_batch
    )
    
    # ============== Generate Forecasts ==============
    output_list = generate_forecasts(
        model=model,
        data_loader=data_loader,
        device=device,
        start_datetime=config.get('forecast_start_datetime', '2024-01-01 00:00:00'),
        time_interval_minutes=config.get('forecast_interval_minutes', 20)
    )
    
    # ============== Save Results ==============
    print(f"\nSaving forecasts to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Convert to JSON using pydantic
        json.dump(output_list.model_dump(), f, indent=2)
    
    print(f"✓ Forecasts saved successfully!")
    print(f"  Total predictions: {len(output_list.outputs)}")
    print(f"  Output file: {output_path}")
    
    # Print sample predictions
    if len(output_list.outputs) > 0:
        print("\nSample predictions:")
        for i, output in enumerate(output_list.outputs[:5]):
            print(f"  {i+1}. {output.DateTime} | "
                  f"Lat: {output.latitude}, Lon: {output.longitude} | "
                  f"Congestion: {output.predicted_congestion_level:.4f}")


if __name__ == "__main__":
    main()
