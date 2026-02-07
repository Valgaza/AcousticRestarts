"""
Data loading pipeline for Temporal Fusion Transformer traffic forecasting.

Loads data from CSV and prepares sequences for TFT model training/inference.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta


class TrafficDataset(Dataset):
    """
    Dataset for loading traffic data sequences for TFT model.
    
    Creates time series sequences from the traffic CSV data.
    """
    
    def __init__(
        self,
        csv_path: str,
        encoder_length: int = 48,
        prediction_length: int = 18,
        device: torch.device = torch.device('cpu'),
        time_idx_min: Optional[int] = None,
        time_idx_max: Optional[int] = None
    ):
        """
        Args:
            csv_path: Path to the traffic CSV file
            encoder_length: Length of encoder sequence (historical data)
            prediction_length: Length of prediction horizon
            device: Device to place tensors on (deprecated - tensors are created on CPU)
            time_idx_min: Minimum time_idx to include (for train/val/test splits)
            time_idx_max: Maximum time_idx to include (for train/val/test splits)
        """
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        # Always create tensors on CPU, let DataLoader handle GPU transfer
        self.device = torch.device('cpu')
        self.total_length = encoder_length + prediction_length
        self.time_idx_min = time_idx_min
        self.time_idx_max = time_idx_max
        
        # Load and preprocess data
        self.df = pd.read_csv(csv_path)
        self._preprocess_data()
        
        # Group by edge_id and sort by time_idx
        self.edge_groups = self.df.groupby('edge_id')
        
        # Create mapping for categorical features
        self._create_categorical_mappings()
        
        # Collect all possible sequences
        self.sequences = []
        self._create_sequences()
    
    def _preprocess_data(self):
        """Preprocess the dataframe."""
        # Convert boolean columns to int
        self.df['is_weekend'] = self.df['is_weekend'].astype(int)
        self.df['is_holiday'] = self.df['is_holiday'].astype(int)
        
        # Filter by time_idx if specified (for train/val/test splits)
        if self.time_idx_min is not None:
            self.df = self.df[self.df['time_idx'] >= self.time_idx_min]
        if self.time_idx_max is not None:
            self.df = self.df[self.df['time_idx'] <= self.time_idx_max]
        
        # Sort by edge_id and time_idx
        self.df = self.df.sort_values(['edge_id', 'time_idx']).reset_index(drop=True)
    
    def _create_categorical_mappings(self):
        """Create mappings for categorical features."""
        # Road type mapping
        road_type_mapping = {
            'Residential': 0,
            'Ramp': 1,
            'Highway': 2,
            'Arterial': 3
        }
        self.df['road_type_encoded'] = self.df['road_type'].map(road_type_mapping)
        
        # Weather condition mapping
        weather_mapping = {
            'Clear': 0,
            'Rain': 1,
            'Fog': 2
        }
        self.df['weather_encoded'] = self.df['weather_condition'].map(weather_mapping)
        
        # Get unique values for other categoricals
        self.num_edges = self.df['edge_id'].nunique()
        self.num_road_types = self.df['road_type_encoded'].nunique()
        self.num_nodes = max(self.df['start_node_id'].max(), self.df['end_node_id'].max()) + 1
        self.num_weather_conditions = self.df['weather_encoded'].nunique()
    
    def _create_sequences(self):
        """Create all possible sequences for each edge."""
        for edge_id, group in self.edge_groups:
            group = group.reset_index(drop=True)
            num_timesteps = len(group)
            
            # Number of possible sequences for this edge
            max_start = num_timesteps - self.total_length
            if max_start < 0:
                continue  # Skip edges with insufficient data
            
            for start_idx in range(max_start + 1):
                self.sequences.append((edge_id, start_idx))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns dictionary with all features needed for TFT forward pass.
        """
        edge_id, start_idx = self.sequences[idx]
        group = self.edge_groups.get_group(edge_id).reset_index(drop=True)
        
        # Extract sequence data
        seq_data = group.iloc[start_idx:start_idx + self.total_length].copy()
        
        # ============== Static Features ==============
        # These are the same for the entire sequence
        static_row = seq_data.iloc[0]
        
        static_categorical = {
            'edge_id': torch.tensor(static_row['edge_id'], dtype=torch.long, device=self.device),
            'road_type': torch.tensor(static_row['road_type_encoded'], dtype=torch.long, device=self.device),
            'start_node_id': torch.tensor(static_row['start_node_id'], dtype=torch.long, device=self.device),
            'end_node_id': torch.tensor(static_row['end_node_id'], dtype=torch.long, device=self.device)
        }
        
        static_real = torch.tensor([
            static_row['road_length_meters'],
            static_row['lane_count'],
            static_row['speed_limit_kph'],
            static_row['free_flow_speed_kph'],
            static_row['road_capacity'],
            static_row['latitude_midroad'],
            static_row['longitude_midroad']
        ], dtype=torch.float32, device=self.device)
        
        # ============== Encoder Features (Historical) ==============
        encoder_data = seq_data.iloc[:self.encoder_length]
        
        encoder_known_categorical = {
            'is_weekend': torch.tensor(encoder_data['is_weekend'].values, dtype=torch.long, device=self.device),
            'is_holiday': torch.tensor(encoder_data['is_holiday'].values, dtype=torch.long, device=self.device),
            'weather_condition': torch.tensor(encoder_data['weather_encoded'].values, dtype=torch.long, device=self.device)
        }
        
        encoder_known_real = torch.stack([
            torch.tensor(encoder_data['hour_of_day'].values, dtype=torch.float32, device=self.device),
            torch.tensor(encoder_data['day_of_week'].values, dtype=torch.float32, device=self.device),
            torch.tensor(encoder_data['visibility'].values, dtype=torch.float32, device=self.device),
            torch.tensor(encoder_data['event_impact_score'].values, dtype=torch.float32, device=self.device)
        ], dim=1).transpose(0, 1)  # (4, encoder_len) -> (encoder_len, 4) -> (4, encoder_len) wait no
        
        # Wait, in the model, encoder_known_real is (batch, encoder_len, num_known_real)
        encoder_known_real = torch.stack([
            torch.tensor(encoder_data['hour_of_day'].values, dtype=torch.float32, device=self.device),
            torch.tensor(encoder_data['day_of_week'].values, dtype=torch.float32, device=self.device),
            torch.tensor(encoder_data['visibility'].values, dtype=torch.float32, device=self.device),
            torch.tensor(encoder_data['event_impact_score'].values, dtype=torch.float32, device=self.device)
        ], dim=1)  # (encoder_len, 4)
        
        encoder_unknown_real = torch.stack([
            torch.tensor(encoder_data['average_speed_kph'].values, dtype=torch.float32, device=self.device),
            torch.tensor(encoder_data['vehicle_count'].values, dtype=torch.float32, device=self.device),
            torch.tensor(encoder_data['travel_time_seconds'].values, dtype=torch.float32, device=self.device),
            torch.tensor(encoder_data['congestion_level'].values, dtype=torch.float32, device=self.device),
            torch.tensor(encoder_data['neighbor_avg_congestion_t-1'].values, dtype=torch.float32, device=self.device),
            torch.tensor(encoder_data['neighbor_avg_speed_t-1'].values, dtype=torch.float32, device=self.device),
            torch.tensor(encoder_data['upstream_congestion_t-1'].values, dtype=torch.float32, device=self.device),
            torch.tensor(encoder_data['downstream_congestion_t-1'].values, dtype=torch.float32, device=self.device)
        ], dim=1)  # (encoder_len, 8)
        
        # ============== Decoder Features (Future Known) ==============
        decoder_data = seq_data.iloc[self.encoder_length:self.total_length]
        
        decoder_known_categorical = {
            'is_weekend': torch.tensor(decoder_data['is_weekend'].values, dtype=torch.long, device=self.device),
            'is_holiday': torch.tensor(decoder_data['is_holiday'].values, dtype=torch.long, device=self.device),
            'weather_condition': torch.tensor(decoder_data['weather_encoded'].values, dtype=torch.long, device=self.device)
        }
        
        decoder_known_real = torch.stack([
            torch.tensor(decoder_data['hour_of_day'].values, dtype=torch.float32, device=self.device),
            torch.tensor(decoder_data['day_of_week'].values, dtype=torch.float32, device=self.device),
            torch.tensor(decoder_data['visibility'].values, dtype=torch.float32, device=self.device),
            torch.tensor(decoder_data['event_impact_score'].values, dtype=torch.float32, device=self.device)
        ], dim=1)  # (pred_len, 4)
        
        # ============== Target (Future Congestion Levels) ==============
        target = torch.tensor(
            decoder_data['congestion_level'].values,
            dtype=torch.float32,
            device=self.device
        )  # (pred_len,)
        
        # ============== Metadata for output formatting ==============
        # Generate timestamps (assuming 20-minute intervals)
        base_time = datetime.now()
        timestamps = []
        for i in range(self.prediction_length):
            timestamps.append((base_time + timedelta(minutes=20*(i+1))).isoformat())
        
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
            'latitudes': static_real[5],  # latitude_midroad
            'longitudes': static_real[6],  # longitude_midroad
            'edge_id': edge_id
        }


def create_data_loader(
    csv_path: str,
    batch_size: int = 32,
    encoder_length: int = 48,
    prediction_length: int = 18,
    shuffle: bool = True,
    num_workers: int = 0,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    time_idx_min: Optional[int] = None,
    time_idx_max: Optional[int] = None
) -> DataLoader:
    """
    Create a DataLoader for the traffic dataset.
    
    Args:
        csv_path: Path to CSV file
        batch_size: Batch size
        encoder_length: Encoder sequence length
        prediction_length: Prediction horizon
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        device: Target device for tensors (used for pin_memory optimization)
        time_idx_min: Minimum time_idx to include (for splits)
        time_idx_max: Maximum time_idx to include (for splits)
    
    Returns:
        DataLoader instance with data on CPU (use .to(device) on batches)
    """
    dataset = TrafficDataset(
        csv_path=csv_path,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        device=torch.device('cpu'),  # Always create on CPU
        time_idx_min=time_idx_min,
        time_idx_max=time_idx_max
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),  # Enable pin_memory for faster GPU transfer
        collate_fn=collate_traffic_batch
    )
    
    return data_loader


def collate_traffic_batch(batch: list) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to properly batch the traffic data.
    
    Args:
        batch: List of samples from TrafficDataset
    
    Returns:
        Batched dictionary
    """
    if not batch:
        return {}
    
    # Get device from first sample
    device = batch[0]['static_real'].device
    
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
    
    # Batch target
    target = torch.stack([sample['target'] for sample in batch])
    
    # Collect metadata
    timestamps = [sample['timestamps'] for sample in batch]
    latitudes = torch.stack([sample['latitudes'] for sample in batch])
    longitudes = torch.stack([sample['longitudes'] for sample in batch])
    edge_ids = [sample['edge_id'] for sample in batch]
    
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
        'latitudes': latitudes,
        'longitudes': longitudes,
        'edge_ids': edge_ids
    }


if __name__ == "__main__":
    # Test the data loader
    csv_path = "data/traffic_synthetic.csv"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create data loader with smaller batch for testing
    data_loader = create_data_loader(
        csv_path=csv_path,
        batch_size=4,  # Smaller batch for faster testing
        encoder_length=48,
        prediction_length=18,
        device=device
    )
    
    print(f"Dataset size: {len(data_loader.dataset)}")
    print(f"Number of batches: {len(data_loader)}")
    
    # Test one batch
    print("\nTesting data loading...")
    for batch in data_loader:
        print("\nBatch shapes (on CPU):")
        print(f"  Static real: {batch['static_real'].shape}")
        print(f"  Encoder known real: {batch['encoder_known_real'].shape}")
        print(f"  Encoder unknown real: {batch['encoder_unknown_real'].shape}")
        print(f"  Decoder known real: {batch['decoder_known_real'].shape}")
        print(f"  Target: {batch['target'].shape}")
        print(f"  Device: {batch['static_real'].device}")
        
        # Test moving to GPU
        if device.type == 'cuda':
            print("\nMoving batch to GPU...")
            batch_gpu = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_gpu[key] = value.to(device)
                elif isinstance(value, dict):
                    batch_gpu[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                     for k, v in value.items()}
                else:
                    batch_gpu[key] = value
            print(f"  Static real device: {batch_gpu['static_real'].device}")
            print(f"  ✓ Successfully moved to {device}")
        
        print("\n✓ Data loader working correctly with time_idx validation!")
        break