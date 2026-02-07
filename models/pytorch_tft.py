"""
Temporal Fusion Transformer using PyTorch Forecasting Library

This module provides an alternative TFT implementation using the pytorch-forecasting
library, which offers a high-level interface for the TFT architecture.

Installation:
    pip install pytorch-forecasting pytorch-lightning

Usage:
    from models.pytorch_tft import TrafficTFTDataModule, create_pytorch_forecasting_tft
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_lightning as pl

class TFTWrapper(pl.LightningModule):
    """
    Wrapper to ensure the TFT model is recognized as a LightningModule.
    """
    
    def __init__(self, tft_model: 'TemporalFusionTransformer'):
        super().__init__()
        self.tft = tft_model
    
    def training_step(self, batch, batch_idx):
        return self.tft.training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.tft.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return self.tft.configure_optimizers()
    
    def predict_step(self, batch, batch_idx):
        return self.tft.predict_step(batch, batch_idx)
    
    def predict(self, *args, **kwargs):
        return self.tft.predict(*args, **kwargs)


def create_pytorch_forecasting_tft(
    training_dataset: 'TimeSeriesDataSet',
    hidden_size: int = 64,
    lstm_layers: int = 2,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    quantiles: List[float] = [0.1, 0.5, 0.9]
) -> TFTWrapper:
    """
    Data module for preparing traffic data for PyTorch Forecasting TFT.
    
    Converts the traffic dataset schema into TimeSeriesDataSet format required
    by PyTorch Forecasting.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        max_encoder_length: int = 48,
        max_prediction_length: int = 18,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Args:
            data: DataFrame with columns matching the dataset schema
            max_encoder_length: Historical context length
            max_prediction_length: Forecast horizon
            batch_size: Batch size for training
            num_workers: Number of data loading workers
        """
        self.data = data
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def prepare_dataset(self) -> 'TimeSeriesDataSet':
        """
        Create a TimeSeriesDataSet from the traffic data.
        
        Returns:
            TimeSeriesDataSet configured for TFT training
        """
        # Define static categorical features
        static_categoricals = [
            'edge_id',
            'road_type',
            'start_node_id',
            'end_node_id'
        ]
        
        # Define static real features
        static_reals = [
            'road_length_meters',
            'lane_count',
            'speed_limit_kph',
            'free_flow_speed_kph',
            'road_capacity',
            'latitude_midroad',
            'longitude_midroad'
        ]
        
        # Define time-varying known categorical features (available in future)
        time_varying_known_categoricals = [
            'is_weekend',
            'is_holiday',
            'weather_condition'
        ]
        
        # Define time-varying known real features (available in future)
        time_varying_known_reals = [
            'hour_of_day',
            'day_of_week',
            'visibility',
            'event_impact_score',
            'time_idx'  # Required by PyTorch Forecasting
        ]
        
        # Define time-varying unknown real features (historical only)
        time_varying_unknown_reals = [
            'average_speed_kph',
            'vehicle_count',
            'travel_time_seconds',
            'congestion_level',  # Target variable
            'neighbor_avg_congestion_t-1',
            'neighbor_avg_speed_t-1',
            'upstream_congestion_t-1',
            'downstream_congestion_t-1'
        ]
        
        # Create TimeSeriesDataSet
        training = TimeSeriesDataSet(
            self.data,
            time_idx='time_idx',
            target='congestion_level',
            group_ids=['edge_id'],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=['edge_id'],
                transformation='softplus'
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        return training
    
    def create_dataloaders(
        self,
        training_dataset: TimeSeriesDataSet,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Tuple:
        """
        Create train and validation dataloaders.
        
        Args:
            training_dataset: TimeSeriesDataSet for training
            validation_data: Optional validation DataFrame
        
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        train_dataloader = training_dataset.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        if validation_data is not None:
            validation_dataset = TimeSeriesDataSet.from_dataset(
                training_dataset,
                validation_data,
                predict=True,
                stop_randomization=True
            )
            val_dataloader = validation_dataset.to_dataloader(
                train=False,
                batch_size=self.batch_size * 2,
                num_workers=self.num_workers
            )
        else:
            val_dataloader = None
        
        return train_dataloader, val_dataloader


def create_pytorch_forecasting_tft(
    training_dataset: 'TimeSeriesDataSet',
    hidden_size: int = 64,
    lstm_layers: int = 2,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    quantiles: List[float] = [0.1, 0.5, 0.9]
) -> 'TemporalFusionTransformer':
    """
    Create a TFT model using PyTorch Forecasting.
    
    Args:
        training_dataset: TimeSeriesDataSet used for training
        hidden_size: Size of hidden layers
        lstm_layers: Number of LSTM layers
        attention_head_size: Number of attention heads
        dropout: Dropout rate
        learning_rate: Learning rate for optimizer
        quantiles: Quantiles for prediction intervals
    
    Returns:
        Configured TemporalFusionTransformer model
    """
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_size // 4,
        output_size=len(quantiles),
        loss=QuantileLoss(quantiles=quantiles),
        log_interval=10,
        reduce_on_plateau_patience=4,
        optimizer='Adam'
    )
    
    return TFTWrapper(tft)


def train_tft_model(
    model: TFTWrapper,
    train_dataloader,
    val_dataloader=None,
    max_epochs: int = 50,
    gpus: int = 0,
    gradient_clip_val: float = 0.1
) -> 'pl.Trainer':
    """
    Train the TFT model using PyTorch Lightning.
    
    Args:
        model: TFT model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        max_epochs: Maximum training epochs
        gpus: Number of GPUs to use (0 for CPU)
        gradient_clip_val: Gradient clipping value
    
    Returns:
        Trained PyTorch Lightning Trainer
    """
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto", 
        devices=1,
        gradient_clip_val=gradient_clip_val,
        limit_train_batches=50,  # Can be removed for full training
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")
        ]
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    return trainer


def predict_and_format(
    model: TFTWrapper,
    dataloader,
    output_format: str = 'dict'
) -> Dict:
    """
    Generate predictions and format them.
    
    Args:
        model: Trained TFT model
        dataloader: Data loader for prediction
        output_format: 'dict' or 'dataframe'
    
    Returns:
        Predictions in specified format
    """
    predictions = model.predict(dataloader, mode="prediction", return_x=True)
    
    if output_format == 'dict':
        return {
            'predictions': predictions.output,
            'x': predictions.x,
            'index': predictions.index
        }
    elif output_format == 'dataframe':
        # Convert to DataFrame format
        pred_df = pd.DataFrame({
            'prediction': predictions.output.cpu().numpy()
        })
        return pred_df
    
    return predictions


# ============== Example Usage ==============

def example_usage():
    """
    Example of how to use the PyTorch Forecasting TFT implementation.
    """
    
    # 1. Prepare dummy data
    print("Creating dummy traffic dataset...")
    num_edges = 100
    time_steps = 1000
    
    data = []
    for edge_id in range(num_edges):
        for t in range(time_steps):
            data.append({
                'time_idx': t,
                'edge_id': str(edge_id),
                'road_type': np.random.choice(['highway', 'arterial', 'local']),
                'start_node_id': str(np.random.randint(0, 200)),
                'end_node_id': str(np.random.randint(0, 200)),
                'road_length_meters': np.random.uniform(100, 1100),
                'lane_count': np.random.randint(1, 5),
                'speed_limit_kph': np.random.uniform(30, 110),
                'free_flow_speed_kph': np.random.uniform(40, 110),
                'road_capacity': np.random.uniform(500, 2500),
                'latitude_midroad': np.random.uniform(40.7, 40.8),
                'longitude_midroad': np.random.uniform(-74.0, -73.9),
                'is_weekend': str(np.random.choice([0, 1])),  # Convert to string
                'is_holiday': str(int(np.random.random() < 0.05)),  # Convert to string
                'weather_condition': np.random.choice(['sunny', 'rainy', 'cloudy', 'snowy', 'foggy']),
                'hour_of_day': t % 24,
                'day_of_week': (t // 24) % 7,
                'visibility': np.random.uniform(0, 10),
                'event_impact_score': np.random.uniform(0, 1),
                'average_speed_kph': np.random.uniform(10, 90),
                'vehicle_count': np.random.randint(0, 500),
                'travel_time_seconds': np.random.uniform(60, 360),
                'congestion_level': np.random.uniform(0, 1),
                'neighbor_avg_congestion_t-1': np.random.uniform(0, 1),
                'neighbor_avg_speed_t-1': np.random.uniform(10, 90),
                'upstream_congestion_t-1': np.random.uniform(0, 1),
                'downstream_congestion_t-1': np.random.uniform(0, 1)
            })
    
    df = pd.DataFrame(data)
    
    # 2. Create data module
    print("Creating data module...")
    data_module = TFTWrapper(
        data=df,
        max_encoder_length=48,
        max_prediction_length=18,
        batch_size=32
    )
    
    # 3. Prepare dataset
    print("Preparing TimeSeriesDataSet...")
    training_dataset = data_module.prepare_dataset()
    
    # 4. Create dataloaders
    print("Creating dataloaders...")
    train_dataloader, val_dataloader = data_module.create_dataloaders(training_dataset)
    
    # 5. Create model
    print("Creating TFT model...")
    model = create_pytorch_forecasting_tft(
        training_dataset=training_dataset,
        hidden_size=64,
        lstm_layers=2,
        attention_head_size=4,
        dropout=0.1,
        learning_rate=1e-3,
        quantiles=[0.1, 0.5, 0.9]
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 6. Train model (optional - commented out for quick demo)
    print("Training model...")
    trainer = train_tft_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        max_epochs=2
    )
    
    # 7. Make predictions (requires trained model)
    predictions = predict_and_format(model, val_dataloader)
    
    print("✓ PyTorch Forecasting TFT setup complete!")
    print("\nTo train the model, uncomment the training section in the example.")


if __name__ == "__main__":
    example_usage()
