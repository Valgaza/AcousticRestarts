"""
Training script for LSTM Seq2Seq baseline traffic forecasting model.

Trains a simple LSTM encoder-decoder model on traffic data with time-based train/val/test splits.

Usage:
    python train_model_lstm.py
    
    Or with uv:
    uv run python train_model_lstm.py

Features:
    - 50 epochs of training
    - Learning rate scheduling with ReduceLROnPlateau
    - Time-based data splits (70% train, 15% val, 15% test)
    - Early stopping with patience of 10 epochs
    - Checkpoint saving for best model
    - Progress bars with tqdm showing all metrics
    - Training history and test results saved as JSON
    - 6-hour forecasts at 20-minute intervals

Outputs:
    - checkpoints/lstm_best_model.pth: Best model based on validation loss
    - checkpoints/lstm_config.json: Training configuration
    - checkpoints/lstm_training_history.json: Loss and metric history
    - checkpoints/lstm_test_results.json: Final test set evaluation
    - checkpoints/lstm_forecasts.json: 6-hour forecasts at 20-minute intervals
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple

from data_loader import create_data_loader
from models.lstm_seq2seq import create_lstm_model
from outputs_format import Output, OutputList


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate regression metrics for model evaluation.
    
    Args:
        predictions: Model predictions (batch, pred_len)
        targets: Ground truth values (batch, pred_len)
    
    Returns:
        Dictionary with MAE, RMSE, and R² metrics
    """
    # Calculate MAE
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    # Calculate RMSE
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    
    # Calculate R² score
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


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


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float, float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (loss, mae, rmse, r2)
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_r2 = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', leave=False)
    
    for batch in pbar:
        # Move batch to device
        batch = move_batch_to_device(batch, device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(
            static_categorical=batch['static_categorical'],
            static_real=batch['static_real'],
            encoder_known_categorical=batch['encoder_known_categorical'],
            encoder_known_real=batch['encoder_known_real'],
            encoder_unknown_real=batch['encoder_unknown_real'],
            decoder_known_categorical=batch['decoder_known_categorical'],
            decoder_known_real=batch['decoder_known_real']
        )
        
        # Calculate loss
        predictions = output['predictions']
        loss = criterion(predictions, batch['target'])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            metrics = calculate_metrics(predictions, batch['target'])
        
        # Accumulate metrics
        total_loss += loss.item()
        total_mae += metrics['mae']
        total_rmse += metrics['rmse']
        total_r2 += metrics['r2']
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mae': f"{metrics['mae']:.4f}",
            'rmse': f"{metrics['rmse']:.4f}"
        })
    
    return (
        total_loss / num_batches,
        total_mae / num_batches,
        total_rmse / num_batches,
        total_r2 / num_batches
    )


def validate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, float, float, float]:
    """
    Validate for one epoch.
    
    Returns:
        Tuple of (loss, mae, rmse, r2)
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_r2 = 0.0
    num_batches = 0
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]', leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            # Move batch to device
            batch = move_batch_to_device(batch, device)
            
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
            
            # Calculate loss
            predictions = output['predictions']
            loss = criterion(predictions, batch['target'])
            
            # Calculate metrics
            metrics = calculate_metrics(predictions, batch['target'])
            
            # Accumulate metrics
            total_loss += loss.item()
            total_mae += metrics['mae']
            total_rmse += metrics['rmse']
            total_r2 += metrics['r2']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mae': f"{metrics['mae']:.4f}",
                'rmse': f"{metrics['rmse']:.4f}"
            })
    
    return (
        total_loss / num_batches,
        total_mae / num_batches,
        total_rmse / num_batches,
        total_r2 / num_batches
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_path: str
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved best model: {checkpoint_path}")


def generate_forecasts(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    start_datetime: datetime,
    prediction_interval_minutes: int = 20,
    prediction_horizon_hours: int = 6,
    max_samples: int = 1000
) -> OutputList:
    """
    Generate forecasts for specified time horizon.
    
    Args:
        model: Trained LSTM model
        data_loader: DataLoader with test data
        device: Device to run inference on
        start_datetime: Starting datetime for predictions
        prediction_interval_minutes: Time interval between predictions (default 20 min)
        prediction_horizon_hours: Total forecast horizon in hours (default 6 hours)
        max_samples: Maximum number of forecast entries to generate
    
    Returns:
        OutputList containing predictions for different locations and times
    """
    model.eval()
    outputs = []
    
    # Calculate number of prediction steps
    total_minutes = prediction_horizon_hours * 60
    num_steps = total_minutes // prediction_interval_minutes
    
    print(f"\nGenerating {num_steps} forecasts at {prediction_interval_minutes}-minute intervals...")
    print(f"Forecast period: {start_datetime} to {start_datetime + timedelta(hours=prediction_horizon_hours)}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Generating forecasts")):
            # Move batch to device
            batch = move_batch_to_device(batch, device)
            
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
            
            # Get predictions (batch, pred_len)
            predictions = output['predictions']
            
            # Get location info from batch
            # static_real shape: (batch, 7) - [length, lanes, speed_limit, free_flow_speed, capacity, lat, lon]
            latitudes = batch['static_real'][:, 5].cpu().numpy()
            longitudes = batch['static_real'][:, 6].cpu().numpy()
            
            batch_size = predictions.size(0)
            pred_len = predictions.size(1)
            
            # Generate outputs for each sample in batch
            for sample_idx in range(batch_size):
                lat = float(latitudes[sample_idx])
                lon = float(longitudes[sample_idx])
                
                # Generate predictions for each time step (up to num_steps)
                for time_step in range(min(pred_len, num_steps)):
                    # Calculate datetime for this prediction
                    time_offset = timedelta(minutes=time_step * prediction_interval_minutes)
                    pred_datetime = start_datetime + time_offset
                    
                    # Get predicted congestion level
                    congestion_level = float(predictions[sample_idx, time_step].cpu().item())
                    
                    # Create output object
                    output_obj = Output(
                        DateTime=pred_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                        latitude=int(lat * 1000000),  # Convert to integer representation
                        longitude=int(lon * 1000000),
                        predicted_congestion_level=congestion_level
                    )
                    outputs.append(output_obj)
                    
                    # Limit total outputs
                    if len(outputs) >= max_samples:
                        break
                
                if len(outputs) >= max_samples:
                    break
            
            # Limit total outputs to avoid generating too many predictions
            if len(outputs) >= max_samples:
                break
    
    return OutputList(outputs=outputs)


def main():
    """Main training function."""
    # ============== Configuration ==============
    config = {
        'csv_path': 'data/traffic_synthetic.csv',
        'encoder_length': 48,
        'prediction_length': 18,
        'batch_size': 64,
        'num_epochs': 4,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,  # L2 regularization
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'checkpoint_dir': 'checkpoints',
        'early_stopping_patience': 10,
        
        # Time-based split parameters (504 time steps total: 0-503)
        'train_time_max': 353,      # 70% for training (0-353)
        'val_time_min': 354,         # 15% for validation (354-428)
        'val_time_max': 428,
        'test_time_min': 429,        # 15% for testing (429-503)
        'test_time_max': 503,
        
        # Forecast generation parameters
        'forecast_start_datetime': '2024-01-01 00:00:00',
        'forecast_interval_minutes': 20,
        'forecast_horizon_hours': 6
    }
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config['checkpoint_dir'], 'lstm_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Saved configuration to {config_path}")
    
    # ============== Create Data Loaders ==============
    print("\nCreating data loaders...")
    
    # Training data loader
    train_loader = create_data_loader(
        csv_path=config['csv_path'],
        batch_size=config['batch_size'],
        encoder_length=config['encoder_length'],
        prediction_length=config['prediction_length'],
        shuffle=True,
        device=device,
        time_idx_min=0,
        time_idx_max=config['train_time_max']
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    
    # Validation data loader
    val_loader = create_data_loader(
        csv_path=config['csv_path'],
        batch_size=config['batch_size'],
        encoder_length=config['encoder_length'],
        prediction_length=config['prediction_length'],
        shuffle=False,
        device=device,
        time_idx_min=config['val_time_min'],
        time_idx_max=config['val_time_max']
    )
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Test data loader
    test_loader = create_data_loader(
        csv_path=config['csv_path'],
        batch_size=config['batch_size'],
        encoder_length=config['encoder_length'],
        prediction_length=config['prediction_length'],
        shuffle=False,
        device=device,
        time_idx_min=config['test_time_min'],
        time_idx_max=config['test_time_max']
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # ============== Create Model ==============
    print("\nCreating LSTM Seq2Seq model...")
    
    # Get dataset info
    dataset = train_loader.dataset
    num_edges = dataset.num_edges
    num_road_types = dataset.num_road_types
    num_nodes = dataset.num_nodes
    
    print(f"Number of edges: {num_edges}")
    print(f"Number of road types: {num_road_types}")
    print(f"Number of nodes: {num_nodes}")
    
    model = create_lstm_model(
        num_edges=num_edges,
        num_road_types=num_road_types,
        num_nodes=num_nodes,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        encoder_length=config['encoder_length'],
        prediction_length=config['prediction_length']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ============== Setup Training ==============
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_mae': [],
        'train_rmse': [],
        'train_r2': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # ============== Training Loop ==============
    print(f"\nStarting training for up to {config['num_epochs']} epochs...")
    print(f"Initial learning rate: {config['learning_rate']}")
    print(f"Early stopping patience: {config['early_stopping_patience']} epochs")
    
    epoch_pbar = tqdm(range(1, config['num_epochs'] + 1), desc='Training Progress')
    
    for epoch in epoch_pbar:
        # Train
        train_loss, train_mae, train_rmse, train_r2 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_mae, val_rmse, val_r2 = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_rmse'].append(train_rmse)
        history['train_r2'].append(train_r2)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        history['lr'].append(current_lr)
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'train_loss': f"{train_loss:.4f}",
            'val_loss': f"{val_loss:.4f}",
            'val_mae': f"{val_mae:.4f}",
            'lr': f"{current_lr:.2e}"
        })
        
        # Check if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Save best model
            best_model_path = os.path.join(config['checkpoint_dir'], 'lstm_best_model.pth')
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss,
                best_model_path
            )
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= config['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"No improvement for {config['early_stopping_patience']} consecutive epochs")
            break
    
    # ============== Load Best Model ==============
    print("\nLoading best model for final evaluation...")
    best_model_path = os.path.join(config['checkpoint_dir'], 'lstm_best_model.pth')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    # ============== Test Evaluation ==============
    print("\nEvaluating on test set...")
    test_loss, test_mae, test_rmse, test_r2 = validate_epoch(
        model, test_loader, criterion, device, epoch='Test'
    )
    
    print(f"\nFinal Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': best_val_loss
    }
    test_results_path = os.path.join(config['checkpoint_dir'], 'lstm_test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    
    # Save training history
    history_path = os.path.join(config['checkpoint_dir'], 'lstm_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    # ============== Generate Forecasts ==============
    print("\n" + "="*60)
    print("GENERATING 6-HOUR FORECASTS AT 20-MINUTE INTERVALS")
    print("="*60)
    
    # Parse start datetime
    start_datetime = datetime.strptime(config['forecast_start_datetime'], "%Y-%m-%d %H:%M:%S")
    
    # Generate forecasts using test data
    forecasts = generate_forecasts(
        model=model,
        data_loader=test_loader,
        device=device,
        start_datetime=start_datetime,
        prediction_interval_minutes=config['forecast_interval_minutes'],
        prediction_horizon_hours=config['forecast_horizon_hours'],
        max_samples=1000
    )
    
    # Save forecasts as JSON
    forecasts_path = os.path.join(config['checkpoint_dir'], 'lstm_forecasts.json')
    with open(forecasts_path, 'w') as f:
        # Use the Pydantic model to serialize
        json.dump(forecasts.model_dump(), f, indent=2)
    
    print(f"\nGenerated {len(forecasts.outputs)} forecast entries")
    print(f"Forecasts saved to: {forecasts_path}")
    
    # Print sample forecasts
    print("\nSample forecasts (first 5):")
    for i, forecast in enumerate(forecasts.outputs[:5]):
        print(f"  {i+1}. {forecast.DateTime} | "
              f"Lat: {forecast.latitude}, Lon: {forecast.longitude} | "
              f"Congestion: {forecast.predicted_congestion_level:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"All artifacts saved to: {config['checkpoint_dir']}")


if __name__ == "__main__":
    main()
