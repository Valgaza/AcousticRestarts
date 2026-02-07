"""Quick test of the training pipeline with 2 epochs."""

import torch
from train_model import main
import sys

# Override config for quick test
if __name__ == "__main__":
    print("=" * 60)
    print("QUICK TRAINING TEST - 2 epochs with small batch")
    print("=" * 60)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Monkey patch config in the main function
    import train_model as tm
    original_main = tm.main
    
    def test_main():
        # This is a quick test version
        tm.config = {
            'csv_path': 'data/traffic_synthetic.csv',
            'encoder_length': 48,
            'prediction_length': 18,
            'batch_size': 16,  # Smaller batch for faster test
            'num_epochs': 2,  # Just 2 epochs for testing
            'learning_rate': 1e-3,
            'lr_step_size': 10,
            'lr_gamma': 0.9,
            'hidden_dim': 64,  # Smaller model for faster test
            'num_attention_heads': 4,
            'num_lstm_layers': 2,
            'dropout': 0.1,
            'checkpoint_dir': 'checkpoints_test',
            'checkpoint_every': 1,
            'num_workers': 0,
            'train_time_max': 353,
            'val_time_min': 354,
            'val_time_max': 428,
            'test_time_min': 429,
            'test_time_max': 503
        }
        original_main()
    
    try:
        test_main()
        print("\n✓ Training test completed successfully!")
    except Exception as e:
        print(f"\n✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
