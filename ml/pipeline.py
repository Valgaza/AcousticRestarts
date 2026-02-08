import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Union, Optional, List, Dict

import torch

from models.custom_temporal_transformer import create_tft_model
from outputs_format import Output, OutputList


def get_workspace_root() -> Path:
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "backend").exists() and (current / "ml").exists():
            return current
        current = current.parent
    return Path.cwd()


def find_latest_csv(uploads_dir: Optional[str] = None) -> Optional[str]:
    if uploads_dir is None:
        workspace_root = get_workspace_root()
        upload_path = workspace_root / "backend" / "uploads"
    else:
        upload_path = Path(uploads_dir)
    
    if not upload_path.exists():
        print(f"Upload directory does not exist: {upload_path}")
        return None
    
    csv_files = list(upload_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {upload_path}")
        return None
    
    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Found CSV: {latest}")
    return str(latest)

def prepare_inference_batch(
    df: pd.DataFrame,
    encoder_length: int,
    prediction_length: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    
    road_type_map = {'Residential': 0, 'Ramp': 1, 'Highway': 2, 'Arterial': 3}
    weather_map = {'Clear': 0, 'Rain': 1, 'Fog': 2}
    
    df['road_type_encoded'] = df['road_type'].map(road_type_map).fillna(0)
    df['weather_encoded'] = df['weather_condition'].map(weather_map).fillna(0)
    df['is_weekend'] = df['is_weekend'].astype(int)
    df['is_holiday'] = df['is_holiday'].astype(int)
    
    df = df.sort_values(['edge_id', 'time_idx']).reset_index(drop=True)
    
    edge_groups = df.groupby('edge_id')
    batches = []
    
    # Check if congestion-related columns exist
    has_congestion = 'congestion_level' in df.columns
    has_neighbor_congestion = 'neighbor_avg_congestion_t-1' in df.columns
    has_neighbor_speed = 'neighbor_avg_speed_t-1' in df.columns
    has_upstream = 'upstream_congestion_t-1' in df.columns
    has_downstream = 'downstream_congestion_t-1' in df.columns
    
    for edge_id, group in edge_groups:
        if len(group) < encoder_length:
            continue
        
        encoder_data = group.iloc[:encoder_length]
        static_row = encoder_data.iloc[0]
        
        static_categorical = {
            'edge_id': torch.tensor(static_row['edge_id'], dtype=torch.long),
            'road_type': torch.tensor(static_row['road_type_encoded'], dtype=torch.long),
            'start_node_id': torch.tensor(static_row['start_node_id'], dtype=torch.long),
            'end_node_id': torch.tensor(static_row['end_node_id'], dtype=torch.long)
        }
        
        static_real = torch.tensor([
            static_row['road_length_meters'],
            static_row['lane_count'],
            static_row['speed_limit_kph'],
            static_row['free_flow_speed_kph'],
            static_row['road_capacity'],
            static_row['latitude_midroad'],
            static_row['longitude_midroad']
        ], dtype=torch.float32)
        
        encoder_known_categorical = {
            'is_weekend': torch.tensor(encoder_data['is_weekend'].values, dtype=torch.long),
            'is_holiday': torch.tensor(encoder_data['is_holiday'].values, dtype=torch.long),
            'weather_condition': torch.tensor(encoder_data['weather_encoded'].values, dtype=torch.long)
        }
        
        encoder_known_real = torch.stack([
            torch.tensor(encoder_data['hour_of_day'].values, dtype=torch.float32),
            torch.tensor(encoder_data['day_of_week'].values, dtype=torch.float32),
            torch.tensor(encoder_data['visibility'].values, dtype=torch.float32),
            torch.tensor(encoder_data['event_impact_score'].values, dtype=torch.float32)
        ], dim=1)
        
        # Handle missing congestion columns - use zeros as placeholders
        encoder_unknown_real = torch.stack([
            torch.tensor(encoder_data['average_speed_kph'].values, dtype=torch.float32),
            torch.tensor(encoder_data['vehicle_count'].values, dtype=torch.float32),
            torch.tensor(encoder_data['travel_time_seconds'].values, dtype=torch.float32),
            torch.tensor(encoder_data['congestion_level'].values if has_congestion 
                        else np.zeros(len(encoder_data)), dtype=torch.float32),
            torch.tensor(encoder_data['neighbor_avg_congestion_t-1'].values if has_neighbor_congestion 
                        else np.zeros(len(encoder_data)), dtype=torch.float32),
            torch.tensor(encoder_data['neighbor_avg_speed_t-1'].values if has_neighbor_speed 
                        else np.zeros(len(encoder_data)), dtype=torch.float32),
            torch.tensor(encoder_data['upstream_congestion_t-1'].values if has_upstream 
                        else np.zeros(len(encoder_data)), dtype=torch.float32),
            torch.tensor(encoder_data['downstream_congestion_t-1'].values if has_downstream 
                        else np.zeros(len(encoder_data)), dtype=torch.float32)
        ], dim=1)
        
        last_time = encoder_data.iloc[-1]
        decoder_known_cat = {'is_weekend': [], 'is_holiday': [], 'weather_condition': []}
        decoder_known_vals = [[], [], [], []]
        
        for t in range(prediction_length):
            future_hour = (last_time['hour_of_day'] + (t + 1) * 20 / 60) % 24
            future_day = (last_time['day_of_week'] + int((last_time['hour_of_day'] + (t + 1) * 20 / 60) / 24)) % 7
            
            decoder_known_cat['is_weekend'].append(int(last_time['is_weekend']))
            decoder_known_cat['is_holiday'].append(int(last_time['is_holiday']))
            decoder_known_cat['weather_condition'].append(int(last_time['weather_encoded']))
            
            decoder_known_vals[0].append(future_hour)
            decoder_known_vals[1].append(future_day)
            decoder_known_vals[2].append(last_time['visibility'])
            decoder_known_vals[3].append(last_time.get('event_impact_score', 0))
        
        decoder_known_categorical = {
            k: torch.tensor(v, dtype=torch.long) for k, v in decoder_known_cat.items()
        }
        decoder_known_real = torch.tensor(decoder_known_vals, dtype=torch.float32).T
        
        batches.append({
            'static_categorical': static_categorical,
            'static_real': static_real,
            'encoder_known_categorical': encoder_known_categorical,
            'encoder_known_real': encoder_known_real,
            'encoder_unknown_real': encoder_unknown_real,
            'decoder_known_categorical': decoder_known_categorical,
            'decoder_known_real': decoder_known_real,
            'latitude': static_row['latitude_midroad'],
            'longitude': static_row['longitude_midroad'],
            'edge_id': edge_id  # Add edge_id for tracking
        })
    
    if not batches:
        return None
    
    batch = {
        'static_categorical': {k: torch.stack([b['static_categorical'][k] for b in batches]).to(device) 
                               for k in batches[0]['static_categorical'].keys()},
        'static_real': torch.stack([b['static_real'] for b in batches]).to(device),
        'encoder_known_categorical': {k: torch.stack([b['encoder_known_categorical'][k] for b in batches]).to(device)
                                       for k in batches[0]['encoder_known_categorical'].keys()},
        'encoder_known_real': torch.stack([b['encoder_known_real'] for b in batches]).to(device),
        'encoder_unknown_real': torch.stack([b['encoder_unknown_real'] for b in batches]).to(device),
        'decoder_known_categorical': {k: torch.stack([b['decoder_known_categorical'][k] for b in batches]).to(device)
                                       for k in batches[0]['decoder_known_categorical'].keys()},
        'decoder_known_real': torch.stack([b['decoder_known_real'] for b in batches]).to(device),
        'latitudes': [b['latitude'] for b in batches],
        'longitudes': [b['longitude'] for b in batches],
        'edge_ids': [b['edge_id'] for b in batches]  # Include edge IDs
    }
    
    return batch

def infer_from_csv(
    csv_source: Union[str, bytes, None] = None,
    model_checkpoint_path: Optional[str] = None,
    output_json_path: Optional[str] = None,
    encoder_length: int = 48,
    prediction_length: int = 18,
    device: Optional[torch.device] = None,
    forecast_start_datetime: Optional[str] = None,
    forecast_interval_minutes: int = 20
) -> str:
    
    workspace_root = get_workspace_root()
    
    if model_checkpoint_path is None:
        model_checkpoint_path = str(workspace_root / "ml" / "checkpoints" / "best_model.pth")
    
    if output_json_path is None:
        output_json_path = str(workspace_root / "backend" / "outputs" / "forecasts.json")
    
    if csv_source is None:
        csv_path = find_latest_csv()
        if csv_path is None:
            raise FileNotFoundError("No CSV files found in backend/uploads")
    elif isinstance(csv_source, bytes):
        workspace_root = get_workspace_root()
        upload_dir = workspace_root / "backend" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        csv_path = upload_dir / f"upload_{int(datetime.utcnow().timestamp())}.csv"
        csv_path.write_bytes(csv_source)
        csv_path = str(csv_path)
    else:
        csv_path = str(csv_source)
    
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    df = pd.read_csv(csv_path)
    
    num_edges = df['edge_id'].nunique()
    num_road_types = df['road_type'].nunique()
    num_nodes = max(df['start_node_id'].max(), df['end_node_id'].max()) + 1
    
    model = create_tft_model(
        num_edges=num_edges,
        num_road_types=num_road_types,
        num_nodes=num_nodes,
        hidden_dim=32,
        encoder_length=encoder_length,
        prediction_length=prediction_length
    )
    
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    batch = prepare_inference_batch(df, encoder_length, prediction_length, device)
    if batch is None:
        raise ValueError("No valid sequences found in CSV")
    
    if forecast_start_datetime:
        start_dt = datetime.strptime(forecast_start_datetime, "%Y-%m-%d %H:%M:%S")
    else:
        start_dt = datetime.utcnow()
    
    outputs = []
    
    with torch.no_grad():
        out = model(
            static_categorical=batch['static_categorical'],
            static_real=batch['static_real'],
            encoder_known_categorical=batch['encoder_known_categorical'],
            encoder_known_real=batch['encoder_known_real'],
            encoder_unknown_real=batch['encoder_unknown_real'],
            decoder_known_categorical=batch['decoder_known_categorical'],
            decoder_known_real=batch['decoder_known_real']
        )
        
        preds = out['quantile_predictions'][:, :, 1]
        
        for i in range(preds.size(0)):
            lat = batch['latitudes'][i]
            lon = batch['longitudes'][i]
            
            for t in range(preds.size(1)):
                pred_dt = start_dt + timedelta(minutes=t * forecast_interval_minutes)
                congestion = float(preds[i, t].cpu().item())
                
                outputs.append(Output(
                    DateTime=pred_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    latitude=int(lat * 1_000_000),
                    longitude=int(lon * 1_000_000),
                    predicted_congestion_level=congestion
                ))
    
    out_list = OutputList(outputs=outputs)
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out_list.model_dump(), f, indent=2)
    
    return output_json_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TFT inference pipeline on CSV from backend/uploads")
    parser.add_argument("--csv", default=None, help="Path to CSV file (auto-finds latest in backend/uploads if not specified)")
    parser.add_argument("--model", default=None, help="Path to .pth model checkpoint")
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--start", default=None, help="Forecast start datetime, format 'YYYY-MM-DD HH:MM:SS'")
    args = parser.parse_args()
    
    workspace_root = get_workspace_root()
    model_path = args.model or str(workspace_root / "ml" / "checkpoints" / "best_model.pth")
    output_path = args.out or str(workspace_root / "backend" /  "outputs" / "forecasts.json")
    
    outpath = infer_from_csv(
        csv_source=args.csv,
        model_checkpoint_path=model_path,
        output_json_path=output_path,
        forecast_start_datetime=args.start
    )
    print(f"Saved forecasts to: {outpath}")