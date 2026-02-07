"""
LSTM Seq2Seq Baseline Model for Traffic Congestion Forecasting

A simple encoder-decoder LSTM architecture for multi-horizon traffic prediction.
This serves as a baseline for comparison with the more complex TFT model.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class LSTMSeq2Seq(nn.Module):
    """
    Simple LSTM Encoder-Decoder for traffic forecasting.
    
    Architecture:
    - Encoder: LSTM that processes historical sequence
    - Decoder: LSTM that generates future predictions
    - Input: Concatenated feature vector at each timestep
    - Output: Predicted congestion levels for future timesteps
    """
    
    def __init__(
        self,
        # Categorical feature sizes
        num_edges: int = 100,
        num_road_types: int = 10,
        num_nodes: int = 200,
        num_weather_conditions: int = 10,
        # Embedding dimensions
        edge_embedding_dim: int = 16,
        road_type_embedding_dim: int = 8,
        node_embedding_dim: int = 8,
        weather_embedding_dim: int = 8,
        categorical_time_embedding_dim: int = 8,
        # Static real features
        num_static_real: int = 7,
        # Time-varying known features
        num_known_real: int = 4,
        num_known_categorical: int = 3,  # is_weekend, is_holiday, weather
        # Time-varying unknown features (encoder only)
        num_unknown_real: int = 8,
        # Model architecture
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        # Sequence lengths
        encoder_length: int = 48,
        prediction_length: int = 18
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        
        # ============== Embeddings ==============
        # Static categorical embeddings
        self.edge_embedding = nn.Embedding(num_edges, edge_embedding_dim)
        self.road_type_embedding = nn.Embedding(num_road_types, road_type_embedding_dim)
        self.start_node_embedding = nn.Embedding(num_nodes, node_embedding_dim)
        self.end_node_embedding = nn.Embedding(num_nodes, node_embedding_dim)
        
        # Time-varying categorical embeddings
        self.is_weekend_embedding = nn.Embedding(2, categorical_time_embedding_dim)
        self.is_holiday_embedding = nn.Embedding(2, categorical_time_embedding_dim)
        self.weather_embedding = nn.Embedding(num_weather_conditions, weather_embedding_dim)
        
        # Calculate input dimensions
        # Static embeddings (used at every timestep)
        static_embedding_dim = (edge_embedding_dim + road_type_embedding_dim + 
                               2 * node_embedding_dim + num_static_real)
        
        # Time-varying embeddings
        time_varying_embedding_dim = (3 * categorical_time_embedding_dim + 
                                     num_known_real)
        
        # Encoder input: static + time-varying known + unknown
        self.encoder_input_dim = (static_embedding_dim + 
                                 time_varying_embedding_dim + 
                                 num_unknown_real)
        
        # Decoder input: static + time-varying known only (no unknown features)
        self.decoder_input_dim = static_embedding_dim + time_varying_embedding_dim
        
        # ============== Encoder ==============
        self.encoder = nn.LSTM(
            input_size=self.encoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # ============== Decoder ==============
        self.decoder = nn.LSTM(
            input_size=self.decoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # ============== Output Layer ==============
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                elif 'embedding' in name:
                    nn.init.normal_(param, mean=0.0, std=0.01)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def _embed_static_features(
        self,
        static_categorical: Dict[str, torch.Tensor],
        static_real: torch.Tensor
    ) -> torch.Tensor:
        """
        Embed static features.
        
        Args:
            static_categorical: Dict with edge_id, road_type, start_node_id, end_node_id
            static_real: (batch, num_static_real)
        
        Returns:
            (batch, static_embedding_dim)
        """
        edge_emb = self.edge_embedding(static_categorical['edge_id'])
        road_type_emb = self.road_type_embedding(static_categorical['road_type'])
        start_node_emb = self.start_node_embedding(static_categorical['start_node_id'])
        end_node_emb = self.end_node_embedding(static_categorical['end_node_id'])
        
        # Concatenate all static features
        static_features = torch.cat([
            edge_emb, road_type_emb, start_node_emb, end_node_emb, static_real
        ], dim=-1)
        
        return static_features
    
    def _embed_time_varying_features(
        self,
        known_categorical: Dict[str, torch.Tensor],
        known_real: torch.Tensor
    ) -> torch.Tensor:
        """
        Embed time-varying known features.
        
        Args:
            known_categorical: Dict with is_weekend, is_holiday, weather_condition
                              Each (batch, seq_len)
            known_real: (batch, seq_len, num_known_real)
        
        Returns:
            (batch, seq_len, time_varying_embedding_dim)
        """
        batch_size, seq_len = known_real.shape[:2]
        
        # Embed categorical features
        weekend_emb = self.is_weekend_embedding(known_categorical['is_weekend'])  # (batch, seq_len, emb_dim)
        holiday_emb = self.is_holiday_embedding(known_categorical['is_holiday'])
        weather_emb = self.weather_embedding(known_categorical['weather_condition'])
        
        # Concatenate all time-varying features
        time_varying_features = torch.cat([
            weekend_emb, holiday_emb, weather_emb, known_real
        ], dim=-1)
        
        return time_varying_features
    
    def forward(
        self,
        static_categorical: Dict[str, torch.Tensor],
        static_real: torch.Tensor,
        encoder_known_categorical: Dict[str, torch.Tensor],
        encoder_known_real: torch.Tensor,
        encoder_unknown_real: torch.Tensor,
        decoder_known_categorical: Dict[str, torch.Tensor],
        decoder_known_real: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            static_categorical: Static categorical features
            static_real: Static real features (batch, num_static_real)
            encoder_known_categorical: Encoder known categorical (batch, enc_len)
            encoder_known_real: Encoder known real (batch, enc_len, num_known_real)
            encoder_unknown_real: Encoder unknown real (batch, enc_len, num_unknown_real)
            decoder_known_categorical: Decoder known categorical (batch, pred_len)
            decoder_known_real: Decoder known real (batch, pred_len, num_known_real)
        
        Returns:
            Dictionary with 'predictions' of shape (batch, pred_len)
        """
        batch_size = static_real.size(0)
        
        # ============== Embed Features ==============
        # Static features (batch, static_dim)
        static_features = self._embed_static_features(static_categorical, static_real)
        
        # Encoder time-varying features (batch, enc_len, time_varying_dim)
        encoder_time_features = self._embed_time_varying_features(
            encoder_known_categorical, encoder_known_real
        )
        
        # Decoder time-varying features (batch, pred_len, time_varying_dim)
        decoder_time_features = self._embed_time_varying_features(
            decoder_known_categorical, decoder_known_real
        )
        
        # ============== Encoder ==============
        # Expand static features to match encoder sequence length
        static_expanded_enc = static_features.unsqueeze(1).expand(-1, self.encoder_length, -1)
        
        # Concatenate: static + time-varying + unknown
        encoder_input = torch.cat([
            static_expanded_enc,
            encoder_time_features,
            encoder_unknown_real
        ], dim=-1)  # (batch, enc_len, encoder_input_dim)
        
        # Encode
        encoder_output, (hidden, cell) = self.encoder(encoder_input)
        # hidden, cell: (num_layers, batch, hidden_dim)
        
        # ============== Decoder ==============
        # Expand static features to match decoder sequence length
        static_expanded_dec = static_features.unsqueeze(1).expand(-1, self.prediction_length, -1)
        
        # Concatenate: static + time-varying (no unknown features in decoder)
        decoder_input = torch.cat([
            static_expanded_dec,
            decoder_time_features
        ], dim=-1)  # (batch, pred_len, decoder_input_dim)
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        # decoder_output: (batch, pred_len, hidden_dim)
        
        # ============== Output ==============
        # Generate predictions for each timestep
        predictions = self.output_layer(decoder_output)  # (batch, pred_len, 1)
        predictions = predictions.squeeze(-1)  # (batch, pred_len)
        
        return {
            'predictions': predictions,
            'encoder_output': encoder_output,
            'decoder_output': decoder_output
        }
    
    def predict(
        self,
        static_categorical: Dict[str, torch.Tensor],
        static_real: torch.Tensor,
        encoder_known_categorical: Dict[str, torch.Tensor],
        encoder_known_real: torch.Tensor,
        encoder_unknown_real: torch.Tensor,
        decoder_known_categorical: Dict[str, torch.Tensor],
        decoder_known_real: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate predictions (convenience method).
        
        Returns:
            Tensor of shape (batch, pred_len) with predicted congestion levels
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                static_categorical, static_real,
                encoder_known_categorical, encoder_known_real, encoder_unknown_real,
                decoder_known_categorical, decoder_known_real
            )
        return output['predictions']


def create_lstm_model(
    num_edges: int = 100,
    num_road_types: int = 10,
    num_nodes: int = 200,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    encoder_length: int = 48,
    prediction_length: int = 18
) -> LSTMSeq2Seq:
    """
    Factory function to create an LSTM seq2seq model for traffic forecasting.
    
    Args:
        num_edges: Number of unique edge IDs
        num_road_types: Number of road type categories
        num_nodes: Number of unique node IDs
        hidden_dim: Hidden dimension of LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        encoder_length: Length of encoder sequence
        prediction_length: Length of prediction horizon
    
    Returns:
        Configured LSTM seq2seq model
    """
    model = LSTMSeq2Seq(
        num_edges=num_edges,
        num_road_types=num_road_types,
        num_nodes=num_nodes,
        num_weather_conditions=10,
        edge_embedding_dim=16,
        road_type_embedding_dim=8,
        node_embedding_dim=8,
        weather_embedding_dim=8,
        categorical_time_embedding_dim=8,
        num_static_real=7,
        num_known_real=4,
        num_known_categorical=3,
        num_unknown_real=8,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        encoder_length=encoder_length,
        prediction_length=prediction_length
    )
    
    return model
