"""
Temporal Fusion Transformer (TFT) for Multi-Horizon Traffic Congestion Forecasting

This implementation follows the TFT architecture for time series forecasting,
adapted for traffic congestion prediction with graph-derived neighbor features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class TimeDistributed(nn.Module):
    """Applies a module over the time dimension."""
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        batch_size, time_steps, features = x.size()
        x_reshape = x.contiguous().view(batch_size * time_steps, features)
        y = self.module(x_reshape)
        return y.view(batch_size, time_steps, -1)


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) activation."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        return out[..., :self.output_dim] * torch.sigmoid(out[..., self.output_dim:])


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - Core building block of TFT.
    
    Applies non-linear processing with gating mechanism and optional context.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
        batch_first: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        
        # Primary fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Context projection if provided
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_fc = None
        
        # ELU activation
        self.elu = nn.ELU()
        
        # Secondary fully connected layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Gated Linear Unit
        self.glu = GatedLinearUnit(output_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Skip connection projection if dimensions differ
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Store residual
        if self.skip_proj is not None:
            residual = self.skip_proj(x)
        else:
            residual = x
        
        # Primary non-linear layer
        hidden = self.fc1(x)
        
        # Add context if provided
        if self.context_fc is not None and context is not None:
            # Expand context to match hidden dimensions if needed
            if context.dim() == 2 and hidden.dim() == 3:
                context = context.unsqueeze(1).expand(-1, hidden.size(1), -1)
            hidden = hidden + self.context_fc(context)
        
        # ELU activation
        hidden = self.elu(hidden)
        
        # Secondary layer
        hidden = self.fc2(hidden)
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # GLU gating
        gated = self.glu(hidden)
        
        # Add residual and normalize
        output = self.layer_norm(gated + residual)
        
        return output


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) for feature importance.
    
    Learns to weight the importance of different input features.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        
        # Per-variable GRNs for feature transformation
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                context_dim=None
            )
            for _ in range(num_inputs)
        ])
        
        # GRN for computing variable weights
        self.weight_grn = GatedResidualNetwork(
            input_dim=input_dim * num_inputs,
            hidden_dim=hidden_dim,
            output_dim=num_inputs,
            dropout=dropout,
            context_dim=context_dim
        )
        
        # Softmax for weight normalization
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self,
        inputs: List[torch.Tensor],
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: List of tensors, each of shape (batch, time, input_dim) or (batch, input_dim)
            context: Optional context tensor of shape (batch, context_dim)
        
        Returns:
            combined: Weighted combination of transformed inputs
            weights: Variable selection weights
        """
        # Concatenate all inputs
        if inputs[0].dim() == 3:
            # Time-varying inputs: (batch, time, features)
            flattened = torch.cat(inputs, dim=-1)
        else:
            # Static inputs: (batch, features)
            flattened = torch.cat(inputs, dim=-1)
        
        # Compute variable weights
        weights = self.weight_grn(flattened, context)
        weights = self.softmax(weights)
        
        # Transform each variable
        transformed = []
        for i, (var_grn, inp) in enumerate(zip(self.var_grns, inputs)):
            transformed.append(var_grn(inp))
        
        # Stack transformed variables: (batch, [time,] num_inputs, hidden_dim)
        if transformed[0].dim() == 3:
            stacked = torch.stack(transformed, dim=2)
            # Expand weights for broadcasting
            weights_expanded = weights.unsqueeze(-1)
        else:
            stacked = torch.stack(transformed, dim=1)
            weights_expanded = weights.unsqueeze(-1)
        
        # Weighted sum
        combined = (stacked * weights_expanded).sum(dim=-2)
        
        return combined, weights


class StaticCovariateEncoder(nn.Module):
    """
    Encodes static features into context vectors for the model.
    
    Produces four context vectors:
    - c_s: For static enrichment
    - c_e: For encoder variable selection
    - c_d: For decoder variable selection  
    - c_h, c_c: For LSTM state initialization
    """
    
    def __init__(
        self,
        num_static_categorical: int,
        num_static_real: int,
        categorical_embedding_dims: Dict[str, Tuple[int, int]],
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Embeddings for categorical static features
        self.categorical_embeddings = nn.ModuleDict({
            name: nn.Embedding(num_classes, embed_dim)
            for name, (num_classes, embed_dim) in categorical_embedding_dims.items()
        })
        
        # Calculate total embedding dimension
        total_cat_dim = sum(dim for _, dim in categorical_embedding_dims.values())
        total_static_dim = total_cat_dim + num_static_real
        
        # Linear projection for real features
        if num_static_real > 0:
            self.real_proj = nn.Linear(num_static_real, num_static_real)
        else:
            self.real_proj = None
        
        # Variable selection for static features
        self.vsn = VariableSelectionNetwork(
            input_dim=1,  # Each feature projected separately
            num_inputs=num_static_categorical + num_static_real,
            hidden_dim=hidden_dim,
            dropout=dropout,
            context_dim=None
        )
        
        # GRNs for producing context vectors
        self.grn_cs = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.grn_ce = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.grn_cd = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.grn_ch = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.grn_cc = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # Simple projection for combined input
        self.input_proj = nn.Linear(total_static_dim, hidden_dim)
    
    def forward(
        self,
        static_categorical: Dict[str, torch.Tensor],
        static_real: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            static_categorical: Dict of categorical features, each (batch,)
            static_real: Tensor of real features (batch, num_real)
        
        Returns:
            c_s, c_e, c_d, c_h, c_c: Context vectors, each (batch, hidden_dim)
        """
        # Embed categorical features
        embedded = [
            self.categorical_embeddings[name](static_categorical[name])
            for name in self.categorical_embeddings.keys()
        ]
        
        # Concatenate all embeddings
        cat_embedded = torch.cat(embedded, dim=-1)
        
        # Combine with real features
        if self.real_proj is not None:
            real_proj = self.real_proj(static_real)
            combined = torch.cat([cat_embedded, real_proj], dim=-1)
        else:
            combined = cat_embedded
        
        # Project to hidden dimension
        static_rep = self.input_proj(combined)
        
        # Generate context vectors
        c_s = self.grn_cs(static_rep)
        c_e = self.grn_ce(static_rep)
        c_d = self.grn_cd(static_rep)
        c_h = self.grn_ch(static_rep)
        c_c = self.grn_cc(static_rep)
        
        return c_s, c_e, c_d, c_h, c_c


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention mechanism.
    
    Uses additive attention for interpretability of attention weights.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len, hidden_dim)
            key: (batch, seq_len, hidden_dim)
            value: (batch, seq_len, hidden_dim)
            mask: Optional attention mask
        
        Returns:
            output: Attended values (batch, seq_len, hidden_dim)
            attention_weights: Attention weights (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.size()
        
        # Project Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        output = self.out_proj(context)
        
        return output, attention_weights


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for Multi-Horizon Traffic Congestion Forecasting.
    
    Architecture follows the paper: "Temporal Fusion Transformers for 
    Interpretable Multi-horizon Time Series Forecasting"
    """
    
    def __init__(
        self,
        # Feature configuration
        static_categorical_features: Dict[str, Tuple[int, int]],  # name -> (num_classes, embed_dim)
        num_static_real: int = 7,
        num_known_categorical: int = 3,
        num_known_real: int = 4,
        num_unknown_real: int = 8,  # includes target and graph-derived features
        
        # Model configuration
        hidden_dim: int = 64,
        num_attention_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        
        # Forecasting configuration
        encoder_length: int = 48,
        prediction_length: int = 6,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        
        # Known categorical embedding dims
        known_categorical_embedding_dims: Dict[str, Tuple[int, int]] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # Default known categorical embeddings
        if known_categorical_embedding_dims is None:
            known_categorical_embedding_dims = {
                'is_weekend': (2, 8),
                'is_holiday': (2, 8),
                'weather_condition': (10, 16)
            }
        
        # ============== Static Covariate Encoder ==============
        self.static_encoder = StaticCovariateEncoder(
            num_static_categorical=len(static_categorical_features),
            num_static_real=num_static_real,
            categorical_embedding_dims=static_categorical_features,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # ============== Temporal Feature Processing ==============
        
        # Embeddings for known categorical time-varying features
        self.known_cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(num_classes, embed_dim)
            for name, (num_classes, embed_dim) in known_categorical_embedding_dims.items()
        })
        
        # Calculate input dimensions for temporal processing
        known_cat_total = sum(dim for _, dim in known_categorical_embedding_dims.values())
        self.encoder_input_dim = known_cat_total + num_known_real + num_unknown_real
        self.decoder_input_dim = known_cat_total + num_known_real
        
        # Input projections for encoder and decoder
        self.encoder_input_proj = nn.Linear(self.encoder_input_dim, hidden_dim)
        self.decoder_input_proj = nn.Linear(self.decoder_input_dim, hidden_dim)
        
        # ============== LSTM Encoder-Decoder ==============
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # LSTM output gating
        self.encoder_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.decoder_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        
        # ============== Static Enrichment ==============
        self.static_enrichment = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
            context_dim=hidden_dim
        )
        
        # ============== Temporal Self-Attention ==============
        self.self_attention = InterpretableMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        self.attention_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # ============== Position-wise Feed-Forward ==============
        self.positionwise_grn = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 4,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # ============== Final Temporal Fusion Decoder ==============
        self.final_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # ============== Quantile Output Layers ==============
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_quantiles)
            for _ in range(prediction_length)
        ])
        
        # Store number of LSTM layers for state initialization
        self.num_lstm_layers = num_lstm_layers
        
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Creates causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0
    
    def forward(
        self,
        # Static features
        static_categorical: Dict[str, torch.Tensor],  # Each (batch,)
        static_real: torch.Tensor,  # (batch, num_static_real)
        
        # Encoder inputs (historical)
        encoder_known_categorical: Dict[str, torch.Tensor],  # Each (batch, encoder_len)
        encoder_known_real: torch.Tensor,  # (batch, encoder_len, num_known_real)
        encoder_unknown_real: torch.Tensor,  # (batch, encoder_len, num_unknown_real)
        
        # Decoder inputs (future known)
        decoder_known_categorical: Dict[str, torch.Tensor],  # Each (batch, pred_len)
        decoder_known_real: torch.Tensor  # (batch, pred_len, num_known_real)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the TFT model.
        
        Returns:
            Dictionary containing:
            - 'quantile_predictions': (batch, pred_len, num_quantiles)
            - 'attention_weights': attention scores for interpretability
        """
        batch_size = static_real.size(0)
        device = static_real.device
        
        # ============== 1. Static Covariate Encoding ==============
        c_s, c_e, c_d, c_h, c_c = self.static_encoder(static_categorical, static_real)
        
        # ============== 2. Process Temporal Features ==============
        
        # Embed known categorical features for encoder
        encoder_cat_embedded = [
            self.known_cat_embeddings[name](encoder_known_categorical[name])
            for name in self.known_cat_embeddings.keys()
        ]
        encoder_cat = torch.cat(encoder_cat_embedded, dim=-1)
        
        # Combine encoder features
        encoder_features = torch.cat([
            encoder_cat,
            encoder_known_real,
            encoder_unknown_real
        ], dim=-1)
        
        # Project encoder input
        encoder_input = self.encoder_input_proj(encoder_features)
        
        # Embed known categorical features for decoder
        decoder_cat_embedded = [
            self.known_cat_embeddings[name](decoder_known_categorical[name])
            for name in self.known_cat_embeddings.keys()
        ]
        decoder_cat = torch.cat(decoder_cat_embedded, dim=-1)
        
        # Combine decoder features
        decoder_features = torch.cat([decoder_cat, decoder_known_real], dim=-1)
        
        # Project decoder input
        decoder_input = self.decoder_input_proj(decoder_features)
        
        # ============== 3. LSTM Encoder ==============
        # Initialize LSTM states with static context
        h_0 = c_h.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)
        c_0 = c_c.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)
        
        encoder_output, (h_n, c_n) = self.lstm_encoder(encoder_input, (h_0, c_0))
        
        # Apply gating and add residual
        encoder_gated = self.encoder_gate(encoder_output)
        encoder_output = self.encoder_norm(encoder_gated + encoder_input)
        
        # ============== 4. LSTM Decoder ==============
        decoder_output, _ = self.lstm_decoder(decoder_input, (h_n, c_n))
        
        # Apply gating and add residual
        decoder_gated = self.decoder_gate(decoder_output)
        decoder_output = self.decoder_norm(decoder_gated + decoder_input)
        
        # ============== 5. Concatenate Encoder and Decoder ==============
        temporal_features = torch.cat([encoder_output, decoder_output], dim=1)
        
        # ============== 6. Static Enrichment ==============
        enriched = self.static_enrichment(temporal_features, c_s)
        
        # ============== 7. Temporal Self-Attention ==============
        total_len = self.encoder_length + self.prediction_length
        mask = self._create_causal_mask(total_len, device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
        
        attention_output, attention_weights = self.self_attention(
            enriched, enriched, enriched, mask
        )
        
        # Gated residual
        attention_gated = self.attention_gate(attention_output)
        attention_output = self.attention_norm(attention_gated + enriched)
        
        # ============== 8. Position-wise Feed-Forward ==============
        positionwise_output = self.positionwise_grn(attention_output)
        
        # ============== 9. Final Gated Residual ==============
        final_gated = self.final_gate(positionwise_output)
        final_output = self.final_norm(final_gated + temporal_features)
        
        # ============== 10. Extract Decoder Positions and Generate Quantiles ==============
        # Only use decoder positions for predictions
        decoder_positions = final_output[:, self.encoder_length:, :]
        
        # Generate quantile predictions for each horizon
        quantile_predictions = []
        for t, output_layer in enumerate(self.output_layers):
            pred = output_layer(decoder_positions[:, t, :])
            quantile_predictions.append(pred)
        
        # Stack: (batch, pred_len, num_quantiles)
        quantile_predictions = torch.stack(quantile_predictions, dim=1)
        
        return {
            'quantile_predictions': quantile_predictions,
            'attention_weights': attention_weights
        }
    
    def get_multi_horizon_predictions(
        self,
        output: Dict[str, torch.Tensor],
        horizons: List[int] = [2, 4, 6]
    ) -> Dict[int, torch.Tensor]:
        """
        Extract predictions at specific horizons.
        
        Args:
            output: Forward pass output dictionary
            horizons: List of horizon indices (1-indexed, e.g., [2, 4, 6] for t+2h, t+4h, t+6h)
        
        Returns:
            Dictionary mapping horizon to predictions (batch, num_quantiles)
        """
        predictions = output['quantile_predictions']
        result = {}
        for h in horizons:
            # Convert 1-indexed horizon to 0-indexed
            idx = h - 1
            if idx < predictions.size(1):
                result[h] = predictions[:, idx, :]
        return result


class QuantileLoss(nn.Module):
    """Quantile Loss for multi-horizon forecasting."""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(
        self,
        predictions: torch.Tensor,  # (batch, pred_len, num_quantiles)
        targets: torch.Tensor  # (batch, pred_len)
    ) -> torch.Tensor:
        """Compute quantile loss."""
        losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = predictions[:, :, i]
            errors = targets - pred_q
            loss_q = torch.max(q * errors, (q - 1) * errors)
            losses.append(loss_q)
        
        # Average over all quantiles
        loss = torch.stack(losses, dim=-1).mean()
        return loss


def create_tft_model(
    num_edges: int = 100,
    num_road_types: int = 10,
    num_nodes: int = 200,
    hidden_dim: int = 32,  # Reduced from 64 to prevent overfitting
    encoder_length: int = 48,
    prediction_length: int = 18
) -> TemporalFusionTransformer:
    """
    Factory function to create a TFT model for traffic forecasting.
    
    Args:
        num_edges: Number of unique edge IDs
        num_road_types: Number of road type categories
        num_nodes: Number of unique node IDs
        hidden_dim: Hidden dimension of the model
        encoder_length: Length of encoder sequence
        prediction_length: Length of prediction horizon
    
    Returns:
        Configured TFT model
    """
    # Smaller embeddings to reduce model capacity
    static_categorical_features = {
        'edge_id': (num_edges, 16),  # Reduced from 32
        'road_type': (num_road_types, 8),  # Reduced from 16
        'start_node_id': (num_nodes, 8),  # Reduced from 16
        'end_node_id': (num_nodes, 8)  # Reduced from 16
    }
    
    known_categorical_embedding_dims = {
        'is_weekend': (2, 4),  # Reduced from 8
        'is_holiday': (2, 4),  # Reduced from 8
        'weather_condition': (10, 8)  # Reduced from 16
    }
    
    model = TemporalFusionTransformer(
        static_categorical_features=static_categorical_features,
        num_static_real=7,  # road_length, lane_count, speed_limit, free_flow_speed, capacity, lat, lon
        num_known_categorical=3,  # is_weekend, is_holiday, weather_condition
        num_known_real=4,  # hour_of_day, day_of_week, visibility, event_impact_score
        num_unknown_real=8,  # avg_speed, vehicle_count, travel_time, congestion + 4 graph features
        hidden_dim=hidden_dim,
        num_attention_heads=1,  # Reduced from 4 to single head
        num_lstm_layers=1,  # Reduced from 2 to single layer
        dropout=0.3,  # Increased from 0.1 for stronger regularization
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        quantiles=[0.1, 0.5, 0.9],
        known_categorical_embedding_dims=known_categorical_embedding_dims
    )
    
    return model
