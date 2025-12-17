"""
TrackFormer: Transformers for Charged Particle Track Reconstruction in High Energy Physics
Based on the paper: arXiv:2411.07149v1

This implementation follows the exact architecture described in the paper:
1. Hit Filtering Network with φ-locality sliding window attention
2. Track Reconstruction Model based on MaskFormer architecture
3. Three task heads: classification, mask assignment, and parameter regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for cylindrical coordinates with cyclic encoding for φ"""
    
    def __init__(self, d_model: int, max_len: int = 100000):
        super().__init__()
        self.d_model = d_model
        
        # Standard sinusoidal encoding for r, z
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input embeddings [N, d_model]
            coords: cylindrical coordinates [N, 3] (r, phi, z)
        """
        # Apply cyclic positional encoding for φ
        r, phi, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        # Normalize coordinates for encoding
        r_norm = (r - r.min()) / (r.max() - r.min() + 1e-8)
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        
        # Cyclic encoding for φ
        phi_enc = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)
        
        # Add positional information
        pos_enc = torch.zeros_like(x)
        pos_enc[:, :2] = phi_enc
        pos_enc[:, 2:4] = torch.stack([r_norm, z_norm], dim=-1)
        
        return x + pos_enc


class SlidingWindowAttention(nn.Module):
    """Sliding window attention for φ-locality as described in the paper"""
    
    def __init__(self, d_model: int, n_heads: int, window_size: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, phi_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor [N, d_model]
            phi_indices: sorted indices by φ coordinate [N]
        """
        N, d_model = x.shape
        
        # Adapt window size to sequence length
        actual_window_size = min(self.window_size, N)
        w_half = actual_window_size // 2
        
        # If sequence is too short, use regular attention
        if N <= actual_window_size:
            Q = self.q_proj(x).reshape(N, self.n_heads, self.head_dim)
            K = self.k_proj(x).reshape(N, self.n_heads, self.head_dim)
            V = self.v_proj(x).reshape(N, self.n_heads, self.head_dim)
            
            # Global attention for short sequences
            attn_scores = torch.einsum('nhd,mhd->nhm', Q, K)
            attn_scores = attn_scores / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)
            output = torch.einsum('nhm,mhd->nhd', attn_weights, V)
            output_flat = output.reshape(N, d_model)
            return self.out_proj(output_flat)
        
        # Reorder by φ
        x_ordered = x[phi_indices]
        
        # Add boundary padding for cyclic attention
        x_padded = torch.cat([
            x_ordered[-w_half:],  # Last w/2 hits at beginning
            x_ordered,
            x_ordered[:w_half]   # First w/2 hits at end
        ], dim=0)
        
        # Project to Q, K, V
        Q = self.q_proj(x_padded).reshape(-1, self.n_heads, self.head_dim)
        K = self.k_proj(x_padded).reshape(-1, self.n_heads, self.head_dim)
        V = self.v_proj(x_padded).reshape(-1, self.n_heads, self.head_dim)
        
        # Apply sliding window attention
        output = torch.zeros_like(Q)
        for i in range(w_half, N + w_half):
            start = i - w_half
            end = i + w_half + 1
            
            q_i = Q[i:i+1]  # [1, n_heads, head_dim]
            k_window = K[start:end]  # [window_size, n_heads, head_dim]
            v_window = V[start:end]
            
            # Attention computation
            attn_scores = torch.einsum('bhd,whd->bhw', q_i, k_window)
            attn_scores = attn_scores / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            attn_output = torch.einsum('bhw,whd->bhd', attn_weights, v_window)
            output[i] = attn_output.squeeze(0)
        
        # Remove padding and restore original order
        output = output[w_half:w_half+N]
        output_unordered = torch.zeros_like(output)
        output_unordered[phi_indices] = output
        
        # Final projection
        output_flat = output_unordered.reshape(N, d_model)
        return self.out_proj(output_flat)


class SwiGLU(nn.Module):
    """SwiGLU activation function as used in the paper"""
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerEncoderLayer(nn.Module):
    """Custom transformer encoder layer with sliding window attention"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, window_size: int = 1024):
        super().__init__()
        self.self_attn = SlidingWindowAttention(d_model, n_heads, window_size)
        self.feed_forward = SwiGLU(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, phi_indices: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.self_attn(self.norm1(x), phi_indices)
        x = x + attn_output
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output
        
        return x


class HitFilteringNetwork(nn.Module):
    """Hit filtering network as described in Section 3.2"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 12,
        window_size: int = 1024
    ):
        super().__init__()
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, 2 * d_model, window_size)
            for _ in range(n_layers)
        ])
        
        # Dense classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )
        
    def forward(self, hits: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hits: hit features [N, input_dim]
            coords: cylindrical coordinates [N, 3] (r, phi, z)
        
        Returns:
            hit_scores: classification scores [N]
        """
        # Sort by φ for sliding window attention
        phi = coords[:, 1]
        phi_indices = torch.argsort(phi)
        
        # Input embedding and positional encoding
        x = self.input_embedding(hits)
        x = self.pos_encoding(x, coords)
        
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, phi_indices)
        
        # Classification
        hit_scores = self.classifier(x).squeeze(-1)
        
        return hit_scores


class MaskAttention(nn.Module):
    """Mask attention mechanism from MaskFormer"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
    def forward(
        self, 
        query_embed: torch.Tensor, 
        key_embed: torch.Tensor, 
        value_embed: torch.Tensor,
        mask_proposals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query_embed: [N_queries, d_model]
            key_embed: [N_hits, d_model]
            value_embed: [N_hits, d_model]
            mask_proposals: [N_queries, N_hits] optional attention masks
        """
        # Standard attention computation
        attn_scores = torch.matmul(query_embed, key_embed.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.d_model)
        
        # Apply mask if provided
        if mask_proposals is not None:
            attn_scores = attn_scores + mask_proposals.log()
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, value_embed)
        
        return output


class ObjectDecoderLayer(nn.Module):
    """Object decoder layer for MaskFormer-based track reconstruction"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        
        # Self-attention for object queries
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Cross-attention for object-hit interaction
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Mask attention
        self.mask_attn = MaskAttention(d_model)
        
        # Feed-forward network
        self.feed_forward = SwiGLU(d_model, d_ff)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        queries: torch.Tensor, 
        hit_embeddings: torch.Tensor,
        mask_proposals: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: object queries [N_queries, d_model]
            hit_embeddings: encoded hit features [N_hits, d_model]
            mask_proposals: previous mask proposals [N_queries, N_hits]
        
        Returns:
            updated_queries: [N_queries, d_model]
            new_mask_proposals: [N_queries, N_hits]
        """
        # Self-attention among object queries
        queries_norm = self.norm1(queries)
        self_attn_output, _ = self.self_attn(
            queries_norm.unsqueeze(0), 
            queries_norm.unsqueeze(0), 
            queries_norm.unsqueeze(0)
        )
        queries = queries + self_attn_output.squeeze(0)
        
        # Cross-attention with hits using mask attention
        queries_norm = self.norm2(queries)
        cross_attn_output = self.mask_attn(
            queries_norm, hit_embeddings, hit_embeddings, mask_proposals
        )
        queries = queries + cross_attn_output
        
        # Feed-forward
        queries_norm = self.norm3(queries)
        ff_output = self.feed_forward(queries_norm)
        queries = queries + ff_output
        
        # Generate new mask proposals
        mask_tokens = queries  # Use updated queries as mask tokens
        new_mask_proposals = torch.matmul(mask_tokens, hit_embeddings.transpose(-2, -1))
        new_mask_proposals = torch.sigmoid(new_mask_proposals)
        
        return queries, new_mask_proposals


class TrackReconstructionModel(nn.Module):
    """Track reconstruction model based on MaskFormer (Section 3.3)"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 12,
        n_decoder_layers: int = 8,
        n_queries: int = 2100,
        window_size: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries
        
        # Encoder (same as hit filtering but with smaller window)
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, 2 * d_model, window_size)
            for _ in range(n_encoder_layers)
        ])
        
        # Object queries
        self.object_queries = nn.Parameter(torch.randn(n_queries, d_model))
        
        # Object decoder
        self.decoder_layers = nn.ModuleList([
            ObjectDecoderLayer(d_model, n_heads, 2 * d_model)
            for _ in range(n_decoder_layers)
        ])
        
        # Task heads
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        self.mask_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model)
        )
        
        self.param_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 6)  # vx, vy, vz, px, py, pz
        )
        
        # Hit projection for mask computation
        self.hit_proj = nn.Linear(d_model, d_model)
        
    def forward(self, hits: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hits: filtered hit features [N_filtered, input_dim]
            coords: cylindrical coordinates [N_filtered, 3]
        
        Returns:
            dict with keys: track_logits, hit_assignment, track_properties
        """
        # Sort by φ for sliding window attention
        phi = coords[:, 1]
        phi_indices = torch.argsort(phi)
        
        # Encoder
        x = self.input_embedding(hits)
        x = self.pos_encoding(x, coords)
        
        for layer in self.encoder_layers:
            x = layer(x, phi_indices)
        
        hit_embeddings = x
        
        # Object decoder
        queries = self.object_queries.unsqueeze(0).expand(1, -1, -1).squeeze(0)
        mask_proposals = None
        
        # Store intermediate outputs for auxiliary loss
        intermediate_outputs = []
        
        for layer in self.decoder_layers:
            queries, mask_proposals = layer(queries, hit_embeddings, mask_proposals)
            
            # Compute intermediate predictions for auxiliary loss
            intermediate_class = self.class_head(queries).squeeze(-1)
            intermediate_params = self.param_head(queries)
            
            # Mask computation
            mask_tokens = self.mask_head(queries)
            hit_features = self.hit_proj(hit_embeddings)
            intermediate_masks = torch.matmul(mask_tokens, hit_features.transpose(-2, -1))
            
            intermediate_outputs.append({
                'track_logits': intermediate_class,
                'track_properties': intermediate_params,
                'hit_assignment': intermediate_masks
            })
        
        # Final predictions
        track_logits = self.class_head(queries).squeeze(-1)
        track_properties = self.param_head(queries)
        
        # Final mask computation
        mask_tokens = self.mask_head(queries)
        hit_features = self.hit_proj(hit_embeddings)
        hit_assignment = torch.matmul(mask_tokens, hit_features.transpose(-2, -1))
        
        return {
            'track_logits': track_logits,
            'hit_assignment': hit_assignment,
            'track_properties': track_properties,
            'intermediate_outputs': intermediate_outputs
        }


class TrackFormer(nn.Module):
    """
    Complete TrackFormer model combining hit filtering and track reconstruction
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        hit_filter_layers: int = 12,
        track_encoder_layers: int = 12,
        track_decoder_layers: int = 8,
        n_queries: int = 2100,
        hit_filter_window: int = 1024,
        track_window: int = 512,
        filter_threshold: float = 0.001
    ):
        super().__init__()
        self.filter_threshold = filter_threshold
        
        # Hit filtering network
        self.hit_filter = HitFilteringNetwork(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=hit_filter_layers,
            window_size=hit_filter_window
        )
        
        # Track reconstruction model
        self.track_reconstructor = TrackReconstructionModel(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=track_encoder_layers,
            n_decoder_layers=track_decoder_layers,
            n_queries=n_queries,
            window_size=track_window
        )
        
    def forward(self, hits: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hits: hit features [N_hits, input_dim]
            coords: cylindrical coordinates [N_hits, 3] (r, phi, z)
        
        Returns:
            dict with filtering and tracking results
        """
        # Hit filtering
        hit_scores = self.hit_filter(hits, coords)
        hit_probs = torch.sigmoid(hit_scores)
        
        # Apply filtering threshold
        keep_mask = hit_probs > self.filter_threshold
        filtered_hits = hits[keep_mask]
        filtered_coords = coords[keep_mask]
        
        if filtered_hits.shape[0] == 0:
            # No hits passed filtering
            return {
                'hit_scores': hit_scores,
                'filtered_indices': keep_mask,
                'track_logits': torch.empty(0),
                'hit_assignment': torch.empty(0, 0),
                'track_properties': torch.empty(0, 4)
            }
        
        # Track reconstruction
        track_results = self.track_reconstructor(filtered_hits, filtered_coords)
        
        # Combine results
        return {
            'hit_scores': hit_scores,
            'filtered_indices': keep_mask,
            **track_results
        }


def create_trackformer_600mev(input_dim: int) -> TrackFormer:
    """Create TrackFormer model for 600 MeV configuration"""
    return TrackFormer(
        input_dim=input_dim,
        d_model=256,
        n_heads=8,
        hit_filter_layers=12,
        track_encoder_layers=12,
        track_decoder_layers=8,
        n_queries=2100,
        hit_filter_window=1024,
        track_window=512,
        filter_threshold=0.1
    )


def create_trackformer_750mev(input_dim: int) -> TrackFormer:
    """Create TrackFormer model for 750 MeV configuration"""
    return TrackFormer(
        input_dim=input_dim,
        d_model=256,
        n_heads=8,
        hit_filter_layers=12,
        track_encoder_layers=12,
        track_decoder_layers=8,
        n_queries=1800,
        hit_filter_window=1024,
        track_window=512,
        filter_threshold=0.1
    )


def create_trackformer_1gev(input_dim: int) -> TrackFormer:
    """Create TrackFormer model for 1 GeV configuration"""
    return TrackFormer(
        input_dim=input_dim,
        d_model=256,
        n_heads=8,
        hit_filter_layers=8,  # Paper mentions 8 layers for 1GeV model
        track_encoder_layers=12,
        track_decoder_layers=8,
        n_queries=1100,
        hit_filter_window=1024,
        track_window=512,
        filter_threshold=0.1
    )
