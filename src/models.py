import torch
import torch.nn as nn
import torch.nn.functional as F


class UniTrackFormer(nn.Module):
    def __init__(
        self,
        input_dim=27,
        model_dim=128,
        n_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=3,
        num_queries=64,
    ):
        super().__init__()

        self.num_queries = num_queries

        self.max_hits = 1000

        self.hit_filter = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, model_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.queries = nn.Parameter(torch.randn(num_queries, model_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=n_heads, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.cls_head = nn.Linear(model_dim, 1) 
        self.param_head = nn.Linear(model_dim, 6)  

        self.mask_head = nn.Linear(model_dim, model_dim)
        self.hit_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x):  # x: [N_hits, D]
        N_hits = x.shape[0]
        scores = self.hit_filter(x).squeeze(-1)
        K = min(self.max_hits, N_hits)
        topk_idx = torch.topk(scores, K).indices
        x = x[topk_idx]
        N_hits = x.shape[0]

        x = self.input_proj(x)
        hits_encoded = self.encoder(x.unsqueeze(0))  

        memory = hits_encoded 
        queries = self.queries.unsqueeze(0) 

        # Decoder
        track_tokens = self.decoder(queries, memory)  
        track_tokens = track_tokens.squeeze(0) 

        track_logits = self.cls_head(track_tokens).squeeze(-1)  

        track_params = self.param_head(track_tokens)

        query_feat = self.mask_head(track_tokens)
        hit_feat = self.hit_proj(hits_encoded.squeeze(0))
        mask_logits = torch.matmul(query_feat, hit_feat.T)

        return {
            "track_logits": track_logits, 
            "hit_assignment": mask_logits,
            "track_properties": track_params,
            "topk_idx": topk_idx,
        }
