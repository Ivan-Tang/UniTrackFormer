import torch
import torch.nn as nn
import torch.nn.functional as F

class UniTrackFormer(nn.Module):
    def __init__(self, 
                 input_dim=27,
                 model_dim=128,
                 n_heads=8,
                 num_encoder_layers=4,
                 num_decoder_layers=3,
                 num_queries=64):
        super().__init__()

        self.num_queries = num_queries

        self.max_hits = 1000
        
        self.hit_filter = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 输入嵌入层：将 (x, y, z, r, φ, layer_id) 映射到高维
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )

        # Transformer 编码器：编码 hits 上下文信息
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, model_dim))

        # Transformer 解码器：每个 query 聚焦 hits
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=n_heads, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 输出头
        self.cls_head = nn.Linear(model_dim, 1)          # 分类：是否是真轨迹
        self.param_head = nn.Linear(model_dim, 6)        # 参数回归：vx, vy, vz, px, py, pz

        # 掩码预测：每个 query 与 hits 交互后输出 hit 分配（mask）
        self.mask_head = nn.Linear(model_dim, model_dim)
        self.hit_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x):  # x: [N_hits, D]
        N_hits = x.shape[0]
        #计算hits的评分并且筛选出评分靠前的hits
        scores = self.hit_filter(x).squeeze(-1)
        K = min(self.max_hits, N_hits)
        topk_idx = torch.topk(scores, K).indices
        x = x[topk_idx]
        N_hits = x.shape[0]

        x = self.input_proj(x)  # [N_hits, model_dim]
        hits_encoded = self.encoder(x.unsqueeze(0))  # [1, N_hits, model_dim]

        # 重复 hits 编码，用于 decoder cross attention
        memory = hits_encoded  # [1, N_hits, model_dim]

        # Learnable queries
        queries = self.queries.unsqueeze(0)  # [1, Q, model_dim]

        # Decoder
        track_tokens = self.decoder(queries, memory)  # [1, Q, model_dim]
        track_tokens = track_tokens.squeeze(0)  # [Q, model_dim]

        # 分类头：是否是有效轨迹
        track_logits = self.cls_head(track_tokens).squeeze(-1)  # [Q]

        # 参数回归
        track_params = self.param_head(track_tokens)  # [Q, 6]

        # 掩码头：每个轨迹 query 与 hits 做点积匹配
        query_feat = self.mask_head(track_tokens)       # [Q, D]
        hit_feat = self.hit_proj(hits_encoded.squeeze(0))  # [N_hits, D]
        mask_logits = torch.matmul(query_feat, hit_feat.T)  # [Q, N_hits]

        return {
            'track_logits': track_logits,          # [Q]
            'hit_assignment': mask_logits,         # [Q, N_hits]
            'track_properties': track_params,      # [Q, 6]
            'topk_idx': topk_idx                    # [K]
        }