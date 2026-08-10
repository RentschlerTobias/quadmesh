import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from positional_encoder import RoPEPositionalEncoding


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, is_cross_attention=False, max_position=10000):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.is_cross_attention = is_cross_attention

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.rope = RoPEPositionalEncoding(self.d_k, max_seq_len=max_position)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, position_ids=None):
        """
        Kompatibel mit HourglassTransformer (erwartet position_ids als Keyword)
        """
        batch_size = q.size(0)
        len_q = q.size(1)
        len_k = k.size(1)

        # 1. Projektion: [B, H, S, D_K]
        Q = self.w_q(q).view(batch_size, len_q,
                             self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, len_k,
                             self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, len_k,
                             self.n_heads, self.d_k).transpose(1, 2)

        # 2. RoPE Positions-Handling
        # Q_pos sind immer die position_ids des aktuellen Schritts
        if position_ids is None:
            q_pos = torch.arange(len_q, device=q.device).unsqueeze(
                0).expand(batch_size, -1)
        else:
            q_pos = position_ids

        # K_pos: In Cross-Attention ist die Punktwolke oft statisch (0, 1, 2...)
        if self.is_cross_attention:
            k_pos = torch.arange(len_k, device=k.device).unsqueeze(
                0).expand(batch_size, -1)
        else:
            k_pos = q_pos

        # BEIDE rotieren (Wichtig für relatives Alignment!)
        Q = self.rope(Q, q_pos)
        K = self.rope(K, k_pos)

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Maske für Multi-Head broadcasten [B, 1, S_q, S_k]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(self.dropout(weights), V)

        # 4. Concatenate & Output
        attn_output = attn_output.transpose(
            1, 2).contiguous().view(batch_size, len_q, self.d_model)
        return self.w_o(attn_output)
