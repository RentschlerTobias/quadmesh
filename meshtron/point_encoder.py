import torch.nn as nn
import torch


class PerceiverPointEncoder(nn.Module):

    """Wie MeshTron - Cross-Attention mit learned queries"""

    def __init__(self, d_model=128, input_dim=2, n_latents=32):
        super().__init__()

        # Learned latent queries
        self.latents = nn.Parameter(torch.randn(n_latents, d_model))

        # Point projection
        self.point_proj = nn.Linear(input_dim, d_model)

        # Cross-attention: latents attend to points
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            num_heads=4,
            batch_first=True
        )

        # Self-attention zwischen latents
        self.self_attention = nn.MultiheadAttention(
            d_model,
            num_heads=4,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, points):
        """
        points: [batch, n_points, 2]
        output: [batch, n_latents, d_model]
        """
        batch_size = points.shape[0]

        # Project points
        point_features = self.point_proj(points)  # [batch, n_points, d_model]

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: latents query points
        attn_out, _ = self.cross_attention(
            query=latents,           # [batch, n_latents, d_model]
            key=point_features,      # [batch, n_points, d_model]
            value=point_features
        )
        latents = latents + attn_out
        latents = self.norm1(latents)

        # Self-attention between latents
        self_attn_out, _ = self.self_attention(latents, latents, latents)
        latents = latents + self_attn_out
        latents = self.norm2(latents)

        return latents  # [batch, n_latents, d_model]
