import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from embedding import Embedding
from attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """Ein kompletter Transformer Block mit Attention und Feed-Forward"""
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


class MeshTransformer(nn.Module):
    """Kompletter Transformer fuer Mesh-Generierung"""
    def __init__(self,
                 vocab_size=259,
                 d_model=512,
                 n_heads=8,
                 n_layers=6,
                 d_ff=2048,
                 max_len=5000,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Embedding layers
        self.embedding = Embedding(vocab_size, d_model, max_len)
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        # Output head - predicts next token
        self.output_projection = nn.Linear(d_model, vocab_size)
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tokens, mask=None):
        """
        Args:
            input_tokens: [batch_size, seq_len] - Input token IDs
            mask: Optional attention mask
        Returns:
            logits: [batch_size, seq_len, vocab_size] - Predictions for each position
        """
        # Embed tokens
        x = self.embedding(input_tokens)  # [batch_size, seq_len, d_model]
        x = self.dropout(x)
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        # Project to vocabulary size
        logits = self.output_projection(x)  # [batch_size, seq_len, vocab_size]
        return logits
