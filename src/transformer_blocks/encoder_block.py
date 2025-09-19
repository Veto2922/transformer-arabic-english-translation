import torch
import torch.nn as nn

from .multi_head_attention_block import MultiHeadAttentionBlock
from .feed_forward import FeedForward
from .residual_connections import ResidualConnection
from .layer_normalization import LayerNormalization


class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block.

    This block consists of:
    1. Multi-Head Self-Attention (with residual connection & normalization).
    2. Feed-Forward Network (with residual connection & normalization).

    Args:
        self_attention_block (MultiHeadAttentionBlock): Multi-head self-attention layer.
        feed_forward_block (FeedForward): Position-wise feed-forward layer.
        dropout (float): Dropout probability.
        d_model (int): Model dimensionality (hidden size).

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Mask:  (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
        - Output: (batch_size, seq_len, d_model)
    """

    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForward, dropout: float, d_model: int):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        # Two residual connections: one for attention, one for feed-forward
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(2)
        ])

    def forward(self, x, src_mask):
        """
        Forward pass of the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # 1. Apply multi-head self-attention with residual connection
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )

        # 2. Apply feed-forward network with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    """
    Transformer Encoder.

    The encoder is a stack of N encoder blocks followed by a final layer normalization.

    Args:
        layers (nn.ModuleList): List of EncoderBlock modules.

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Mask:  (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
        - Output: (batch_size, seq_len, d_model)
    """

    def __init__(self, layers: nn.ModuleList , d_model):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Encoded output of shape (batch_size, seq_len, d_model).
        """
        # Pass input through each encoder block
        for layer in self.layers:
            x = layer(x, mask)

        # Final layer normalization
        return self.norm(x)
