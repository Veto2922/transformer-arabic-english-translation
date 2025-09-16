import torch
import torch.nn as nn

from transformer_blocks.multi_head_attention_block import MultiHeadAttentionBlock
from transformer_blocks.feed_forward import FeedForward
from transformer_blocks.residual_connections import ResidualConnection
from transformer_blocks.layer_normalization import LayerNormalization


class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block.

    Each decoder block consists of 3 main components:
    1. Masked Multi-Head Self-Attention (with residual connection & layer norm).
    2. Cross-Attention (attending to encoder outputs) 
       with residual connection & layer norm.
    3. Feed-Forward Network (with residual connection & layer norm).

    Args:
        self_attention_block (MultiHeadAttentionBlock): 
            Multi-head self-attention module (for target sequence).
        cross_attention_block (MultiHeadAttentionBlock): 
            Multi-head cross-attention module (target attends to encoder output).
        feed_forward_block (FeedForward): 
            Position-wise feed-forward network.
        dropout (float): 
            Dropout probability.
        d_model (int): 
            Model dimensionality (hidden size).

    Shape:
        - Input: 
            x (batch_size, target_seq_len, d_model)
            encoder_output (batch_size, source_seq_len, d_model)
            src_mask (batch_size, 1, 1, source_seq_len)
            trg_mask (batch_size, 1, target_seq_len, target_seq_len)
        - Output: 
            (batch_size, target_seq_len, d_model)
    """

    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForward,
                 dropout: float, d_model: int):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        # Three residual connections:
        # 1) Self-attention
        # 2) Cross-attention
        # 3) Feed-forward
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(3)
        ])

    def forward(self, x, encoder_output, src_mask, trg_mask):
        """
        Forward pass of the decoder block.

        Args:
            x (torch.Tensor): 
                Target input embeddings (batch_size, target_seq_len, d_model).
            encoder_output (torch.Tensor): 
                Encoder output (batch_size, source_seq_len, d_model).
            src_mask (torch.Tensor): 
                Source mask to avoid attending to <pad> tokens 
                (batch_size, 1, 1, source_seq_len).
            trg_mask (torch.Tensor): 
                Target mask (causal + padding) 
                (batch_size, 1, target_seq_len, target_seq_len).

        Returns:
            torch.Tensor: 
                Output of decoder block (batch_size, target_seq_len, d_model).
        """

        # 1. Masked self-attention (decoder looks only at previous target tokens)
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, trg_mask)
        )

        # 2. Cross-attention (decoder queries attend over encoder outputs)
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )

        # 3. Feed-forward network
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):
    """
    Transformer Decoder.

    The decoder is a stack of N decoder blocks followed by a final layer normalization.

    Args:
        layers (nn.ModuleList): 
            List of DecoderBlock modules.

    Shape:
        - Input: 
            x (batch_size, target_seq_len, d_model)
            encoder_output (batch_size, source_seq_len, d_model)
            src_mask (batch_size, 1, 1, source_seq_len)
            trg_mask (batch_size, 1, target_seq_len, target_seq_len)
        - Output: 
            (batch_size, target_seq_len, d_model)
    """

    def __init__(self, layers: nn.ModuleList , d_model):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): 
                Target input embeddings (batch_size, target_seq_len, d_model).
            encoder_output (torch.Tensor): 
                Encoder output (batch_size, source_seq_len, d_model).
            src_mask (torch.Tensor): 
                Source mask to avoid padding tokens in source.
            trg_mask (torch.Tensor): 
                Target mask to ensure causal decoding.

        Returns:
            torch.Tensor: 
                Decoded output (batch_size, target_seq_len, d_model).
        """

        # Pass input through each decoder block
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)

        # Final layer normalization
        return self.norm(x)
