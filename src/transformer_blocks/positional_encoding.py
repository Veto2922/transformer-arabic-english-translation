import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding Layer for Transformers.

    Instead of using fixed sinusoidal functions (as in the original Transformer),
    this module assigns each position in the sequence a trainable embedding vector. 
    This allows the model to learn positional patterns directly from data, which is 
    useful for tasks like machine translation where sentence lengths are limited.

    Args:
        d_model (int): Dimension of the model (embedding size).
        max_len (int): Maximum sequence length expected.
        dropout (float): Dropout probability applied after adding positional embeddings.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Positional embeddings are trainable (shape: [max_len, d_model])
        self.pe = nn.Embedding(max_len, d_model)

        # Initialize embeddings with small random values
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pe.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for positional encoding.

        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Embeddings with positional encodings added, same shape as input.
        """
        # Create position indices for the current sequence length
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)  # (1, seq_len)

        # Add positional embeddings to token embeddings
        x = x + self.pe(positions)

        # Apply dropout to improve generalization
        return self.dropout(x)
