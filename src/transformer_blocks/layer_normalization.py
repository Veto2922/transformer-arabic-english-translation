import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        """
        Custom implementation of Layer Normalization.
        Unlike BatchNorm, this normalizes across the last (feature) dimension
        of each sequence element, making it suitable for NLP/sequence tasks.

        Args:
            features (int): Size of the feature dimension (d_model).
            eps (float): Small constant to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        # Learnable scale (γ) and shift (β) parameters
        self.alpha = nn.Parameter(torch.ones(features))   # shape: (d_model,)
        self.bias = nn.Parameter(torch.zeros(features))   # shape: (d_model,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Layer Normalization.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Normalized tensor with the same shape.
        """
        # Compute mean and variance along the feature dimension only
        mean = x.mean(dim=-1, keepdim=True)               # (batch_size, seq_len, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # (batch_size, seq_len, 1)

        # Standard deviation
        std = (var + self.eps).sqrt()

        # Normalize, then scale (alpha) and shift (bias)
        x_norm = (x - mean) / std
        return self.alpha * x_norm + self.bias
