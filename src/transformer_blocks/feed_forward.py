import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Position-wise Feed Forward Network (FFN).
        Applies two linear transformations with a non-linearity in between.
        
        Args:
            d_model (int): Dimensionality of the input embeddings.
            d_ff (int): Hidden dimension inside the FFN (usually 4 * d_model).
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # expand
        self.fc2 = nn.Linear(d_ff, d_model)  # compress
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # GELU can also be used for better performance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FFN.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor: Output tensor of the same shape (batch_size, seq_len, d_model).
        """
        return self.fc2(self.dropout(self.activation(self.fc1(x))))
