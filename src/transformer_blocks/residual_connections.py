import torch
import torch.nn as nn

from src.transformer_blocks.layer_normalization import LayerNormalization

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int ,dropout: float = 0.1):
        """
        Residual connection followed by layer normalization.
        
        Args:
            dropout (float): Dropout probability applied to the sublayer output.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)  

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Apply residual connection to any sublayer.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            sublayer (nn.Module): The sublayer (e.g., attention or feed-forward).
            
        Returns:
            Tensor: Normalized tensor after residual connection.
        """
        # Run sublayer and apply dropout
        out = self.dropout(sublayer(x))
        # Residual + LayerNorm
        return self.norm(x + out)
