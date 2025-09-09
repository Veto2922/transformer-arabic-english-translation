import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    """
    Final Linear + Softmax Projection Layer.
    Projects the decoder's output (d_model) into the vocabulary space,
    and applies log_softmax to produce probabilities for each token.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        """
        Args:
            x (Tensor): Decoder output of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor: Log probabilities of shape (batch_size, seq_len, vocab_size)
        """
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)