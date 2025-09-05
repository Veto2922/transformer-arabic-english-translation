import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    """
    InputEmbedding is responsible for converting token IDs into dense vectors
    that can be processed by the Transformer. 
    
    - Each token ID is mapped to a learnable embedding vector.
    - The embeddings are scaled by sqrt(d_model) as suggested in the Transformer paper.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimensionality of the embeddings.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Forward pass of the InputEmbedding.
        
        Args:
            x (Tensor): Input token IDs of shape (batch_size, seq_length).
        
        Returns:
            Tensor: Scaled embeddings of shape (batch_size, seq_length, d_model).
        """
        # Convert token IDs to embeddings and scale them
        return self.embedding(x) * math.sqrt(self.d_model)
