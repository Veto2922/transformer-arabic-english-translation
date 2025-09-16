import torch
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention block implementation.

    This class implements the multi-head attention mechanism as introduced 
    in the Transformer architecture (Vaswani et al., 2017).
    
    Args:
        d_model (int): Dimensionality of the input embeddings (hidden size).
        h (int): Number of attention heads.
        dropout (float, optional): Dropout probability. Default is 0.1.

    Shape:
        - Input: q, k, v (batch_size, seq_len, d_model)
        - mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
        - Output: (batch_size, seq_len, d_model)
    """
    
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.h = h
        self.dropout_rate = dropout
        
        # Ensure d_model is divisible by h
        assert d_model % h == 0, "d_model must be divisible by number of heads h"
        
        # Dimension of each head
        self.d_k = d_model // h
        
        # Learnable projection matrices for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv
        
        # Final output projection
        self.w_o = nn.Linear(d_model, d_model)  # Wo
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod    
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute scaled dot-product attention.
        
        Args:
            query: (batch_size, h, seq_len, d_k)
            key:   (batch_size, h, seq_len, d_k)
            value: (batch_size, h, seq_len, d_k)
            mask:  Mask tensor for padding or causal masking
            dropout: Dropout layer applied on attention scores
            
        Returns:
            output: Weighted sum of values (batch_size, h, seq_len, d_k)
            attention_score: Attention weights (batch_size, h, seq_len, seq_len)
        """
        d_k = query.shape[-1]
        
        # Compute raw attention scores: QK^T / sqrt(d_k)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask (if provided) -> very negative values where mask == 0
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float('-inf'))
        
        # Normalize scores into probabilities
        attention_score = attention_score.softmax(dim=-1)
        
        # Apply dropout to attention weights
        if dropout is not None:
            attention_score = dropout(attention_score)
            
        # Multiply attention weights by values
        return (attention_score @ value), attention_score
        
    def forward(self, q, k, v, mask):
        """
        Forward pass of the Multi-Head Attention.
        
        Args:
            q, k, v: Input tensors of shape (batch_size, seq_len, d_model)
            mask: Mask tensor (optional)
            
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Linear projections
        query = self.w_q(q)  # (batch, seq_len, d_model)
        key = self.w_k(k)    # (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model)
        
        # Reshape & transpose for multi-head attention
        # (batch, seq_len, d_model) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key   = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Concatenate heads back: (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # Final linear layer
        return self.w_o(x) 
