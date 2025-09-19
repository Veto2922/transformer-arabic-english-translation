# Transformer Architecture Documentation

This document provides a detailed explanation of the Transformer architecture implemented in this project, including all the building blocks and their relationships.

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Encoder Architecture](#encoder-architecture)
4. [Decoder Architecture](#decoder-architecture)
5. [Attention Mechanisms](#attention-mechanisms)
6. [Training Process](#training-process)
7. [Inference Process](#inference-process)

## Overview

The Transformer architecture is a sequence-to-sequence model that uses attention mechanisms to process sequences in parallel, making it much faster than recurrent neural networks. Our implementation follows the original "Attention Is All You Need" paper with some modern improvements.

### Key Features

- **Parallel Processing**: Unlike RNNs, all positions are processed simultaneously
- **Self-Attention**: Each position can attend to all other positions in the sequence
- **Multi-Head Attention**: Multiple attention heads capture different types of relationships
- **Residual Connections**: Help with gradient flow and training stability
- **Layer Normalization**: Stabilizes training and improves convergence

## Core Components

### 1. Input Embedding (`input_embedding.py`)

Converts token IDs to dense vector representations.

```python
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```

**Key Points:**
- Scales embeddings by √d_model to balance with positional encodings
- Separate embeddings for source (Arabic) and target (English)
- Vocabulary sizes: 32K (Arabic), 26K (English)

### 2. Positional Encoding (`positional_encoding.py`)

Adds position information to embeddings since Transformers have no inherent notion of sequence order.

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
    
    def forward(self, x):
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:seq_len])
```

**Key Points:**
- Uses learned positional embeddings instead of sinusoidal
- Better for fixed maximum sequence lengths
- Dropout prevents overfitting to specific positions

### 3. Layer Normalization (`layer_normalization.py`)

Normalizes inputs across the feature dimension for each token independently.

```python
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Scale
        self.bias = nn.Parameter(torch.zeros(features))  # Shift
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
```

**Key Points:**
- Normalizes across the last dimension (features)
- Learnable scale and shift parameters
- More stable than batch normalization for NLP

### 4. Residual Connections (`residual_connections.py`)

Adds skip connections to help with gradient flow and prevent vanishing gradients.

```python
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

**Key Points:**
- Implements pre-norm architecture (modern approach)
- Dropout applied to sublayer output, not residual
- Helps with training deep networks

### 5. Feed-Forward Network (`feed_forward.py`)

Applies non-linear transformations to each position independently.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
```

**Key Points:**
- Two linear layers with ReLU activation
- d_ff = 4 * d_model (2048 for d_model=512)
- Dropout for regularization

## Encoder Architecture

The encoder processes the input sequence and creates contextual representations.

### Encoder Block (`encoder_block.py`)

Each encoder block contains:
1. Multi-head self-attention
2. Feed-forward network
3. Residual connections and layer normalization around each

```python
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block, feed_forward_block, dropout, d_model):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(2)
        ])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
```

### Full Encoder (`encoder_block.py`)

Stacks multiple encoder blocks with final layer normalization.

```python
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

## Decoder Architecture

The decoder generates the output sequence step by step, attending to both previous target tokens and encoder outputs.

### Decoder Block (`decoder_block.py`)

Each decoder block contains:
1. Masked multi-head self-attention
2. Cross-attention (decoder-encoder attention)
3. Feed-forward network
4. Residual connections and layer normalization around each

```python
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block, cross_attention_block, 
                 feed_forward_block, dropout, d_model):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(3)
        ])
    
    def forward(self, x, encoder_output, src_mask, trg_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, trg_mask)
        )
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
```

## Attention Mechanisms

### Multi-Head Attention (`multi_head_attention_block.py`)

The core attention mechanism that allows the model to focus on different parts of the input.

```python
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(out)
```

**Key Points:**
- **Query (Q)**: What the model is looking for
- **Key (K)**: What the model is looking at
- **Value (V)**: The actual content to extract
- **Scaling**: Division by √d_k prevents softmax saturation
- **Masking**: Prevents attention to padding or future tokens

### Attention Types

1. **Self-Attention (Encoder)**: Each position attends to all positions in the same sequence
2. **Masked Self-Attention (Decoder)**: Each position attends only to previous positions
3. **Cross-Attention (Decoder)**: Decoder positions attend to encoder outputs

## Training Process

### Loss Function

Cross-entropy loss with label smoothing:

```python
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
```

**Key Points:**
- Ignores padding tokens
- Label smoothing prevents overconfidence
- Teacher forcing during training

### Optimizer

Adam optimizer with small epsilon:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-9)
```

### Mixed Precision Training

Uses FP16 for faster training and lower memory usage:

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    # Forward pass
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Inference Process

### Greedy Decoding

Generates one token at a time by selecting the most likely token:

```python
def greedy_decode(model, src, src_mask, max_len, device):
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
        
        # Start with BOS token
        decoder_input = torch.tensor([[BOS_ID]], device=device)
        
        for _ in range(max_len):
            decoder_mask = create_look_ahead_mask(decoder_input.size(1))
            decoder_output = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)
            logits = model.project(decoder_output)
            
            # Get the most likely next token
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == EOS_ID:
                break
        
        return decoder_input
```

### Beam Search

Maintains multiple hypotheses and selects the best one:

```python
def beam_search(model, src, src_mask, max_len, beam_size, device):
    # Implementation maintains beam_size hypotheses
    # Selects the sequence with highest probability
    pass
```

## Model Configuration

Our model uses the following hyperparameters:

```python
config = {
    "d_model": 512,           # Hidden dimension
    "N": 4,                   # Number of encoder/decoder layers
    "h": 4,                   # Number of attention heads
    "d_ff": 2048,             # Feed-forward dimension
    "dropout": 0.1,           # Dropout rate
    "src_vocab_size": 32000,  # Arabic vocabulary size
    "trg_vocab_size": 26000,  # English vocabulary size
    "seq_len": 80,            # Maximum sequence length
}
```

## Performance Characteristics

- **Parameters**: ~72M total parameters
- **Memory Usage**: ~4GB GPU memory during training
- **Training Speed**: ~45 minutes per epoch on T4 GPU
- **Inference Speed**: ~100ms per sentence on CPU

This architecture provides a solid foundation for Arabic-English translation while being educational and easy to understand. The modular design makes it easy to experiment with different configurations and improvements.
