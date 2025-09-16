import torch
import torch.nn as nn

from transformer_blocks.decoder_block import Decoder, DecoderBlock
from transformer_blocks.encoder_block import Encoder, EncoderBlock
from transformer_blocks.input_embedding import InputEmbedding
from transformer_blocks.positional_encoding import LearnedPositionalEncoding
from transformer_blocks.projection_layer import ProjectionLayer
from transformer_blocks.multi_head_attention_block import MultiHeadAttentionBlock
from transformer_blocks.feed_forward import FeedForward


class Transformer(nn.Module):
    """
    Full Transformer model consisting of:
    - Encoder
    - Decoder
    - Embedding layers
    - Positional encodings
    - Final projection layer
    """

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbedding,
                 trg_embed: InputEmbedding,
                 src_pos: LearnedPositionalEncoding,
                 trg_pos: LearnedPositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        Encode the source sequence.
        Args:
            src: (batch, src_seq_len) source token IDs
            src_mask: mask for padding tokens in source
        Returns:
            encoder output representations
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, trg, trg_mask):
        """
        Decode the target sequence with encoder output.
        Args:
            encoder_output: output from the encoder
            src_mask: source mask
            trg: (batch, trg_seq_len) target token IDs
            trg_mask: target mask (look-ahead + padding)
        Returns:
            decoder output representations
        """
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg, encoder_output, src_mask, trg_mask)

    def project(self, x):
        """
        Project decoder outputs into vocabulary logits.
        Args:
            x: decoder output (batch, trg_seq_len, d_model)
        Returns:
            log-probabilities (batch, trg_seq_len, vocab_size)
        """
        return self.projection_layer(x)


def building_transformer(src_vocab_size: int, trg_vocab_size: int,
                         src_seq_len: int,
                         trg_seq_len: int,
                         d_model: int = 512,
                         N: int = 4,
                         h: int = 4,
                         dropout: float = 0.1,
                         d_ff: int = 2048):
    """
    Build a Transformer model with given hyperparameters.
    Args:
        src_vocab_size: size of source vocabulary
        trg_vocab_size: size of target vocabulary
        src_seq_len: maximum length of source sequence
        trg_seq_len: maximum length of target sequence
        d_model: hidden dimension size
        N: number of encoder/decoder layers
        h: number of attention heads
        dropout: dropout rate
        d_ff: hidden size of feed-forward network
    Returns:
        A Transformer model instance
    """

    # Source & Target embeddings
    src_embed = InputEmbedding(src_vocab_size, d_model)
    trg_embed = InputEmbedding(trg_vocab_size, d_model)

    # Positional encodings
    src_pos = LearnedPositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = LearnedPositionalEncoding(d_model, trg_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout, d_model)
        encoder_blocks.append(encoder_block)

    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,
                                     decoder_cross_attention_block,
                                     feed_forward_block,
                                     dropout, d_model)
        decoder_blocks.append(decoder_block)

    # Wrap encoder & decoder layers into Modules
    encoder = Encoder(nn.ModuleList(encoder_blocks), d_model )
    decoder = Decoder(nn.ModuleList(decoder_blocks) , d_model)

    # Projection layer to vocab size
    projection_layer = ProjectionLayer(d_model, trg_vocab_size)

    # Create the full transformer
    transformer = Transformer(encoder, decoder, src_embed,
                              trg_embed,
                              src_pos,
                              trg_pos,
                              projection_layer)

    # Initialize weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


def get_model(config):
    model = building_transformer(config['src_vocab_size'] , config['trg_vocab_size'] , config['seq_len'] , config['seq_len'] , config['d_model'] )
    
    return model