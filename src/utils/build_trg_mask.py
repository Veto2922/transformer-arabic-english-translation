import torch
from data_loader import PAD_ID

def build_trg_mask(decoder_input: torch.Tensor) -> torch.Tensor:
    """
    Build target mask combining causal mask and key padding mask.
    Shapes:
      - decoder_input: (B, T)
      - return: (B, 1, T, T)
    """
    batch_size, tgt_len = decoder_input.size(0), decoder_input.size(1)
    device = decoder_input.device

    # Causal mask (1, 1, T, T)
    causal_mask = torch.tril(
        torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device)
    ).unsqueeze(0).unsqueeze(1)

    # Key padding mask (B, 1, 1, T)
    key_padding_mask = (decoder_input != PAD_ID).unsqueeze(1).unsqueeze(2)

    # Broadcast to (B, 1, T, T)
    return causal_mask & key_padding_mask