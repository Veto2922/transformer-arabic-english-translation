import torch

from data_loader import PAD_ID, BOS_ID, EOS_ID


def ids_to_text(tokenizer, ids: torch.Tensor) -> str:
    """
    Convert a tensor of token ids to text, stripping BOS/EOS/PAD.

    - Supports SentencePiece via `decode_ids` or `decode`.
    - Falls back to space-joined ids if tokenizer lacks decoding.
    """
    if ids.dim() != 1:
        ids = ids.view(-1)

    filtered = [int(t) for t in ids.tolist() if t not in (BOS_ID, EOS_ID, PAD_ID)]

    if hasattr(tokenizer, "decode_ids"):
        return tokenizer.decode_ids(filtered)
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(filtered)

    return " ".join(str(i) for i in filtered)


def text_to_ids(tokenizer, text: str, max_len: int) -> torch.Tensor:
    """
    Encode raw text into token ids using SentencePiece, adding BOS/EOS.

    - Truncates to fit max_len including BOS and EOS.
    - Returns a 1D LongTensor of shape (T,).
    """
    if hasattr(tokenizer, "encode_as_ids"):
        ids = tokenizer.encode_as_ids(text)
    elif hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
    else:
        # Fallback: split by whitespace and map to ints if possible
        try:
            ids = [int(tok) for tok in text.strip().split()]
        except Exception:
            ids = []

    # Reserve 2 places for BOS/EOS
    inner_max = max(0, max_len - 2)
    ids = ids[:inner_max]
    ids = [BOS_ID] + ids + [EOS_ID]

    return torch.tensor(ids, dtype=torch.long)
