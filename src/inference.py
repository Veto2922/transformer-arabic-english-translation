import sys
from pathlib import Path
from typing import Optional

import torch
import sentencepiece as spm

from src.config import get_config, get_weights_file_path
from src.transformer_model import get_model
from src.utils.text import ids_to_text, text_to_ids
from src.utils.decoding import greedy_decode


def load_model_and_tokenizers(cfg: dict, device: torch.device):
    model = get_model(cfg).to(device)
    if cfg.get("preload"):
        weights_path = get_weights_file_path(cfg, cfg["preload"])
        print("weight_path: " , weights_path)
        if Path(weights_path).exists():
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state["model_state_dict"])
        else:
            print(f"Warning: checkpoint not found at {weights_path}; using random weights.")

    proj_root = Path(__file__).resolve().parent.parent
    tok_dir = proj_root / "tokenizer_models"
    tokenizer_src = spm.SentencePieceProcessor(model_file=str(tok_dir / "spm_ar_unigram.model"))
    tokenizer_trg = spm.SentencePieceProcessor(model_file=str(tok_dir / "spm_en_unigram.model"))
    return model, tokenizer_src, tokenizer_trg


def translate_sentence(model, tokenizer_src, tokenizer_trg, text: str, cfg: dict, device: torch.device) -> str:
    model.eval()
    with torch.no_grad():
        src_ids = text_to_ids(tokenizer_src, text, cfg["seq_len"]).unsqueeze(0).to(device)  # (1, S)
        src_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2)  # PAD_ID == 0
        out_ids = greedy_decode(
            model,
            src_ids,
            src_mask,
            cfg["seq_len"],
            device,
        )
        return ids_to_text(tokenizer_trg, out_ids)



