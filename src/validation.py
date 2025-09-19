import torch
import torch.nn as nn
import sentencepiece as spm
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.cer import CharErrorRate

from .data_loader import PAD_ID , BOS_ID , EOS_ID
from .utils.text import ids_to_text
from .utils.decoding import greedy_decode , beam_search_decode
from .utils.metrics import compute_text_metrics

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .config import get_config, get_weights_file_path
from .transformer_model import get_model
from .data_loader import get_data_loader, collate_fn 



def run_validation(model, validation_ds, tokenizer_src, tokenizer_trg, max_len, device, print_msg, global_state, writer, num_examples=2):
    
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    prediction = []
    
    console_width = 80



    # Determine step base
    try:
        from time import time as _time
        step_base = int(global_state) if global_state is not None else int(_time())
    except Exception:
        step_base = int(global_state) if global_state is not None else 0
    
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['src_input'].to(device)
            encoder_mask = batch['src_mask'].to(device)
            
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Greedy decode prediction
            greedy_ids = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                max_len,
                device,
            )  # (T')
            
            # beam decode prediction
            # beam_search_ids = beam_search_decode(model,
            #     encoder_input,
            #     encoder_mask,
            #     max_len,
            #     device)

            # Prepare expected target ids from batch (decoder_target already shifted)
            target_ids = batch['decoder_target'][0].to(device)  # (T)

            # Detokenize
            src_text = ids_to_text(tokenizer_src, encoder_input[0])
            tgt_text = ids_to_text(tokenizer_trg, target_ids)
            pred_text = ids_to_text(tokenizer_trg, greedy_ids)

            source_texts.append(src_text)
            expected.append(tgt_text)
            prediction.append(pred_text)

            # Pretty print
            if print_msg is not None:
                print_msg("-" * console_width)
                print_msg(f"Source:      {src_text}")
                print_msg(f"Expected:    {tgt_text}")
                print_msg(f"Prediction:  {pred_text}")


            if count >= num_examples:
                break

    # Compute and log metrics (BLEU, WER, CER) 
    try:
        bleu_value, wer_value, cer_value = compute_text_metrics(prediction, expected)
    except Exception as e:
        if print_msg is not None:
            print_msg(f"Metrics computation failed: {e}")
        bleu_value, wer_value, cer_value = 0.0, 0.0, 0.0

    # TensorBoard logging 
    if writer is not None:
        step_avg = step_base + count
        writer.add_scalar("val/bleu", float(bleu_value), step_avg)
        writer.add_scalar("val/wer", float(wer_value), step_avg)
        writer.add_scalar("val/cer", float(cer_value), step_avg)
        writer.flush()

    return {
        "sources": source_texts,
        "expected": expected,
        "predictions": prediction,
        "bleu": float(bleu_value),
        "wer": float(wer_value),
        "cer": float(cer_value),
    }


if __name__ == "__main__":
    # Minimal runnable entry point for quick validation preview
    

    cfg = get_config()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")

    # Load tokenizers
    proj_root = Path(__file__).resolve().parent.parent
    tok_dir = proj_root / "tokenizer_models"
    tokenizer_src = spm.SentencePieceProcessor(model_file=str(tok_dir / "spm_ar_unigram.model"))
    tokenizer_trg = spm.SentencePieceProcessor(model_file=str(tok_dir / "spm_en_unigram.model"))

    # Build data loaders and force validation batch_size=1
    train_loader, valid_loader_full = get_data_loader(cfg)
    valid_loader = DataLoader(
        valid_loader_full.dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Load model and optional checkpoint
    model = get_model(cfg).to(device)
    if cfg.get('preload'):
        weights_path = get_weights_file_path(cfg, cfg['preload'])
        if Path(weights_path).exists():
            print(f"Loading checkpoint: {weights_path}")
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state["model_state_dict"])
        else:
            print(f"Checkpoint not found at {weights_path}, running with random weights.")

    writer = SummaryWriter(cfg['experiment_name'])

    def print_msg(msg: str):
        print(msg)

    num_of_val_ex = 2000
    results = run_validation(
        model=model,
        validation_ds=valid_loader,
        tokenizer_src=tokenizer_src,
        tokenizer_trg=tokenizer_trg,
        max_len=cfg['seq_len'],
        device=device,
        print_msg=print_msg,
        global_state=0,
        writer=writer,
        num_examples= num_of_val_ex,
    )
    

    print("\nMetrics with number of val examble:" , num_of_val_ex)
    
    print({k: v for k, v in results.items() if k in ("bleu", "wer", "cer")})
    
    
    



