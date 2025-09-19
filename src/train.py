from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import get_config, get_weights_file_path
from .data_loader import get_data_loader, PAD_ID
from .transformer_model import get_model


def train_model(cfg: dict) -> None:
    """
    Train the Transformer model using Mixed Precision (FP16).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure the weights folder exists
    Path(cfg['model_folder']).mkdir(parents=True, exist_ok=True)

    # Data loaders
    train_loader, valid_loader = get_data_loader(cfg)

    # Model
    model = get_model(cfg).to(device)

    # TensorBoard writer
    writer = SummaryWriter(cfg['experiment_name'])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], eps=1e-9)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Optionally preload a checkpoint
    initial_epoch = 0
    global_step = 0
    if cfg['preload']:
        model_filename = get_weights_file_path(cfg, cfg['preload'])
        print(f"Preloading model: {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = int(state.get('epoch', -1)) + 1
        global_step = int(state.get('global_step', 0))

    # Loss: ignore padding; label smoothing for stability
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

    for epoch in range(initial_epoch, cfg['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator:
            # Inputs
            encoder_input = batch['src_input'].to(device)        # (B, S)
            decoder_input = batch['decoder_input'].to(device)    # (B, T)
            encoder_mask = batch['src_mask'].to(device)          # (B, 1, 1, S)
            decoder_mask = batch['trg_mask'].to(device)          # (B, 1, T, T)

            labels = batch['decoder_target'].to(device)          # (B, T)

            optimizer.zero_grad(set_to_none=True)

            # Forward + loss with autocast
            with torch.cuda.amp.autocast():
                encoder_output = model.encode(encoder_input, encoder_mask)                 
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  
                logits = model.project(decoder_output)                                     
                loss = loss_fn(logits.reshape(-1, cfg['trg_vocab_size']), labels.reshape(-1))

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.flush()

            # Backpropagation with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

        # Save checkpoint each epoch
        model_filename = get_weights_file_path(cfg, f'{epoch:02d}')
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        }, model_filename)


if __name__ == "__main__":
    cfg = get_config()
    train_model(cfg)
