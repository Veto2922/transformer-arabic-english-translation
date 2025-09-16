from pathlib import Path


def get_config():
    return {
        "batch_size": 128,
        "num_epochs": 10,
        'lr': 10**-4,
        "seq_len": 80,
        "d_model": 512,
        "src_vocab_size": 32000,
        "trg_vocab_size": 26000,
        "lang_src": "ar",
        "lang_trg": "en",
        # Store checkpoints inside the project under ./weights
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "01",
        # TensorBoard run directory inside the project
        "experiment_name": str((Path("runs") / "tmodel").as_posix())
        
    }
    
    
def get_weights_file_path(config , epoch:str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    
    # Build a path like ./weights/tmodel_XX.pt (resolved under project cwd)
    return str((Path(model_folder) / model_filename).resolve())
    
    
    
