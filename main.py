import torch

from src.config import get_config
from src.inference import load_model_and_tokenizers , translate_sentence


def main() -> None:
    cfg = get_config()
    device = torch.device("cpu")
    print(f"Using device: {device}")

    model, tok_src, tok_trg = load_model_and_tokenizers(cfg, device)

    print("Type an Arabic sentence and press Enter (or just press Enter to quit).\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            print("Bye.")
            break
        translation = translate_sentence(model, tok_src, tok_trg, line, cfg, device)
        print(f"EN: {translation}")


if __name__ == "__main__":
    main()