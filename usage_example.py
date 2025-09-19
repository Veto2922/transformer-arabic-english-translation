#!/usr/bin/env python3
"""
Example usage of the Arabic-English Translation Transformer from Hugging Face Hub
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download

def load_model_from_hf(repo_name: str):
    """Load model and tokenizers from Hugging Face Hub"""
    print(f"🔄 Loading model from {repo_name}...")
    
    # Download model files
    model_path = hf_hub_download(repo_id=repo_name, filename="weights/tmodel_02.pt")
    tokenizer_ar_path = hf_hub_download(repo_id=repo_name, filename="tokenizer_models/spm_ar_unigram.model")
    tokenizer_en_path = hf_hub_download(repo_id=repo_name, filename="tokenizer_models/spm_en_unigram.model")
    
    # Load tokenizers
    import sentencepiece as spm
    tokenizer_src = spm.SentencePieceProcessor(model_file=tokenizer_ar_path)
    tokenizer_trg = spm.SentencePieceProcessor(model_file=tokenizer_en_path)
    
    # Load model
    from src.config import get_config
    from src.transformer_model import get_model
    
    cfg = get_config()
    model = get_model(cfg)
    
    # Load weights
    device = torch.device("cpu")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    
    print("✅ Model loaded successfully!")
    return model, tokenizer_src, tokenizer_trg, cfg, device

def translate_text(model, tokenizer_src, tokenizer_trg, cfg, device, text: str):
    """Translate Arabic text to English"""
    from src.utils.text import text_to_ids, ids_to_text
    from src.utils.decoding import greedy_decode
    
    model.eval()
    with torch.no_grad():
        # Tokenize input
        src_ids = text_to_ids(tokenizer_src, text, cfg["seq_len"]).unsqueeze(0).to(device)
        src_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2)
        
        # Decode
        out_ids = greedy_decode(
            model, src_ids, src_mask, cfg["seq_len"], device
        )
        
        # Convert to text
        translation = ids_to_text(tokenizer_trg, out_ids)
        return translation

def main():
    """Example usage"""
    print("🤗 Arabic-English Translation Transformer - Usage Example")
    print("=" * 60)
    
    # Replace with your actual repository name
    repo_name = "your-username/arabic-english-transformer"
    
    try:
        # Load model
        model, tokenizer_src, tokenizer_trg, cfg, device = load_model_from_hf(repo_name)
        
        # Example translations
        examples = [
            "مرحبا بالعالم",
            "أنا أحب البرمجة",
            "كيف حالك؟",
            "هذا كتاب جميل",
            "أريد أن أتعلم العربية"
        ]
        
        print("\n📝 Example Translations:")
        print("-" * 40)
        
        for arabic_text in examples:
            translation = translate_text(model, tokenizer_src, tokenizer_trg, cfg, device, arabic_text)
            print(f"Arabic:  {arabic_text}")
            print(f"English: {translation}")
            print()
        
        # Interactive mode
        print("💬 Interactive Mode (type 'quit' to exit):")
        while True:
            try:
                text = input("\nEnter Arabic text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if text:
                    translation = translate_text(model, tokenizer_src, tokenizer_trg, cfg, device, text)
                    print(f"Translation: {translation}")
            except KeyboardInterrupt:
                break
        
        print("\n👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the model is uploaded to Hugging Face Hub and the repository name is correct")

if __name__ == "__main__":
    main()
