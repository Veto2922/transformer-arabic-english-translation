import gradio as gr
import torch
from src.inference import load_model_and_tokenizers, translate_sentence
from src.config import get_config

# ---------------------
# Load Model
# ---------------------
device = torch.device("cpu")
cfg = get_config()
cfg["preload"] = "02"  # checkpoint name

model, tokenizer_src, tokenizer_trg = load_model_and_tokenizers(cfg, device)

def translate_fn(text):
    return translate_sentence(model, tokenizer_src, tokenizer_trg, text, cfg, device)

# ---------------------
# UI
# ---------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue")) as demo:
    gr.Markdown(
        """
        # 🌍 Arabic → English Translation
        Translate Arabic sentences into English using a custom Transformer model.  
        ✨ Just type a sentence in Arabic below and hit **Translate!**
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_box = gr.Textbox(
                label="Arabic Input",
                placeholder="اكتب جملة بالعربي...",
                lines=3
            )
            translate_btn = gr.Button("🚀 Translate", variant="primary")

        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label="English Translation",
                placeholder="The translation will appear here...",
                lines=3
            )

    translate_btn.click(fn=translate_fn, inputs=input_box, outputs=output_box)

    gr.Markdown(
        """
        ---
        ✅ **Model Info:**  
        - Built with a Transformer architecture from scratch.  
        - Trained on parallel Arabic ↔ English text.  
        - Tokenized using [SentencePiece](https://github.com/google/sentencepiece).  

        🛠 *Created with ❤️ by Abdelrahman*
        """
    )

demo.launch()
