# Quick Start Guide

This guide will help you get up and running with the Arabic-English Translation Transformer in just a few minutes.

## 🚀 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- GPU with CUDA support (optional but recommended)

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/transformer-arabic-english-translation.git
cd transformer-arabic-english-translation
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📁 Download Required Files

### Download Tokenizers

1. Go to [Google Drive - Tokenizers](https://drive.google.com/drive/folders/1VpmEeEwo6OZZ4nGuw5hDnAKsRqvlDOv4?usp=sharing)
2. Download these files:
   - `spm_ar_unigram.model`
   - `spm_ar_unigram.vocab`
   - `spm_en_unigram.model`
   - `spm_en_unigram.vocab`
3. Place them in the `tokenizer_models/` directory

### Download Model Weights

1. Go to [Google Drive - Model Weights](https://drive.google.com/drive/folders/1dcu8r-c28E3-V7cs0ArpNu5gWn_VjHP2?usp=sharing)
2. Download these files:
   - `tmodel_00.pt`
   - `tmodel_01.pt`
   - `tmodel_02.pt`
3. Place them in the `weights/` directory

## 🎯 Quick Usage

### Option 1: Command Line Interface

```bash
python main.py
```

Then type Arabic text and press Enter:

```
> مرحبا بالعالم
EN: Hello world

> أنا أحب البرمجة
EN: I love programming

> كيف حالك؟
EN: How are you?
```

### Option 2: Web Interface

```bash
python app.py
```

This will start a Gradio web interface. Open your browser and go to the URL shown in the terminal (usually `http://127.0.0.1:7860`).

### Option 3: Python Script

```python
import torch
from src.inference import load_model_and_tokenizers, translate_sentence
from src.config import get_config

# Load model
cfg = get_config()
device = torch.device("cpu")  # or "cuda" if you have GPU
model, tokenizer_src, tokenizer_trg = load_model_and_tokenizers(cfg, device)

# Translate
arabic_text = "مرحبا بالعالم"
english_translation = translate_sentence(
    model, tokenizer_src, tokenizer_trg, arabic_text, cfg, device
)
print(f"Arabic: {arabic_text}")
print(f"English: {english_translation}")
```

## 🔧 Configuration

You can modify the model configuration in `src/config.py`:

```python
def get_config():
    return {
        "batch_size": 128,        # Training batch size
        "num_epochs": 10,         # Number of training epochs
        "lr": 10**-4,             # Learning rate
        "seq_len": 80,            # Maximum sequence length
        "d_model": 512,           # Hidden dimension
        "src_vocab_size": 32000,  # Arabic vocabulary size
        "trg_vocab_size": 26000,  # English vocabulary size
        "preload": "02",          # Which checkpoint to load
    }
```

## 🏃‍♂️ Training (Optional)

If you want to train the model yourself:

```bash
python src/train.py
```

**Note**: Training requires significant computational resources and time. The provided pre-trained weights are ready to use.

## 📊 Evaluation

To evaluate the model on test data:

```bash
python src/validation.py
```

This will compute BLEU, WER, and CER metrics on the test set.

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'src'"**
   - Make sure you're in the project root directory
   - Check that your virtual environment is activated

2. **"FileNotFoundError: tokenizer model"**
   - Ensure you've downloaded the tokenizer files
   - Check that they're in the `tokenizer_models/` directory

3. **"FileNotFoundError: model weights"**
   - Ensure you've downloaded the model weights
   - Check that they're in the `weights/` directory

4. **CUDA out of memory**
   - Use CPU instead: `device = torch.device("cpu")`
   - Reduce batch size in config

5. **Poor translation quality**
   - The model was trained for only 3 epochs
   - Try different checkpoints (change `preload` in config)
   - Some Arabic text may need preprocessing

### Getting Help

- Check the [main README](README.md) for detailed documentation
- Look at the [architecture documentation](ARCHITECTURE.md) for technical details
- Open an issue on GitHub if you encounter bugs

## 🎉 Next Steps

Now that you have the model running, you can:

1. **Experiment with different Arabic texts**
2. **Try the web interface** for a better user experience
3. **Explore the code** to understand how Transformers work
4. **Modify the configuration** to experiment with different settings
5. **Train your own model** with different data or hyperparameters

## 📚 Learn More

- [Architecture Documentation](ARCHITECTURE.md) - Detailed technical explanation
- [Contributing Guide](CONTRIBUTING.md) - How to contribute to the project
- [Original Transformer Paper](https://arxiv.org/abs/1706.03762) - The foundational research

Happy translating! 🚀
