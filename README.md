
# Arabic-English Translation Transformer

A complete implementation of the Transformer architecture from scratch for **Arabic-to-English machine translation**, built with PyTorch. This project implements every component of the original *"Attention Is All You Need"* paper, including multi-head attention, positional encodings, and encoder-decoder architecture, specifically designed for Arabic text processing.

Beyond merely replicating the original design, this endeavor also incorporated several modern practices and enhancements to improve stability and efficiency. These included using learned positional embeddings for greater flexibility, implementing pre-normalization for enhanced training stability, employing SentencePiece with a Unigram model for robust tokenization of Arabic, and integrating training optimizations such as label smoothing and mixed-precision (FP16) training with a gradient scaler.

📖 **Full Project Article**: [Building a Transformer from Scratch for Arabic-English Translation](https://medium.com/@abdelrahman.m2922/building-a-transformer-from-scratch-for-arabic-english-translation-1790864e55b0)

🤗 **Model & Tokenizers on Hugging Face**: [Arabic-English Transformer](https://huggingface.co/Abdelrahman2922/arabic-english-transformer)

<img width="1901" height="573" alt="image" src="https://github.com/user-attachments/assets/1d2147aa-a2a0-4903-99e7-2a1c95913418" />

---

## 🌟 Features

* **Complete Transformer Implementation**: Built from scratch following the original paper
* **Arabic-Specific Optimizations**: Handles Arabic diacritics, morphology, and linguistic features
* **Custom Tokenization**: SentencePiece Unigram tokenizers trained specifically for Arabic and English
* **Modern Training Techniques**: Mixed precision training, label smoothing, and gradient scaling
* **Interactive Demo**: Gradio-based web interface for real-time translation
* **Comprehensive Evaluation**: BLEU, WER, and CER metrics with attention visualization
* **Hugging Face Compatible**: Pretrained weights and tokenizers available on [Hugging Face](https://huggingface.co/Abdelrahman2922/arabic-english-transformer)

---

## 📊 Model Architecture

* **Parameters**: \~72M
* **Architecture**: 4-layer encoder-decoder with 4 attention heads
* **Hidden Dimension**: 512
* **Vocabulary Sizes**: 32K (Arabic), 26K (English)
* **Sequence Length**: 80 tokens maximum
* **Training Data**: OPUS-100 Arabic-English parallel corpus (\~1M sentence pairs)

---

## 📈 Performance

* **BLEU Score**: 0.225 (Greedy), 0.237 (Beam Search)
* **Training**: 3 epochs on Google Colab (T4 GPU)
* **Time per epoch**: \~45 minutes
* **Model Size**: 72M parameters (\~4GB GPU memory)

## 🚀 Quick Start

### Prerequisites

* Python 3.8+
* PyTorch 2.7.1+
* CUDA-compatible GPU (recommended)

### Installation

```bash
git clone https://github.com/Veto2922/transformer-arabic-english-translation.git
cd transformer-arabic-english-translation
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Download Required Files

You can download tokenizers and pretrained weights directly from Hugging Face:
👉 [Hugging Face Model Hub](https://huggingface.co/Abdelrahman2922/arabic-english-transformer)

Or from Google Drive (legacy links):

* [Tokenizers](https://drive.google.com/drive/folders/1VpmEeEwo6OZZ4nGuw5hDnAKsRqvlDOv4?usp=sharing)
* [Model Weights](https://drive.google.com/drive/folders/1dcu8r-c28E3-V7cs0ArpNu5gWn_VjHP2?usp=sharing)

---

## 🔧 Usage

### 1. Command Line

```bash
python main.py
```

### 2. Web Interface (Gradio)

```bash
python app.py
```

### 3. Training

```bash
python src/train.py
```

---

## 🏗️ Project Structure

```
transformer-arabic-english-translation/
├── src/
│   ├── transformer_blocks/       # Core Transformer components
│   ├── utils/                    # Utility functions
│   ├── transformer_model.py      # Full model assembly
│   ├── train.py                  # Training script
│   ├── inference.py              # Inference utilities
│   ├── validation.py             # Evaluation
│   └── config.py                 # Configuration
├── Data/                         # Dataset files
├── tokenizer_models/             # SentencePiece models
├── weights/                      # Model checkpoints
├── Notebooks/                    # Jupyter notebooks
├── app.py                        # Gradio app
├── main.py                       # CLI interface
└── requirements.txt              # Dependencies
```


---

## 🔬 Attention Visualization

The project includes **attention visualization** to better understand translation behavior:

* Encoder self-attention (Arabic word relationships)
* Decoder self-attention (English generation patterns)
* Cross-attention (Arabic-English word alignments)

---

## 🔗 Links

* 📖 [Project Article](https://medium.com/@abdelrahman.m2922/building-a-transformer-from-scratch-for-arabic-english-translation-1790864e55b0)
* 🤗 [Hugging Face Model Hub](https://huggingface.co/Abdelrahman2922/arabic-english-transformer)
* 📂 [Google Drive - Tokenizers](https://drive.google.com/drive/folders/1VpmEeEwo6OZZ4nGuw5hDnAKsRqvlDOv4?usp=sharing)
* 📂 [Google Drive - Weights](https://drive.google.com/drive/folders/1dcu8r-c28E3-V7cs0ArpNu5gWn_VjHP2?usp=sharing)
* 📑 [Original Transformer Paper](https://arxiv.org/abs/1706.03762)

---

⚠️ **Note**: This is a research and educational project. For production use, consider fine-tuning larger pre-trained models or using established translation services.


