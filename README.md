# Arabic-English Translation Transformer

A complete implementation of the Transformer architecture from scratch for Arabic-to-English machine translation, built with PyTorch. This project implements every component of the original "Attention Is All You Need" paper, including multi-head attention, positional encodings, and encoder-decoder architecture, specifically designed for Arabic text processing.

## 🌟 Features

- **Complete Transformer Implementation**: Built from scratch following the original paper
- **Arabic-Specific Optimizations**: Handles Arabic diacritics, morphology, and linguistic features
- **Custom Tokenization**: SentencePiece Unigram tokenizers trained specifically for Arabic and English
- **Modern Training Techniques**: Mixed precision training, label smoothing, and gradient scaling
- **Interactive Demo**: Gradio-based web interface for real-time translation
- **Comprehensive Evaluation**: BLEU, WER, and CER metrics with attention visualization
- **Hugging Face Compatible**: Ready for model sharing and deployment

## 📊 Model Architecture

- **Parameters**: ~72M parameters
- **Architecture**: 4-layer encoder-decoder with 4 attention heads
- **Hidden Dimension**: 512
- **Vocabulary Sizes**: 32K (Arabic), 26K (English)
- **Sequence Length**: 80 tokens maximum
- **Training Data**: OPUS-100 Arabic-English parallel corpus (~1M sentence pairs)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.7.1+
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Veto2922/transformer-arabic-english-translation.git
cd transformer-arabic-english-translation
```

2. **Create and activate virtual environment:**

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### Download Required Files

Before running the model, you need to download the tokenizers and model weights:

#### Tokenizers

Download from: [Google Drive - Tokenizers](https://drive.google.com/drive/folders/1VpmEeEwo6OZZ4nGuw5hDnAKsRqvlDOv4?usp=sharing)

Place these files in the `tokenizer_models/` directory:

- `spm_ar_unigram.model`
- `spm_ar_unigram.vocab`
- `spm_en_unigram.model`
- `spm_en_unigram.vocab`

#### Model Weights

Download from: [Google Drive - Model Weights](https://drive.google.com/drive/folders/1dcu8r-c28E3-V7cs0ArpNu5gWn_VjHP2?usp=sharing)

Place these files in the `weights/` directory:

- `tmodel_00.pt`
- `tmodel_01.pt`
- `tmodel_02.pt`

### Usage

#### 1. Command Line Interface
```bash
python main.py
```

#### 2. Web Interface (Gradio)
```bash
python app.py
```

#### 3. Training
```bash
python src/train.py
```

## 🏗️ Project Structure

```
transformer-arabic-english-translation/
├── src/
│   ├── transformer_blocks/          # Core Transformer components
│   │   ├── encoder_block.py         # Encoder implementation
│   │   ├── decoder_block.py         # Decoder implementation
│   │   ├── multi_head_attention_block.py  # Multi-head attention
│   │   ├── feed_forward.py          # Feed-forward networks
│   │   ├── input_embedding.py       # Token embeddings
│   │   ├── positional_encoding.py   # Positional encodings
│   │   ├── layer_normalization.py   # Layer normalization
│   │   ├── residual_connections.py  # Skip connections
│   │   └── projection_layer.py      # Output projection
│   ├── utils/                       # Utility functions
│   │   ├── text.py                  # Text processing
│   │   ├── decoding.py              # Decoding strategies
│   │   ├── metrics.py               # Evaluation metrics
│   │   └── build_trg_mask.py        # Target masking
│   ├── transformer_model.py         # Full model assembly
│   ├── train.py                     # Training script
│   ├── inference.py                 # Inference utilities
│   ├── validation.py                # Validation and evaluation
│   └── config.py                    # Configuration
├── Data/                           # Dataset files
│   ├── raw_data/                   # Original parallel corpus
│   ├── cleaned_data/               # Preprocessed text
│   └── encoded_data/               # Tokenized sequences
├── tokenizer_models/               # SentencePiece models
├── weights/                        # Model checkpoints
├── Notebooks/                      # Jupyter notebooks for analysis
├── app.py                          # Gradio web interface
├── main.py                         # CLI interface
└── requirements.txt                # Dependencies
```

## 🔧 Key Components

### 1. Data Preprocessing
- **Arabic Text Normalization**: Removes diacritics, normalizes alef/hamza forms, converts Arabic digits
- **Custom Tokenization**: SentencePiece Unigram models trained on Arabic and English corpora
- **Data Loading**: Efficient PyTorch DataLoader with padding and masking

### 2. Transformer Architecture

#### Encoder
- **Multi-Head Self-Attention**: Captures relationships within Arabic sentences
- **Feed-Forward Networks**: Adds non-linearity and representational power
- **Residual Connections**: Prevents vanishing gradients
- **Layer Normalization**: Stabilizes training

#### Decoder
- **Masked Self-Attention**: Prevents looking at future tokens
- **Cross-Attention**: Attends to encoder outputs for translation
- **Feed-Forward Networks**: Similar to encoder
- **Residual Connections & Layer Norm**: Same as encoder

### 3. Training Features
- **Mixed Precision Training**: FP16 for faster training and lower memory usage
- **Label Smoothing**: Prevents overconfidence and improves generalization
- **Teacher Forcing**: Uses ground truth during training for stability
- **Gradient Scaling**: Prevents underflow in mixed precision training

### 4. Evaluation & Inference
- **Greedy Decoding**: Fast, deterministic translation
- **Beam Search**: More accurate but slower translation
- **BLEU Score**: Standard machine translation metric
- **Attention Visualization**: Understand model behavior

## 📈 Performance

After 3 epochs of training on Google Colab (T4 GPU):
- **BLEU Score**: 0.225 (Greedy), 0.237 (Beam Search)
- **Training Time**: ~45 minutes per epoch
- **Model Size**: 72M parameters
- **Memory Usage**: ~4GB GPU memory

## 🎯 Arabic-Specific Features

### Text Preprocessing
- **Diacritic Removal**: Handles inconsistent Arabic diacritics
- **Alef Normalization**: Unifies different forms of Arabic alef (أ, إ, آ → ا)
- **Yaa' Normalization**: Standardizes yaa' forms (ى → ي)
- **Digit Conversion**: Converts Arabic numerals to Latin numerals

### Tokenization
- **Separate Tokenizers**: Dedicated models for Arabic and English
- **Unigram Algorithm**: Better handling of Arabic morphology
- **Vocabulary Sizes**: 32K (Arabic), 26K (English)

## 🔬 Attention Visualization

The project includes attention visualization capabilities to understand how the model processes Arabic text:

- **Encoder Self-Attention**: Shows how Arabic words relate to each other
- **Decoder Self-Attention**: Reveals English word generation patterns
- **Cross-Attention**: Displays Arabic-English word alignments

## 🙏 Acknowledgments

- Original Transformer paper: "Attention Is All You Need" (Vaswani et al., 2017)
- OPUS-100 dataset for parallel Arabic-English corpus
- SentencePiece for subword tokenization
- PyTorch team for the deep learning framework



## 🔗 Links
- [Hugging Face Model Hub](https://huggingface.co/Abdelrahman2922/arabic-english-transformer)
- [Google Drive - Tokenizers](https://drive.google.com/drive/folders/1VpmEeEwo6OZZ4nGuw5hDnAKsRqvlDOv4?usp=sharing)
- [Google Drive - Model Weights](https://drive.google.com/drive/folders/1dcu8r-c28E3-V7cs0ArpNu5gWn_VjHP2?usp=sharing)
- [Original Transformer Paper](https://arxiv.org/abs/1706.03762)

---

**Note**: This is a research and educational project. For production use, consider fine-tuning larger pre-trained models or using established translation services."" 
