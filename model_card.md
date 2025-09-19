---
language:
- ar
- en
license: mit
tags:
- translation
- transformer
- arabic
- english
- machine-translation
- pytorch
- attention
metrics:
- bleu
- wer
- cer
datasets:
- opus-100
model-index:
- name: Arabic-English Translation Transformer
  results:
  - task:
      type: translation
      name: Arabic to English Translation
    dataset:
      type: opus-100
      name: OPUS-100 Arabic-English
    metrics:
    - type: bleu
      value: 0.237
      name: BLEU Score
    - type: wer
      value: 0.701
      name: Word Error Rate
    - type: cer
      value: 0.516
      name: Character Error Rate
---

# Arabic-English Translation Transformer

A complete implementation of the Transformer architecture from scratch for Arabic-to-English machine translation, built with PyTorch.

## Model Description

This model is a sequence-to-sequence Transformer that translates Arabic text to English. It implements every component of the original "Attention Is All You Need" paper, including:

- Multi-head attention mechanism
- Positional encodings
- Encoder-decoder architecture
- Residual connections and layer normalization
- Custom tokenization for Arabic and English

### Model Architecture

- **Parameters**: ~72M parameters
- **Layers**: 4 encoder + 4 decoder layers
- **Attention Heads**: 4 heads per layer
- **Hidden Dimension**: 512
- **Vocabulary Sizes**: 32K (Arabic), 26K (English)
- **Sequence Length**: 80 tokens maximum

## Training Data

The model was trained on the OPUS-100 Arabic-English parallel corpus, which contains approximately 1 million sentence pairs.

## Usage

### Python

```python
import torch
from src.inference import load_model_and_tokenizers, translate_sentence
from src.config import get_config

# Load model
cfg = get_config()
device = torch.device("cpu")
model, tokenizer_src, tokenizer_trg = load_model_and_tokenizers(cfg, device)

# Translate
arabic_text = "مرحبا بالعالم"
english_translation = translate_sentence(
    model, tokenizer_src, tokenizer_trg, arabic_text, cfg, device
)
print(english_translation)  # "Hello world"
```

### Command Line

```bash
python main.py
```

### Web Interface

```bash
python app.py
```

## Performance

After 3 epochs of training:

| Metric | Greedy Decoding | Beam Search (k=3) |
|--------|----------------|-------------------|
| BLEU   | 0.225          | 0.237             |
| WER    | 0.694          | 0.701             |
| CER    | 0.509          | 0.516             |

## Limitations

- The model was trained for only 3 epochs and may benefit from longer training
- Performance is limited compared to larger pre-trained models
- Arabic text preprocessing removes diacritics, which may affect some translations
- Maximum sequence length is limited to 80 tokens



## License

This model is licensed under the MIT License.
