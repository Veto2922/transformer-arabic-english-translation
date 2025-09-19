from typing import List, Tuple
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.cer import CharErrorRate

def compute_text_metrics(predictions: List[str], references: List[str]) -> Tuple[float, float, float]:
    """
    Compute BLEU (up to 4-gram, smoothed), WER, and CER 
    over the entire corpus (not averaged per sample).

    Args:
        predictions: list of predicted strings
        references:  list of reference strings (same length as predictions)

    Returns:
        (bleu, wer, cer) as floats
    """
    if not predictions:
        return 0.0, 0.0, 0.0

    # Convert references to the format expected by BLEUScore:
    # List of list of references, e.g. [["ref1"], ["ref2"], ...]
    refs_nested = [[ref] for ref in references]

    bleu_metric = BLEUScore(n_gram=4, smooth=True)
    wer_metric = WordErrorRate()
    cer_metric = CharErrorRate()

    bleu_value = float(bleu_metric(predictions, refs_nested))
    wer_value = float(wer_metric(predictions, references))
    cer_value = float(cer_metric(predictions, references))

    return bleu_value, wer_value, cer_value
