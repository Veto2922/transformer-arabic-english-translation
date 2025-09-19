import torch

from data_loader import BOS_ID, EOS_ID
from utils.build_trg_mask import build_trg_mask


def greedy_decode(model, src, src_mask, max_len, device):
    """
    Greedy decoding: iteratively select argmax token until EOS or max_len.
    Returns a 1D tensor of generated ids (including BOS/EOS when present).
    """
    sos_idx = BOS_ID
    eos_idx = EOS_ID

    encoder_output = model.encode(src, src_mask)

    decoder_input = torch.empty(1, 1, dtype=src.dtype, device=device).fill_(sos_idx)

    while True:
        if decoder_input.size(1) >= max_len:
            break

        trg_mask = build_trg_mask(decoder_input)
        decoder_output = model.decode(encoder_output, src_mask, decoder_input, trg_mask)
        logits = model.project(decoder_output)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        decoder_input = torch.cat([decoder_input, next_token], dim=1)

        if next_token.item() == eos_idx:
            break

    return decoder_input.squeeze(0)



def beam_search_decode(model, src, src_mask, max_len, device, beam_size=3, length_norm=True):
    sos_idx = BOS_ID
    eos_idx = EOS_ID

    encoder_output = model.encode(src, src_mask)

    # نبدأ بـ beam واحد فيه <BOS>
    beams = [(torch.tensor([[sos_idx]], dtype=src.dtype, device=device), 0.0)]  # (sequence, log_prob)

    completed_sequences = []

    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_idx:  # لو خلصنا
                completed_sequences.append((seq, score))
                continue

            trg_mask = build_trg_mask(seq)
            decoder_output = model.decode(encoder_output, src_mask, seq, trg_mask)
            logits = model.project(decoder_output)
            next_token_logits = logits[:, -1, :]  # الاحتمالات للكلمة الجاية
            probs = torch.log_softmax(next_token_logits, dim=-1)

            # خد أفضل beam_size كلمات
            topk_probs, topk_ids = torch.topk(probs, beam_size, dim=-1)

            for i in range(beam_size):
                new_seq = torch.cat([seq, topk_ids[:, i].unsqueeze(0)], dim=1)
                new_score = score + topk_probs[0, i].item()
                new_beams.append((new_seq, new_score))

        # رتب بالأعلى log_prob واحتفظ بأفضل beam_size
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        beams = new_beams

        # لو كله خلص بـ EOS
        if all(seq[0, -1].item() == eos_idx for seq, _ in beams):
            completed_sequences.extend(beams)
            break

    # لو مفيش جمل كاملة (EOS)، نعتبر beams
    if not completed_sequences:
        completed_sequences = beams

    # رتّب النتايج (مع length normalization لو عايز)
    if length_norm:
        completed_sequences = [(seq, score / len(seq[0])) for seq, score in completed_sequences]

    best_seq = sorted(completed_sequences, key=lambda x: x[1], reverse=True)[0][0]

    return best_seq.squeeze(0)
