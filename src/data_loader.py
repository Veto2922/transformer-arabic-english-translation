import torch
from torch.utils.data import Dataset, DataLoader

PAD_ID = 0   # نفس اللي حددته في SentencePiece
BOS_ID = 1
EOS_ID = 2

class TranslationDataset(Dataset):
    def __init__(self, src_file, trg_file, max_len=50):
        self.src_data = self.load_file(src_file, max_len)
        self.trg_data = self.load_file(trg_file, max_len)
        assert len(self.src_data) == len(self.trg_data), "Source and Target not aligned!"
    
    def load_file(self, path, max_len):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ids = [int(x) for x in line.strip().split()]
                # إضافة BOS/EOS وقص حسب max_len
                ids = [BOS_ID] + ids[:max_len-2] + [EOS_ID]
                data.append(ids)
        return data

    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.trg_data[idx])

# collate function for padding + masks + shifting
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)

    # احسب max length في الباتش
    src_max_len = max([len(s) for s in src_batch])
    trg_max_len = max([len(t) for t in trg_batch])

    # padding
    src_padded = torch.full((len(batch), src_max_len), PAD_ID)
    trg_padded = torch.full((len(batch), trg_max_len), PAD_ID)

    for i, (src, trg) in enumerate(zip(src_batch, trg_batch)):
        src_padded[i, :len(src)] = src
        trg_padded[i, :len(trg)] = trg

    # attention masks (1 للـ tokens و 0 للـ padding)
    src_mask = (src_padded != PAD_ID).int()
    trg_mask = (trg_padded != PAD_ID).int()

    # decoder inputs (shifted right → يبدأ بـ BOS)
    decoder_input = trg_padded[:, :-1]
    # decoder targets (shifted left → ينتهي بـ EOS)
    decoder_target = trg_padded[:, 1:]

    return {
        "src_input": src_padded,
        "src_mask": src_mask,
        "decoder_input": decoder_input,
        "decoder_target": decoder_target,
        "trg_mask": trg_mask
    }

# train/valid DataLoader
train_dataset = TranslationDataset("../Data/encoded_data/train.ids.ar",
                                   "../Data/encoded_data/train.ids.en",
                                   max_len=80)

valid_dataset = TranslationDataset("../Data/encoded_data/validation.ids.ar",
                                   "../Data/encoded_data/validation.ids.en",
                                   max_len=80)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

print("✅ DataLoader ready")
