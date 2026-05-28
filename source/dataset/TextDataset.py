import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TextDataset(Dataset):
    """Retriever Predict Dataset.
    """

    def __init__(self, samples, text_features_source, tokenizer, text_max_length):
        super(TextDataset, self).__init__()
        self.texts = []
        self.text_features_source = text_features_source
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length

        for sample in tqdm(samples, desc="Reading Texts"):
            
            raw_text = self._get_features(sample)

            span, text = self._split_span_text(raw_text)

            self.texts.append({
                "text_idx": sample["text_idx"],
                "span": span,
                "text": text,
            })
            
    def _split_span_text(self, raw_text):
        parts = raw_text.split(":", 1)

        if len(parts) != 2:
            return "", raw_text

        span = parts[0].strip()
        text = parts[1].strip()

        return span, text

    def _get_features(self, sample):
        if self.text_features_source == "KWD":
            return " ".join([kwd[0] for kwd in sample["keywords"]])
        elif self.text_features_source == "TXT":
            return sample["text"]
        else:
            raise Exception("Source features must be TXT or KWD")

    def _encode(self, sample):
        return {
            "text_idx": sample["text_idx"],
            "span": torch.tensor(
                self.tokenizer.encode(
                    text=sample["span"],
                    max_length=self.text_max_length // 4,
                    padding="max_length",
                    truncation=True
                )
            ),

            "text": torch.tensor(
                self.tokenizer.encode(
                    text=sample["text"],
                    max_length=self.text_max_length,
                    padding="max_length",
                    truncation=True
                )
            )
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self._encode(
            self.texts[idx]
        )
