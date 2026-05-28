import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _load_ids(ids_path):
    with open(ids_path, "rb") as ids_file:
        return pickle.load(ids_file)


class RetrieverFitDataset(Dataset):
    """Retriever Fit Dataset.
    """

    def __init__(self,
                 samples,
                 ids_path,
                 labels_descriptions,
                 pseudo_labels,
                 text_features_source,
                 label_enhancement,
                 tokenizer,
                 text_max_length,
                 label_max_length,
                 amount=1.0
                 ):

        super(RetrieverFitDataset, self).__init__()

        self.samples = []
        self.text_features_source = text_features_source
        self.label_enhancement = label_enhancement
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.label_max_length = label_max_length
        self.labels_descriptions = labels_descriptions
        self.pseudo_labels = pseudo_labels

        for idx in tqdm(_load_ids(ids_path), desc="Reading samples"):
            for label_idx, label in zip(samples[idx]["labels_ids"], samples[idx]["labels"]):
                
                raw_text = self._get_text_features(samples[idx])

                span, text = self._split_span_text(raw_text)

                self.samples.append({
                    "text_idx": samples[idx]["text_idx"],

                    "raw_text": raw_text,

                    "span": span,

                    "context": text,

                    "label_idx": label_idx,

                    "label": f"{label} " + self._get_label_features(label_idx)
                })

        self.samples = self.samples[:round(amount * len(self.samples))]
        
    def _split_span_text(self, raw_text):
        parts = raw_text.split(":", 1)

        if len(parts) != 2:
            return "", raw_text

        span = parts[0].strip()
        text = parts[1].strip()

        return span, text

    def _get_text_features(self, sample):
        if self.text_features_source == "KWD":
            return " ".join([kwd[0] for kwd in sample["keywords"]])
        elif self.text_features_source == "TXT":
            return sample["text"]
        else:
            raise Exception("Source features must be TXT or KWD")

    def _get_label_features(self, label_idx):
        if self.label_enhancement == "PMI":
            return self._get_pseudo_labels(label_idx)
        elif self.label_enhancement == "LLM":
            return self.labels_descriptions.get(label_idx, "")
        elif self.label_enhancement == "NONE":
            return ""
        else:
            raise Exception("Source features must be in [PMI, LLM, NONE]")

    def _get_pseudo_labels(self, label_idx):
        if self.label_enhancement:
            if label_idx in self.pseudo_labels:
                terms_scores = sorted(self.pseudo_labels[label_idx], key=lambda x: x[1], reverse=True)
                return " ".join(x[0] for x in terms_scores)
        return ""

    def _encode(self, sample):
        # print(f"\n\n{sample['label']}\n\n")
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
                    text=sample["raw_text"],
                    max_length=self.text_max_length,
                    padding="max_length",
                    truncation=True
                )
            ),
            "label_idx": sample["label_idx"],
            "label": torch.tensor(
                self.tokenizer.encode(
                    text=sample["label"], max_length=self.label_max_length, padding="max_length", truncation=True
                ))
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )
