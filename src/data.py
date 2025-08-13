from torch.utils.data import Dataset
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "input" / "empatheticdialogues"


class EmpatheticConv(Dataset):
    def __init__(
        self,
        split: str = "train",
        max_seq_len: int = 128,
        tokenizer_name: str = "facebook/blenderbot-400M-distill",
    ):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Add special tokens if they don't exist
        special_tokens_dict = {}
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = "<pad>"
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = "<unk>"

        # Add custom tokens for generation
        additional_special_tokens = []
        if "<start>" not in self.tokenizer.get_vocab():
            additional_special_tokens.append("<start>")
        if "<end>" not in self.tokenizer.get_vocab():
            additional_special_tokens.append("<end>")

        if additional_special_tokens:
            special_tokens_dict["additional_special_tokens"] = additional_special_tokens

        if special_tokens_dict:
            num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Added {num_added_tokens} special tokens to tokenizer")
            print(f"New vocabulary size: {self.tokenizer.vocab_size}")

        self.src, self.tgt = self._preprocess_data()

    def _load_data(self) -> pd.DataFrame:
        file_path = DATA_DIR / f"{self.split}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        df = pd.read_csv(file_path, on_bad_lines="skip")
        df = df.sort_values(["conv_id", "utterance_idx"]).reset_index(drop=True)
        return df

    def _preprocess_data(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        df = self._load_data()
        df["utterance"] = df["utterance"].str.replace("_comma_", ",")
        convs = list()
        for conv_id in df["conv_id"].unique().tolist():
            df_conv = df[df["conv_id"] == conv_id]
            convs.append(df_conv["utterance"].tolist())

        patient = []
        therapist = []

        for conv in convs:
            if len(conv) == 1:
                continue
            if len(conv) % 2 == 1:
                conv.pop()
            patient += conv[0::2]
            therapist += conv[1::2]

        patient_encoding = self.tokenizer(
            patient,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        therapist_encoding = self.tokenizer(
            therapist,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return patient_encoding, therapist_encoding

    def __len__(self) -> int:
        return len(self.src["input_ids"])

    def __getitem__(self, index: int):
        src_input_ids = self.src["input_ids"][index]
        src_attention_mask = self.src["attention_mask"][index]
        tgt_input_ids = self.tgt["input_ids"][index]
        tgt_attention_mask = self.tgt["attention_mask"][index]

        labels = tgt_input_ids.clone()
        labels[tgt_attention_mask == 0] = -100

        return {
            "input_ids": src_input_ids,
            "attention_mask": src_attention_mask,
            "labels": labels,
        }
