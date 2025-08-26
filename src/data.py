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

        # Load tokenizer and check its properties
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Tokenizer vocabulary size: {self.tokenizer.vocab_size}")
        print(
            f"Special tokens - BOS: {self.tokenizer.bos_token_id}, EOS: {self.tokenizer.eos_token_id}, PAD: {self.tokenizer.pad_token_id}"
        )

        self.data = self._preprocess_data()

    def _load_data(self) -> pd.DataFrame:
        file_path = DATA_DIR / f"{self.split}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        df = pd.read_csv(file_path, on_bad_lines="skip")
        df = df.sort_values(["conv_id", "utterance_idx"]).reset_index(drop=True)
        return df

    def _preprocess_data(self) -> List[Dict[str, torch.Tensor]]:
        df = self._load_data()
        df["utterance"] = df["utterance"].str.replace("_comma_", ",")

        # Group conversations
        conversations = {}
        for _, row in df.iterrows():
            conv_id = row["conv_id"]
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(row["utterance"])

        # Create input-output pairs
        pairs = []
        for conv_id, utterances in conversations.items():
            if len(utterances) < 2:
                continue

            # Create pairs: patient (even indices) -> therapist (odd indices)
            for i in range(0, len(utterances) - 1, 2):
                if i + 1 < len(utterances):
                    patient_text = utterances[i]
                    therapist_text = utterances[i + 1]
                    pairs.append((patient_text, therapist_text))

        print(f"Created {len(pairs)} input-output pairs")

        # Tokenize all pairs at once and check for issues
        processed_pairs = []
        for i, (src_text, tgt_text) in enumerate(pairs):
            # Tokenize source (patient)
            src_tokens = self.tokenizer(
                src_text,
                padding=False,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            )

            # Tokenize target (therapist)
            tgt_tokens = self.tokenizer(
                tgt_text,
                padding=False,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            )

            # Validate token IDs
            src_ids = src_tokens["input_ids"][0]
            tgt_ids = tgt_tokens["input_ids"][0]

            if src_ids.max().item() >= self.tokenizer.vocab_size:
                print(
                    f"WARNING: Skipping pair {i} - source token ID {src_ids.max().item()} >= vocab_size {self.tokenizer.vocab_size}"
                )
                continue

            if tgt_ids.max().item() >= self.tokenizer.vocab_size:
                print(
                    f"WARNING: Skipping pair {i} - target token ID {tgt_ids.max().item()} >= vocab_size {self.tokenizer.vocab_size}"
                )
                continue

            processed_pairs.append(
                {
                    "src_input_ids": src_ids,
                    "src_attention_mask": src_tokens["attention_mask"][0],
                    "tgt_input_ids": tgt_ids,
                    "tgt_attention_mask": tgt_tokens["attention_mask"][0],
                }
            )

        print(f"Successfully processed {len(processed_pairs)} valid pairs")
        return processed_pairs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        # Pad sequences to max length
        src_input_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
        src_attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        tgt_input_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
        tgt_attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)

        # Fill with actual data
        src_len = min(len(item["src_input_ids"]), self.max_seq_len)
        tgt_len = min(len(item["tgt_input_ids"]), self.max_seq_len)

        src_input_ids[:src_len] = item["src_input_ids"][:src_len]
        src_attention_mask[:src_len] = item["src_attention_mask"][:src_len]
        tgt_input_ids[:tgt_len] = item["tgt_input_ids"][:tgt_len]
        tgt_attention_mask[:tgt_len] = item["tgt_attention_mask"][:tgt_len]

        # Create labels for training (ignore padding tokens)
        labels = tgt_input_ids.clone()
        labels[tgt_attention_mask == 0] = -100

        # Final safety check
        if (
            src_input_ids.max().item() >= self.tokenizer.vocab_size
            or tgt_input_ids.max().item() >= self.tokenizer.vocab_size
        ):
            # Replace invalid tokens with UNK token
            src_input_ids = torch.clamp(src_input_ids, 0, self.tokenizer.vocab_size - 1)
            tgt_input_ids = torch.clamp(tgt_input_ids, 0, self.tokenizer.vocab_size - 1)
            labels = torch.clamp(labels, -100, self.tokenizer.vocab_size - 1)

        return {
            "input_ids": src_input_ids,
            "attention_mask": src_attention_mask,
            "labels": labels,
        }
