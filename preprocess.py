import os
import argparse
import pandas as pd
from config import RAW_DATA_DIR
import torch
from torch.utils.data import Dataset, DataLoader
import re
from typing import List, Tuple, Dict
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset


# Load the dataset
wmt_14 = pd.read_csv(RAW_DATA_DIR/"wmt14_translate_de-en_train.csv", encoding="utf-8", on_bad_lines='skip', engine='Python')
subset = wmt_14[:1000]


# Loading the tokenizer
tokenizer  = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

def tokenize_function(df):
    model_inputs = tokenizer(
        df["en"], padding="max_length", truncation=True, max_length=64
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            df["de"], padding="max_length", truncation=True, max_length=64
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Converting the pandas dataframe to a huggingface dataset
hf_dataset = Dataset.from_pandas(subset)
tokenized_dataset = hf_dataset.map(tokenize_function, batched=False)





# Training data
train_data = wmt_14["train"]

class Vocabulary:
    def __init__(self, name: str):
        self.name = name
        self.word2index = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.word2count = {}
        self.index2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.n_words = 4  # Count special tokens

    def add_sentence(self, sentence: str):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def normalize_string(s: str) -> str:
    """Normalize string by converting to lowercase and removing extra whitespace."""
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

class TranslationDataset(Dataset):
    def __init__(self, source_sentences: List[str], target_sentences: List[str], 
                 source_vocab: Vocabulary, target_vocab: Vocabulary, max_len: int = 50):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.source_sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.source_sentences[idx]
        target = self.target_sentences[idx]

        # Convert words to indices
        source_indices = [self.source_vocab.word2index.get(word, self.source_vocab.word2index["<unk>"]) 
                         for word in source.split()]
        target_indices = [self.target_vocab.word2index.get(word, self.target_vocab.word2index["<unk>"]) 
                         for word in target.split()]

        # Add special tokens
        source_indices = [self.source_vocab.word2index["<sos>"]] + source_indices + [self.source_vocab.word2index["<eos>"]]
        target_indices = [self.target_vocab.word2index["<sos>"]] + target_indices + [self.target_vocab.word2index["<eos>"]]

        # Pad sequences
        source_indices = self._pad_sequence(source_indices, self.max_len, self.source_vocab.word2index["<pad>"])
        target_indices = self._pad_sequence(target_indices, self.max_len, self.target_vocab.word2index["<pad>"])

        return torch.tensor(source_indices), torch.tensor(target_indices)

    def _pad_sequence(self, sequence: List[int], max_len: int, pad_idx: int) -> List[int]:
        if len(sequence) > max_len:
            return sequence[:max_len]
        return sequence + [pad_idx] * (max_len - len(sequence))

def create_dataloader(source_sentences: List[str], target_sentences: List[str], 
                     batch_size: int = 32, max_len: int = 50) -> Tuple[DataLoader, Vocabulary, Vocabulary]:
    # Create vocabularies
    source_vocab = Vocabulary("source")
    target_vocab = Vocabulary("target")

    # Add sentences to vocabularies
    for src, tgt in zip(source_sentences, target_sentences):
        source_vocab.add_sentence(src)
        target_vocab.add_sentence(tgt)

    # Create dataset
    dataset = TranslationDataset(source_sentences, target_sentences, 
                               source_vocab, target_vocab, max_len)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, source_vocab, target_vocab

def load_data(file_path: str) -> Tuple[List[str], List[str]]:
    """Load and preprocess data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Assuming each line contains a source-target pair separated by a tab
    source_sentences = []
    target_sentences = []
    
    for line in lines:
        source, target = line.strip().split('\t')
        source_sentences.append(normalize_string(source))
        target_sentences.append(normalize_string(target))
    
    return source_sentences, target_sentences

# 