import os
import torch
from datasets import load_dataset
from preprocess import create_dataloader, load_data
import pickle

def download_wmt14():
    """Download and prepare the WMT14 English-German dataset."""
    print("Downloading WMT14 dataset...")
    dataset = load_dataset("wmt14", "de-en", split="train")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save raw data
    with open("data/wmt14_raw.txt", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(f"{item['en']}\t{item['de']}\n")
    
    print("Dataset downloaded and saved to data/wmt14_raw.txt")
    return "data/wmt14_raw.txt"

def prepare_data(file_path: str, batch_size: int = 32, max_len: int = 50):
    """Prepare the data for training."""
    print("Preparing data...")
    
    # Load and preprocess data
    source_sentences, target_sentences = load_data(file_path)
    
    # Create dataloader and vocabularies
    dataloader, source_vocab, target_vocab = create_dataloader(
        source_sentences, target_sentences, batch_size, max_len
    )
    
    # Save vocabularies
    with open("data/source_vocab.pkl", "wb") as f:
        pickle.dump(source_vocab, f)
    with open("data/target_vocab.pkl", "wb") as f:
        pickle.dump(target_vocab, f)
    
    print(f"Vocabulary sizes - Source: {source_vocab.n_words}, Target: {target_vocab.n_words}")
    return dataloader, source_vocab, target_vocab

if __name__ == "__main__":
    # Download and prepare data
    data_file = download_wmt14()
    dataloader, source_vocab, target_vocab = prepare_data(data_file)
    
    # Print sample batch
    for batch_idx, (source, target) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Source shape: {source.shape}")
        print(f"Target shape: {target.shape}")
        break 