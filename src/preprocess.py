from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import torch

# install nltk tokenizer
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# CUDA acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset with 50/50 split
full_dataframe  = pd.read_csv("data/IMDB Dataset.csv")
train_dataframe = full_dataframe.iloc[:25000]
test_dataframe  = full_dataframe.iloc[25000:]

# text normalization
def normalize_text(raw_text: str):
    lowercased_text = raw_text.lower()
    return re.sub(r"[^a-z\s]", "", lowercased_text)

# normalization
train_text_list = train_dataframe["review"].apply(normalize_text).tolist()
test_text_list  = test_dataframe["review"].apply(normalize_text).tolist()

# tokenization
train_token_sequences = [word_tokenize(text) for text in train_text_list]
test_token_sequences  = [word_tokenize(text) for text in test_text_list]

# top 10,000 most frequent words
all_training_tokens = [token for sequence in train_token_sequences for token in sequence]
token_frequencies   = Counter(all_training_tokens)
vocabulary_list     = [word for word, _ in token_frequencies.most_common(10000)]

vocabulary_token_to_index          = {word: (idx + 1) for idx, word in enumerate(vocabulary_list)}
vocabulary_token_to_index["<OOV>"] = len(vocabulary_token_to_index)

# convert tokens to index ids
def map_tokens_to_ids(tokens):

    # OOV handling
    oov_index = vocabulary_token_to_index["<OOV>"]

    return [vocabulary_token_to_index.get(token, oov_index) for token in tokens]

# map tokens to ids
train_index_sequences = [map_tokens_to_ids(sequence) for sequence in train_token_sequences]
test_index_sequences  = [map_tokens_to_ids(sequence) for sequence in test_token_sequences]

# truncate to fixed length
def pad_or_truncate(sequences, max_sequence_length: int):
    num_samples   = len(sequences)
    output_tensor = torch.zeros((num_samples, max_sequence_length), dtype=torch.long, device=device)

    # fill row with sequence indices
    for row_index, sequence in enumerate(sequences):
        trimmed = sequence[:max_sequence_length]
        output_tensor[row_index, :len(trimmed)] = torch.tensor(trimmed, dtype=torch.long, device=device)

    return output_tensor

# tensors in 25, 50, 100 variants
padded_train_sequences_25  = pad_or_truncate(train_index_sequences, 25)
padded_train_sequences_50  = pad_or_truncate(train_index_sequences, 50)
padded_train_sequences_100 = pad_or_truncate(train_index_sequences, 100)

padded_test_sequences_25  = pad_or_truncate(test_index_sequences, 25)
padded_test_sequences_50  = pad_or_truncate(test_index_sequences, 50)
padded_test_sequences_100 = pad_or_truncate(test_index_sequences, 100)

# quick summary for sanity check
print({
        "Train samples": len(train_index_sequences),
        "Test samples": len(test_index_sequences),
        "Vocab size": len(vocabulary_token_to_index),
        "Example tensor shape": padded_train_sequences_50.shape,
        "Tensor device": padded_train_sequences_50.device
    })
