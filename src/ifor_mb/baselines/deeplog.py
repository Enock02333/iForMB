from __future__ import annotations
import numpy as np

def make_next_event_examples(sequences, history=10):
    windows, targets = [], []
    for sequence in sequences:
        tokens = [int(token) for token in sequence if int(token) != 0]
        for index in range(history, len(tokens)):
            windows.append(tokens[index-history:index])
            targets.append(tokens[index])
    return np.asarray(windows, dtype=np.int64), np.asarray(targets, dtype=np.int64)

def build_model(vocabulary_size, embedding_dim=32, hidden_size=64):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install baseline dependencies with pip install -e .[baselines]") from error
    class DeepLog(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
            self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, batch_first=True)
            self.output = torch.nn.Linear(hidden_size, vocabulary_size)
        def forward(self, history):
            hidden, _ = self.lstm(self.embedding(history))
            return self.output(hidden[:, -1])
    return DeepLog()
