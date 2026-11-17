from __future__ import annotations

def build_model(semantic_dim=300, hidden_size=64):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install baseline dependencies with pip install -e .[baselines]") from error
    class LogRobust(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(semantic_dim, hidden_size, batch_first=True, bidirectional=True)
            self.attention = torch.nn.Linear(hidden_size * 2, 1)
            self.output = torch.nn.Linear(hidden_size * 2, 2)
        def forward(self, semantic_sequence):
            hidden, _ = self.lstm(semantic_sequence)
            weights = torch.softmax(self.attention(hidden).squeeze(-1), dim=1)
            context = (hidden * weights.unsqueeze(-1)).sum(dim=1)
            return self.output(context)
    return LogRobust()
