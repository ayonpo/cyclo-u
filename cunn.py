"""Enhanced Chat Brain: LSTM-based neural network for chatbot responses."""

try:
    import torch.nn as nn
    import torch
    import torch.nn.functional as F 

    class EnhancedChatBrain(nn.Module):
        def __init__(self, vocab_size, hidden_size=128, embedding_dim=64, num_layers=2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, hidden=None, return_probs=False):
            # understand words
            x = self.embedding(x)
            # Remember context
            lstm_out, hidden = self.lstm(x, hidden)
            # think of the response - output for each timestep
            logits = self.fc(self.dropout(lstm_out))  # (batch, seq_len, vocab_size)

            if return_probs:
                probs = F.softmax(logits, dim=-1)
                return probs, hidden
            else:
                return logits, hidden
        
        def predict_with_confidence(self, x):
            """Get prediction with confidence scores"""
            with torch.no_grad():
                probs, _ = self.forward(x, return_probs=True)
                top_probs, top_indices = torch.topk(probs, k=3, dim=1)
                return top_probs, top_indices

except Exception:
    # Provide a clear fallback when PyTorch is not installed so imports don't fail.
    class EnhancedChatBrain:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for EnhancedChatBrain. Install torch to use the model.")
