import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLangaugeModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next), dim=1)
        return idx


checkpoint = torch.load("models/BigramLM/bigram_model.pth")
vocab_size = checkpoint["vocab_size"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]


m = BigramLangaugeModel(vocab_size)
m.load_state_dict(checkpoint["model_state_dict"])
m.eval()


encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


# GENERATE
idx = torch.zeros((1,1), dtype=torch.long) # just a tensor having a single "0" to start the generation
print(decode(m.generate(idx,max_new_tokens=1000)[0].tolist()))