import torch
import torch.nn as nn
from torch.nn import functional as F


# LOAD DATASET
with open("models/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# GET ALL UNIQUE CHARACTERS
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print("Unique Characters: ",''.join(chars))
# print("Vocab_Size: ",vocab_size) --> 65

# STRING-TO-INTEGER AND REVERSE MAPPING
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ENCODE COMPLETE DATASET
data = torch.tensor(encode(text), dtype=torch.long) # data.shape --> torch.Size([1115394])

# TRAIN-TEST SPLIT
n = int(0.9*len(data))
train_data = data[:n] # train_data.shape --> torch.Size([1003854])
val_data = data[n:] # val_data.shape --> torch.Size([111540])


# DATA BATCHES FUNC
def get_batch(split):
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y
"""
get_batch USAGE:
    xb,yb = get_batch("train")
    xb.shape --> [32,8] 32 separate occasions/batches of 8 continuous characters
    yb.shape --> [32,8] predicted next character for each character in previous matrix
"""



# HYPERPARAMETERS
batch_size = 32
block_size = 8

torch.manual_seed(1337) # For reproducibility across runs

# BIGRAM LANGUAGE MODEL
class BigramLangaugeModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size) #(65,65)

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

# MODEL
m = BigramLangaugeModel(vocab_size)
"""
logits, loss = m(xb, yb)
print(logits.shape) --> torch.Size([256, 65])
"""


# FUNCTION TO EVALUATE OVERALL TRAIN-VAL LOSS
@torch.no_grad()
def evaluate_full_dataset(split):
    data = train_data if split == 'train' else val_data
    m.eval()

    losses = []
    for i in range(0, len(data) - block_size,block_size):
        x = data[i: i+block_size].unsqueeze(0)
        y = data[i+1: i+block_size+1].unsqueeze(0)

        _, loss = m(x,y)

        losses.append(loss.item())

    m.train()
    return sum(losses)/len(losses)


# TRAINING
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(10001):
    xb, yb  = get_batch('train')            # batch

    logits, loss = m(xb,yb)                 # forward pass & loss calculation
    optimizer.zero_grad(set_to_none=True)   # zero_grad
    loss.backward()                         # backward pass
    optimizer.step()                        # optimize

    if steps % 500 ==0:
        print(f"Step {steps}", end=' | ')
        print(f"Batch Loss: {loss.item():.4f}",end=' | ')

        final_train_loss = evaluate_full_dataset('train')
        final_val_loss = evaluate_full_dataset('val')
        print(f"OVERALL LOSS:", end='\t')
        print(f"Train: {final_train_loss:.4f}",end=' ')
        print(f"Val: {final_val_loss:.4f}")


# SAVE THE MODEL
torch.save({
    'model_state_dict': m.state_dict(),
    'vocab_size': vocab_size,
    'block_size': block_size,
    'stoi': stoi,
    'itos': itos
}, 'models/BigramLM/bigram_model.pth')
print("MODEL SAVED.")