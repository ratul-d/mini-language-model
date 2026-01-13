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
n_embd = 32

torch.manual_seed(1337) # For reproducibility across runs

# SELF-ATTENTION
class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out

# MULTI-HEAD ATTENTION
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

# FEED FORWARD
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

# TRANSFORMER BLOCK
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x


# LANGUAGE MODEL
class BigramLangaugeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd) #(65,32)
        self.position_embedding_table = nn.Embedding(block_size,n_embd) #(8,32)
        self.blocks = nn.Sequential(
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),
        )
        self.lm_head = nn.Linear(n_embd,vocab_size) #(32,65)

    def forward(self, idx, targets=None):
        """
        What is happening in this forward pass:

        Assume Input (idx): (32, 8)
            32 batches, each with 8 token indices

        Token Embedding:
            idx -> token_embedding_table -> (32, 8, 32)
            Each token index is mapped to a 32-dimensional embedding
            Shape: (batch_size, sequence_length, embedding_dim)

        Position Embedding:
            torch.arange(T) -> position_embedding_table -> (8, 32)
            Each position (0-7) is mapped to a 32-dimensional embedding
            Shape: (sequence_length, embedding_dim)

        Combined Embeddings:
            x = tok_emb + pos_emb -> (32, 8, 32)
            Position embeddings are broadcasted and added to token embeddings
            Each token now has both "what it is" and "where it is" information

        Transformer Block * 4:
            Self-attention Head * 4:

            Feed Forward:


        Language Model Head:
            (32, 8, 32) -> lm_head -> (32, 8, 65)
            Batched matrix multiplication: 32 times of (8, 32) @ (32, 65) = (8, 65)
            Projects each embedding to vocabulary size (65 logits per token)
            Shape: (batch_size, sequence_length, vocab_size)

        Summary:
            (32, 8) --token_emb--> (32, 8, 32)
                    --pos_emb----> (8, 32)
                    --token_emb+pos_emb--------> (32, 8, 32)
                    --sa_head----> (32, 8, 32)
                    --linear-----> (32, 8, 65)
        """

        B,T = idx.shape

        tok_embd = self.token_embedding_table(idx) # output.shape --> (B,T,n_embd)
        pos_embd = self.position_embedding_table(torch.arange(T)) # output.shape --> (T,n_embd)
        x = tok_embd + pos_embd # output.shape --> (B,T,n_embd)
        x = self.blocks(x) # output.shape --> (B,T,n_embd)
        logits = self.lm_head(x) # output.shape --> (B,T,vocab_size)

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
            idx_cond = idx[:,-block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next), dim=1)
        return idx


# MODEL
m = BigramLangaugeModel()
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

for steps in range(100):
    xb, yb  = get_batch('train')            # data batch

    logits, loss = m(xb,yb)                 # forward pass & loss calculation
    optimizer.zero_grad(set_to_none=True)   # zero_grad
    loss.backward()                         # backward pass
    optimizer.step()                        # optimize

    if steps % 1 ==0:
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
    'n_embd': n_embd,
    'stoi': stoi,
    'itos': itos
}, 'models/Self-Attention/self-attention.pth')
print("MODEL SAVED.")