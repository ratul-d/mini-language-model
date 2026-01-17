import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizer.bpe_tokenizer import BPETokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# LOAD DATASET
with open("Dataset/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# LOAD tokenizer
tok = BPETokenizer.load("models/Transformer/BPE-Transformer/shakespeare_tokenizer.pkl")


# ENCODE COMPLETE DATASET
data = torch.tensor(tok.encode(text), dtype=torch.long, device=device)
print("Tokenized.")

# TRAIN-TEST SPLIT
n = int(0.9*len(data))
train_data = data[:n] # train_data.shape --> torch.Size([1003854])
val_data = data[n:] # val_data.shape --> torch.Size([111540])


# DATA BATCHES FUNCTION
def get_batch(split):
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,), device=device)
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x,y
"""
get_batch USAGE:
    xb,yb = get_batch("train")
    xb.shape --> [32,8] 32 separate occasions/batches of 8 continuous characters
    yb.shape --> [32,8] predicted next character for each character in previous matrix
"""


# HYPERPARAMETERS
vocab_size = tok.vocab_size
batch_size = 64
block_size = 256
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.2

torch.manual_seed(1337) # For reproducibility across runs

# SELF-ATTENTION
class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self,head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * self.head_size **-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

# MULTI-HEAD ATTENTION
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# FEED FORWARD
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# TRANSFORMER BLOCK
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connections
        x = x + self.ffwd(self.ln2(x)) # residual connections
        return x

# LANGUAGE MODEL
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self, idx, targets=None):
        """
        What is happening in this forward pass:

        Assume Input (idx): (batch_size, block_size) -> (64, 256)
            64 batches, each with 256 token indices

        Token Embedding:
            idx -> token_embedding_table -> (64, 256, 384)
            Each token index is mapped to a 32-dimensional embedding
            Shape: (batch_size, sequence_length, embedding_dim)

        Position Embedding:
            torch.arange(T) -> position_embedding_table -> (256, 384)
            Each position (0-255) is mapped to a 384-dimensional embedding
            Shape: (sequence_length, embedding_dim)

        Combined Embeddings:
            x = tok_emb + pos_emb -> (64, 256, 384)
            Position embeddings are broadcasted and added to token embeddings
            Each token now has both "what it is" and "where it is" information

        Transformer Block(x6):
            Each block has:

            #LayerNorm

            a) Multi-Head Self-Attention
                • Queries, Keys, Values computed from x
                • Causal mask prevents attending to future tokens
                • 6 attention heads run in parallel
                    Linear(n_embd → n_embd)
                    Dropout

            b) Residual connection
                x = x + self_attention_output

            #LayerNorm

            c) Feed-Forward Network
                Linear(n_embd → 4*n_embd)
                ReLU
                Linear(4*n_embd → n_embd)
                Dropout

            d) Residual connection
                x = x + feed_forward_output

        #LayerNorm

        Language Model Head:
            (64, 256, 384) -> lm_head -> (64, 256, 65)
            Projects each token embedding to vocabulary size (65 logits per token)
            Shape: (batch_size, sequence_length, vocab_size)

        Summary:
            (64, 256) --token_emb--> (64, 256, 384)
                    --pos_emb----> (256, 384)
                    --token_emb+pos_emb----> (64, 256, 384)
                    --Transformer Block(x6)----> (64, 256, 384)
                    --lm_head-----> (64, 256, 65)
        """

        B,T = idx.shape

        tok_embd = self.token_embedding_table(idx) # output.shape --> (B,T,n_embd)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device)) # output.shape --> (T,n_embd)
        x = tok_embd + pos_embd # output.shape --> (B,T,n_embd)
        x = self.blocks(x) # output.shape --> (B,T,n_embd)
        x = self.ln_f(x)
        logits = self.lm_head(x) # output.shape --> (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate_stream(self, idx, max_new_tokens, decode_fn):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            token = decode_fn(idx_next[0].tolist())
            print(token, end="", flush=True)
        return idx


# MODEL
m = LanguageModel().to(device)
"""
logits, loss = m(xb, yb)
print(logits.shape) --> torch.Size([256, vocab_size])
"""


# FUNCTION TO EVALUATE OVERALL TRAIN-VAL LOSS
@torch.no_grad()
def evaluate():
    m.eval()

    losses = {}
    for split in ['train', 'val']:
        split_losses = []
        for _ in range(50):  # only 50 batches
            xb, yb = get_batch(split)
            _, loss = m(xb, yb)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)

    m.train()
    return losses


# PARAMS
print("Model Parameters:",sum(p.numel() for p in m.parameters()))


# TRAINING
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(2001):
    xb, yb  = get_batch('train')            # data batch

    logits, loss = m(xb,yb)                 # forward pass & loss calculation
    optimizer.zero_grad(set_to_none=True)   # zero_grad
    loss.backward()                         # backward pass
    optimizer.step()                        # optimize

    if steps % 500 == 0:
        print(f"Step {steps}", end=' | ')
        print(f"Batch Loss: {loss.item():.4f}",end=' | ')

        losses = evaluate()
        print(f"Train: {losses['train']:.4f} | Val: {losses['val']:.4f}")

# SAVE THE MODEL
torch.save({
    'model_state_dict': m.state_dict(),
    'vocab_size': vocab_size,
    'block_size': block_size,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'dropout': dropout,
}, 'models/Transformer/BPE-Transformer/transformer.pth')
print("MODEL SAVED.")