import torch
import torch.nn as nn
from torch.nn import functional as F


# SELF-ATTENTION
class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5
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
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd) #(65,32)
        self.position_embedding_table = nn.Embedding(block_size,n_embd) #(8,32)
        self.blocks = nn.Sequential(
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd,vocab_size) #(32,65)

    def forward(self, idx, targets=None):
        """
        What is happening in this forward pass:

        Assume Input (idx): (batch_size, block_size) -> (32, 8)
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

        Transformer Block(x3):
            Each block has:

            #LayerNorm

            a) Multi-Head Self-Attention
                • Queries, Keys, Values computed from x
                • Causal mask prevents attending to future tokens
                • 4 attention heads run in parallel
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
            (32, 8, 32) -> lm_head -> (32, 8, 65)
            Projects each token embedding to vocabulary size (65 logits per token)
            Shape: (batch_size, sequence_length, vocab_size)

        Summary:
            (32, 8) --token_emb--> (32, 8, 32)
                    --pos_emb----> (8, 32)
                    --token_emb+pos_emb----> (32, 8, 32)
                    --Transformer Block(x3)----> (32, 8, 32)
                    --lm_head-----> (32, 8, 65)
        """

        B,T = idx.shape

        tok_embd = self.token_embedding_table(idx) # output.shape --> (B,T,n_embd)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device)) # output.shape --> (T,n_embd)
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


checkpoint = torch.load("models/Self-Attention/self-attention.pth")
vocab_size = checkpoint["vocab_size"]
block_size = checkpoint["block_size"]
dropout = checkpoint["dropout"]
n_embd = checkpoint["n_embd"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]


m = LanguageModel()
m.load_state_dict(checkpoint["model_state_dict"])
m.eval()


encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


# GENERATE
idx = torch.zeros((1,1), dtype=torch.long) # just a tensor having a single "0" to start the generation
print(decode(m.generate(idx,max_new_tokens=1000)[0].tolist()))