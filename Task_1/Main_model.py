# All the details , references , pitfalls & results are included in the Readme File of the GitHub repository so please go through it once
# Importing necessary Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import batch_size, block_size , max_iters , eval_interval , learning_rate , device , eval_iters , n_embd , n_head , n_layer , dropout


# Reading input text which is the tiny shakespeare dataset from the Huggingface datasets
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Creating character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)

# These are the Mapping each unique character in the text file with an integer which wil then be used for fetching the character form the integer or the integer from the character  
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoding and Decoding Lambda functions
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Training and Validation splits where I am using 90% of the data for training and the rest for validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# Instead of sending the data one by one we send them in batches that helps our efficiency as well as scalability
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Here we have explicitly stated no gradient to be caclculated as we dont want this to recalculated during backpropagation
@torch.no_grad()
# The loss used is a very simple inference used from Andrej Karpathy where we are deriving our losses using the negative log likelihood loss derived via the logits
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Model definition
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.attention = MultiHeadAttention(n_head, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

# Initializing weights for linear layers with a normal distribution (mean=0.0, std=0.02) and biases to zero if present
# For embedding layers, initializing weights with a normal distribution (mean=0.0, std=0.02).
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(idx.size(1), device=device))
        x = tok_emb + pos_emb
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# MultiHeadAttention class with multiple attention heads and linear projection
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
# Head class representing an attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

# FeedForward class with a simple neural network structure
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

model = GPTLanguageModel().to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e3} K parameters in the Main model")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Iterating as well as displaying training and validation loss for each 100th epoch
for iter in range(max_iters):
    if iter % 100 == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = decode(model.generate(context, max_new_tokens=500)[0].tolist())


print(output)

