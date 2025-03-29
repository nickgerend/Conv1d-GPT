import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect

# data loading
def get_batch(train_data, val_data, split, block_size, batch_size, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, block_size, batch_size, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def configure_optimizer(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups, any parameters that is 2D will be weight decayed, otherwise no
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")
    return optimizer

# nanoGPT (Karpathy) style
class CausalAttentionHead(nn.Module):

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        w = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        w = F.softmax(w, dim=-1) # (B, T, T)
        w = self.dropout(w)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = w @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttentionHead(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# causal dilated convolutional layer
class CausalDilatedConv(nn.Module):
    def __init__(self, n_embd, kernel_size=3, dilation=2):
        super().__init__()
        self.conv = nn.Conv1d(n_embd, n_embd, kernel_size, padding=0, dilation=dilation)
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        pad = (self.kernel_size - 1) * self.dilation  # Left padding for causality
        x_padded = F.pad(x, (pad, 0), mode='constant', value=0)
        out = self.conv(x_padded)
        return out.transpose(1, 2)  # Back to (B, T, C)

# Conv1d + Attention
class Block(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout, conv=False, kernel_size=3, dilation=2):
        super().__init__()
        head_size = n_embd // n_head
        self.kernel_size = kernel_size
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.conv = conv
        if conv:
            self.dilated_conv = CausalDilatedConv(n_embd, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        if self.conv:                           # Conv1dGPT
            ln_x = self.ln1(x)
            attn_out = self.sa(ln_x)            # self-attention output
            conv_out = self.dilated_conv(ln_x)  # dilated convolution output
            x = x + attn_out + conv_out         # residual (skip) connection
            x = x + self.ffwd(self.ln2(x))      # feed-forward and residual
        else:                                   # else normal GPT
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        return x

# Conv1d-GPT
class Conv1dGPT(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_head, dropout, conv, kernel_size, n_layer, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout, conv, kernel_size, dilation=2**i) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    def generate_and_so(self, max_new_tokens):
        # starting tokens for "And so," based on character-level mapping:
        # "A" = 30, "n" = 73, "d" = 63, " " = 2, "s" = 78, "o" = 74, "," = 12
        idx = torch.tensor([[30, 73, 63, 2, 78, 74, 12]], dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions and loss (loss is not used here)
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # shape becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # shape: (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # shape: (B, T+1)
        return idx
