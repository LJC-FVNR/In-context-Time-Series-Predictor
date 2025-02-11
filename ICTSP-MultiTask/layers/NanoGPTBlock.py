### From: https://github.com/karpathy/nanoGPT/blob/master/model.py

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        From: https://github.com/meta-llama/llama/blob/main/llama/model.py
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.RoPE = nn.Parameter(rotary_embedding(8192, config.n_embd), requires_grad=False)
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        x = apply_rotary_pos_emb(x, self.RoPE, reverse=True)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

def rotary_embedding(max_len, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    position = torch.arange(0, max_len).unsqueeze(1)
    sinusoid_inp = torch.ger(position.squeeze(), inv_freq).float()
    return torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

def apply_rotary_pos_emb(x, sincos, reverse=False):
    B, seq_len, dim = x.shape
    if reverse:
        sincos = sincos[0:seq_len].flip(0)
    else:
        sincos = sincos[0:seq_len]
    sin, cos = sincos[:, :dim//2].repeat_interleave(2, dim=-1), sincos[:, dim//2:].repeat_interleave(2, dim=-1)
    return (x * cos) + (torch.roll(x, shifts=1, dims=-1) * sin)
    
# class AttentionPlacer(nn.Module):
#     def __init__(self, D, H):
#         super(AttentionPlacer, self).__init__()
#         self.D = D
#         self.H = H
#         self.d = D // H
#         #self.evaluator = nn.Linear(self.d, 1)
#         self.evaluator = nn.Sequential(
#             nn.Linear(self.d, 4 * self.d, bias=False),
#             nn.Dropout(0.1),
#             nn.GELU(),
#             nn.Linear(4 * self.d, 1, bias=False),
#         )
#         self.activation = nn.Softmax(dim=-1)
#         self.eps = 1e-6
#         self.RoPE = nn.Parameter(rotary_embedding(8192, D), requires_grad=False)
    
#     def forward(self, X):
#         B, L, D = X.shape
        
#         # Reshape to (B, L, H, d)
#         X = X.reshape(B, L, self.H, self.d)
        
#         # Permute to (B, H, L, d)
#         X = X.permute(0, 2, 1, 3)
        
#         # Apply linear layer to (B, H, L, d) and get (B, H, L, 1)
#         X_score = self.evaluator(X)  # (B, H, L, 1)
#         X_score = self.activation(X_score.squeeze(-1)) # (B, H, L, 1)
        
#         X_prob_max, X_indices = torch.sort(X_score, dim=-1)
#         X_prob_max, X_indices = X_prob_max.unsqueeze(-1).expand(-1, -1, -1, self.d), X_indices.unsqueeze(-1).expand(-1, -1, -1, self.d)
        
#         X_sorted = torch.gather(X, 2, X_indices)  # (B, H, L, d)
        
#         X_prob_max = X_prob_max + self.eps
#         X_sorted = X_sorted * (X_prob_max / X_prob_max.detach())  # (B, H, L, d)
        
#         X_sorted = X_sorted.permute(0, 2, 1, 3).reshape(B, L, self.H*self.d)   # (B, L, D)
#         X_sorted = apply_rotary_pos_emb(X_sorted, self.RoPE, reverse=True)
        
#         return X_sorted

# class Block(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         self.ln_1 = torch.compile(RMSNorm(config.n_embd))
#         self.attn1 = AttentionPlacer(config.n_embd, config.n_head)
#         #self.attn2 = torch.compile(CausalSelfAttention(config))
#         self.ln_2 = torch.compile(RMSNorm(config.n_embd))
#         self.ln_3 = torch.compile(RMSNorm(config.n_embd))
#         self.mlp = torch.compile(MLP(config))

#     def forward(self, x, src_mask=None):
#         x = self.attn1(self.ln_1(x))
#         #x = x + self.attn2(self.ln_2(x), attn_mask=src_mask)
#         x = x + self.mlp(self.ln_3(x))
#         return x

class AttentionPlacer(nn.Module):
    def __init__(self, D, H):
        super(AttentionPlacer, self).__init__()
        self.D = D
        self.H = H
        self.d = D // H
        #self.evaluator = nn.Linear(self.d, 1)
        self.evaluator = nn.Sequential(
            nn.Linear(self.d, 4 * self.d, bias=False),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(4 * self.d, self.d+1, bias=False),
        )
        self.activation = nn.Softmax(dim=-1)
        self.eps = 1e-6
        self.RoPE = nn.Parameter(rotary_embedding(8192, D), requires_grad=False)
    
    def forward(self, X):
        B, L, D = X.shape
        
        X = apply_rotary_pos_emb(X, self.RoPE, reverse=True)
        
        # Reshape to (B, L, H, d)
        X = X.reshape(B, L, self.H, self.d)
        
        # Permute to (B, H, L, d)
        X = X.permute(0, 2, 1, 3)
        
        # Apply linear layer to (B, H, L, d) and get (B, H, L, 1)
        X_forward = self.evaluator(X)  # (B, H, L, 1)
        X = X_forward[:, :, :, 0:-1]
        X_score = X_forward[:, :, :, [-1]]
        X_score = self.activation(X_score.squeeze(-1)) # (B, H, L, 1)
        
        X_prob_max, X_indices = torch.sort(X_score, dim=-1)
        X_prob_max, X_indices = X_prob_max.unsqueeze(-1).expand(-1, -1, -1, self.d), X_indices.unsqueeze(-1).expand(-1, -1, -1, self.d)
        
        X_sorted = torch.gather(X, 2, X_indices)  # (B, H, L, d)
        
        X_prob_max = X_prob_max + self.eps
        X_sorted = X_sorted * (X_prob_max / X_prob_max.detach())  # (B, H, L, d)
        
        X_sorted = X_sorted.permute(0, 2, 1, 3).reshape(B, L, self.H*self.d)   # (B, L, D)
        X_sorted = apply_rotary_pos_emb(X_sorted, self.RoPE, reverse=False)
        
        return X_sorted

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        #self.attn1 = AttentionPlacer(config.n_embd, config.n_head)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.ln_3 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, src_mask=None):
        # x = x + self.attn1(self.ln_1(x))
        x = x + self.attn(self.ln_2(x), attn_mask=src_mask)
        x = x + self.mlp(self.ln_3(x))
        return x