from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class AttentionHead(nn.Module):
    def __init__(self, input_size: int, head_size: int, dropout: float) -> None:
        super().__init__()
        self.head_size = head_size
        self.keys = nn.Linear(input_size, head_size, bias=False)
        self.queries = nn.Linear(input_size, head_size, bias=False)
        self.values = nn.Linear(input_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.keys(x)
        queries = self.queries(x)
        values = self.values(x)
        attn_scores = queries @ rearrange(keys, 'b t h -> b h t') / (self.head_size**0.5)
        attn_scores = torch.tril(attn_scores)
        attn_scores = attn_scores.masked_fill(attn_scores == 0, float('-inf'))
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        out: torch.Tensor = attn_scores @ values
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, input_size: int, head_size: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([AttentionHead(input_size, head_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, input_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 4 * input_size),
            nn.ReLU(),
            nn.Linear(4 * input_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, input_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        head_size = input_size // num_heads
        if head_size * num_heads != input_size:
            raise ValueError(f"input_size ({input_size}) must be divisible by num_heads ({num_heads})")
        self.self_attention = MultiHeadAttention(num_heads, input_size, head_size, dropout=dropout)
        self.ffn = FeedForward(input_size, dropout=dropout)
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    emb_size: int
    num_layers: int
    num_heads: int
    dropout: float


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.emb_size)
        self.position_embedding_table = nn.Embedding(config.block_size, config.emb_size)
        self.transformer = nn.Sequential(
            *[
                TransformerBlock(config.emb_size, config.num_heads, dropout=config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.ln = nn.LayerNorm(config.emb_size)
        self.lm_head = nn.Linear(config.emb_size, config.vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(tok_emb.shape[1], device=tok_emb.device))
        x = tok_emb + pos_emb
        x = self.transformer(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(rearrange(logits, 'b t c -> (b t) c'), rearrange(targets, 'b t -> (b t)'))
        return logits, loss

    def generate(self, start_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self.forward(start_tokens[:, -self.block_size :])
            logits = logits[:, -1, :]
            predicted_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            start_tokens = torch.cat((start_tokens, predicted_token), dim=1)
        return start_tokens
