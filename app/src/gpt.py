import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class AttentionHead(nn.Module):
    def __init__(self, input_size: int, head_size: int) -> None:
        super().__init__()
        self.head_size = head_size
        self.keys = nn.Linear(input_size, head_size, bias=False)
        self.queries = nn.Linear(input_size, head_size, bias=False)
        self.values = nn.Linear(input_size, head_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.keys(x)
        queries = self.queries(x)
        values = self.values(x)
        attn_scores = queries @ rearrange(keys, 'b t h -> b h t') / (self.head_size**0.5)
        attn_scores = torch.tril(attn_scores)
        attn_scores = attn_scores.masked_fill(attn_scores == 0, float('-inf'))
        attn_scores = F.softmax(attn_scores, dim=1)
        out: torch.Tensor = attn_scores @ values
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, input_size: int, head_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([AttentionHead(input_size, head_size) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.attention_heads], dim=-1)


class GPTModel(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, emb_size: int, num_heads: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, emb_size)
        self.position_embedding_table = nn.Embedding(block_size, emb_size)
        self.sa_heads = MultiHeadAttention(num_heads=num_heads, input_size=emb_size, head_size=emb_size // num_heads)
        self.lm_head = nn.Linear(num_heads * emb_size // num_heads, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(tok_emb.shape[1]))
        x = tok_emb + pos_emb
        x = self.sa_heads(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(rearrange(logits, 'b t c -> (b t) c'), rearrange(targets, 'b t -> (b t)'))
        return logits, loss

    def generate(self, start_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        tokens = start_tokens
        for _ in range(max_new_tokens):
            logits, _ = self.forward(tokens[:, -self.block_size :])
            logits = logits[:, -1, :]
            predicted_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            tokens = torch.cat((tokens, predicted_token), dim=1)
        return tokens
