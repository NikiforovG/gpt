import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.embeddings(idx)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(rearrange(logits, 'b t c -> (b t) c'), rearrange(targets, 'b t -> (b t)'))
        return logits, loss

    def generate(self, start_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        tokens = start_tokens
        for _ in range(max_new_tokens):
            logits, _ = self.forward(tokens)
            logits = logits[:, -1, :]
            predicted_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            tokens = torch.cat((tokens, predicted_token), dim=1)
        return tokens
