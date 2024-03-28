import requests
import torch

from .data import Data


def get_tinyshakespeare_dataset() -> str:
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


@torch.no_grad()
def estimate_loss(
    eval_iters: int, model: torch.nn.Module, data: Data, block_size: int, batch_size: int
) -> dict[str, torch.Tensor]:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = data.get_batch(split, block_size=block_size, batch_size=batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
