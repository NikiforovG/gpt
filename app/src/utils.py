import torch

from .data import Data


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
