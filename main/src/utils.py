import glob
import os
from dataclasses import dataclass
from typing import Any

import requests
import torch

from .data import Data
from .gpt import GPTConfig, GPTModel


def get_tinyshakespeare_dataset() -> str:
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


@torch.no_grad()
def estimate_loss(
    eval_iters: int, model: torch.nn.Module, data: Data, block_size: int, batch_size: int, device: torch.device
) -> dict[str, torch.Tensor]:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = data.get_batch(split, block_size=block_size, batch_size=batch_size)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class TrainingState:
    training_steps: int
    training_time: int
    model_config: GPTConfig
    model: GPTModel
    optimizer_state_dict: dict[str, Any]


def save_training_state(folder: str, training_state: TrainingState) -> None:
    torch.save(
        {
            'model_config': training_state.model_config,
            'model_state_dict': training_state.model.state_dict(),
            'optimizer_state_dict': training_state.optimizer_state_dict,
            'training_steps': training_state.training_steps,
            'training_time': training_state.training_time,
        },
        os.path.join(folder, f'gpt_{training_state.training_steps}.pth'),
    )


def load_training_state(folder: str) -> TrainingState:
    if not os.path.isdir(folder):
        raise ValueError(f"Folder {folder} does not exist.")

    checkpoint_files = glob.glob(os.path.join(folder, 'gpt_*.pth'))
    if not checkpoint_files:
        raise OSError("No 'gpt_*.pth' files found.")

    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    checkpoint = torch.load(latest_checkpoint)

    model = GPTModel(checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])

    return TrainingState(
        training_steps=checkpoint['training_steps'],
        training_time=checkpoint['training_time'],
        model_config=checkpoint['model_config'],
        model=model,
        optimizer_state_dict=checkpoint['optimizer_state_dict'],
    )
