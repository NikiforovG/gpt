import logging
import os
from dataclasses import dataclass
from time import time

import torch

from src.data import Data, Vocabulary
from src.gpt import GPTConfig, GPTModel
from src.utils import (
    count_parameters,
    estimate_loss,
    get_tinyshakespeare_dataset,
    load_training_state,
    save_training_state,
    TrainingState,
)

# Create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    batch_size: int
    max_iters: int
    eval_interval: int
    eval_iters: int


def process_training(
    data: Data,
    training_state: TrainingState,
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> TrainingState:
    timer = time()
    model = training_state.model
    steps = 0
    for steps in range(training_state.training_steps + 1, training_state.training_steps + 1 + config.max_iters):
        xb, yb = data.get_batch(
            'train', block_size=training_state.model_config.block_size, batch_size=config.batch_size
        )
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        if steps % config.eval_interval == 0:
            losses = estimate_loss(
                eval_iters=config.eval_iters,
                model=model,
                data=data,
                block_size=training_state.model_config.block_size,
                batch_size=config.batch_size,
                device=device,
            )
            logger.info(
                "step: %d; training time: %d sec; train loss: %.4f; val loss: %.4f",
                steps,
                round(time() - timer),
                losses['train'],
                losses['val'],
            )

    training_time = round(time() - timer)
    logger.info("Totla training time %d sec", training_time)
    result_training_state = TrainingState(
        model_config=training_state.model_config,
        model=model,
        optimizer_state_dict=optimizer.state_dict(),
        training_time=training_time,
        training_steps=steps,
    )
    return result_training_state


def initialize_training_state(new: bool, folder: str | None = None) -> TrainingState:
    if new:
        model_config = GPTConfig(
            vocab_size=vocab.size,
            block_size=8,
            emb_size=32,
            num_heads=4,
            num_layers=3,
            dropout=0.2,
        )
        model = GPTModel(config=model_config).to(device)

        learning_rate = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        return TrainingState(
            model_config=model_config,
            model=model,
            optimizer_state_dict=optimizer.state_dict(),
            training_time=0,
            training_steps=0,
        )
    if folder is None:
        raise ValueError("folder must be specified when new is False")
    return load_training_state(folder=folder)


if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    logger.info('device: %s', device)

    new_training = False
    weights_folder = './weights'

    text = get_tinyshakespeare_dataset()

    vocab = Vocabulary(text=text)
    data = Data(vocab.encode(text))

    training_state = initialize_training_state(new=new_training, folder=weights_folder)
    optimizer = torch.optim.AdamW(training_state.model.parameters())
    optimizer.load_state_dict(training_state.optimizer_state_dict)

    logger.info('models has %d parameters', count_parameters(training_state.model))

    training_config = TrainingConfig(
        batch_size=32,
        max_iters=10000,
        eval_interval=1000,
        eval_iters=200,
    )
    training_state = process_training(
        data=data, training_state=training_state, optimizer=optimizer, config=training_config, device=device
    )
    save_training_state(weights_folder, training_state)

    sample_generation = vocab.decode(
        training_state.model.generate(start_tokens=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[
            0
        ].tolist()
    )
    with open(
        os.path.join(weights_folder, f'gpt_{training_state.training_steps}_sample_output.txt'), 'w', encoding='utf-8'
    ) as f:
        f.write(sample_generation)
