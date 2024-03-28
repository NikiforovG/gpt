import logging
from dataclasses import dataclass
from time import time

import torch

from src.data import Data, Vocabulary
from src.gpt import GPTConfig, GPTModel
from src.utils import count_parameters, estimate_loss, get_tinyshakespeare_dataset

# Create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    max_iters: int
    eval_interval: int
    eval_iters: int


def process_training(data: Data, model: GPTModel, optimizer: torch.optim.Optimizer, config: TrainingConfig) -> None:
    timer = time()
    for steps in range(config.max_iters):
        xb, yb = data.get_batch('train', block_size=model.block_size, batch_size=config.batch_size)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        if steps % eval_interval == 0:
            losses = estimate_loss(
                eval_iters=config.eval_iters,
                model=model,
                data=data,
                block_size=model.block_size,
                batch_size=config.batch_size,
            )
            logger.info(
                "step: %d; training time: %d sec; train loss: %.4f; val loss: %.4f",
                steps + 1,
                round(time() - timer),
                losses['train'],
                losses['val'],
            )


if __name__ == '__main__':
    torch.manual_seed(42)

    text = get_tinyshakespeare_dataset()

    vocab = Vocabulary(text=text)
    data = Data(vocab.encode(text))

    block_size = 8
    emb_size = 32
    num_heads = 4
    num_layers = 3
    dropout = 0.2

    model_config = GPTConfig(
        vocab_size=vocab.size,
        block_size=block_size,
        emb_size=emb_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = GPTModel(config=model_config)
    logger.info('models has %d parameters', count_parameters(model))

    learning_rate = 1e-3
    batch_size = 32
    max_iters = 10001
    eval_interval = 1000
    eval_iters = 200

    training_config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_iters=max_iters,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    process_training(data, model, optimizer, training_config)
    logger.info(
        vocab.decode(model.generate(start_tokens=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist())
    )
