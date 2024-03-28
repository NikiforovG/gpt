import torch


class Vocabulary:
    def __init__(self, text: str) -> None:
        chars = sorted(list(set(text)))
        self.size = len(chars)

        # create a mapping from characters to integers
        self.itos = dict(enumerate(chars))
        self.stoi = {ch: i for i, ch in self.itos.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]  # encoder: take a string, output a list of integers

    def decode(self, idx: list[int]) -> str:
        return ''.join([self.itos[i] for i in idx])  # decoder: take a list of integers, output a string


class Data:
    def __init__(self, data: list[int], train_test_split: float = 0.8) -> None:
        data_tensor = torch.tensor(data, dtype=torch.long)

        n = int(train_test_split * len(data))
        self.train_data = data_tensor[:n]
        self.val_data = data_tensor[n:]

    def get_batch(self, split: str, block_size: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y
