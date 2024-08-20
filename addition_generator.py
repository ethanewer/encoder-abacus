import random
from typing import Optional

import torch
from torch import Tensor


class AdditionGenerator:
    def __init__(
        self,
        n_digits_max: int,
        batch_size: int,
    ) -> None:
        self.n_digits_max = n_digits_max
        self.batch_size = batch_size
        self.char2int = {c: i for i, c in enumerate(sorted("0123456789+=\n"))}
        self.int2char = {i: c for c, i in self.char2int.items()}

    def generate_number(self, n_digits: int) -> int:
        return random.randint(10 ** (n_digits - 1), 10**n_digits - 1)

    def generate_example(self, n_digits: tuple[int, int]) -> list[int]:
        x = self.generate_number(n_digits[0])
        y = self.generate_number(n_digits[1])

        if random.choice((True, False)):
            x, y = y, x

        z = x + y

        x_rev = str(x)[::-1]
        y_rev = str(y)[::-1]
        z_rev = str(z)[::-1]

        if len(z_rev) == max(n_digits):
            z_rev += "\n"

        s = f"{x_rev}+{y_rev}={z_rev}\n"

        assert len(s) == sum(n_digits) + max(n_digits) + 4, (
            len(s),
            sum(n_digits) + max(n_digits) + 4,
            s,
            n_digits,
        )

        return [self.char2int[c] for c in s]

    def generate_batch(
        self,
        n_digits: Optional[tuple[int, int]] = None,
    ) -> tuple[Tensor, Tensor, list[int]]:
        if n_digits is None:
            n_digits = (
                random.randint(1, self.n_digits_max),
                random.randint(1, self.n_digits_max),
            )

        z_start = sum(n_digits) + 2
        forward_idxs = list(range(z_start - 1, z_start + max(n_digits) + 1))

        data = torch.tensor(
            [self.generate_example(n_digits) for _ in range(self.batch_size)]
        )

        x = data[:, :-1]
        y = data[:, 1:]

        return x, y, forward_idxs
