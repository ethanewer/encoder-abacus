import random

import torch
from torch import Tensor, nn

from .base import TransformerConfig


class Abacus(nn.Module):
    """
    Abacus Embeddings, learned emebddings resued for each digit.
    Integers must be reversed for this to work correctly.
    Transformers Can Do Arithmetic with the Right Embeddings, McLeish et al. (2024)
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(2 * config.n_positions, config.n_embd)
        self.register_buffer("digits", torch.tensor(config.digit_ids), persistent=False)
        self.max_k = config.n_positions

    def __helper(self, mask: Tensor) -> Tensor:
        mask_shape = mask.shape

        # Create a shifted version of the mask to detect changes from 0 to 1
        shifted_mask = torch.cat(
            [
                torch.zeros((mask_shape[0], 1), device=mask.device, dtype=mask.dtype),
                mask[:, :-1],
            ],
            dim=1,
        )
        starts = (shifted_mask != mask) & mask

        # Generate IDs for each segment of 1s, processing row-wise
        segment_ids = torch.cumsum(starts, dim=1)

        # Generate an index array row-wise
        index = torch.arange(mask.size(1)).repeat(mask.size(0), 1).to(mask.device)

        # Reset index at the start of each segment
        reset_index = torch.zeros_like(mask).long()
        second_term = index * starts.long()
        reset_index = reset_index.scatter_add(1, segment_ids, second_term)

        # Calculate positions in segment
        positions = index - reset_index.gather(1, segment_ids) + 1

        # Ensure only values within 1-segments are non-zero
        result = positions * mask

        return result

    def forward(self, input_ids: Tensor) -> Tensor:
        mask = torch.isin(input_ids, self.digits)
        output = self.__helper(mask)

        if self.training:
            output[output > 0] += random.randint(0, self.max_k)

        return self.embedding(output)
