import os
import sys

import torch
from tqdm import trange  # type: ignore

from addition_generator import AdditionGenerator
from nano_abacus_transformer import AbacusTransformerLMHead, TransformerConfig
from util import Config, Environment


def evaluate(config: Config, env: Environment):
    env.seed_everything(config.seed)

    data_generator = AdditionGenerator(config.n_digits_test, 1)

    model_config = TransformerConfig(
        digit_ids=[data_generator.char2int[str(i)] for i in range(10)],
        n_positions=config.block_size,
        vocab_size=len(data_generator.char2int),
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
    )

    model = AbacusTransformerLMHead(model_config, env.compile_blocks).to(env.device)

    model_path = os.path.join(config.model_dir, config.checkpoint_name)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    i = 3
    j = 3

    x, y, forward_idxs = data_generator.generate_batch((i + 1, j + 1))

    x = x[:, : i + j + 4].to(env.device)
    forward_idxs = forward_idxs[:-1]
    assert len(forward_idxs) == max(i + 1, j + 1) + 1

    with env.context:
        y_hat = model.generate(
            x,
            max_new_tokens=len(forward_idxs),
            decoder=config.decoder,
        )[:, 1:]

    y_hat = y_hat.cpu()

    y_hat = y_hat[:, forward_idxs]
    y = y[:, forward_idxs]

    print(y_hat)
    print(y)





if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python eval-addition.py <config-path>")
        exit(1)

    config = Config.from_json(sys.argv[1])
    env = Environment()
    evaluate(config, env)
