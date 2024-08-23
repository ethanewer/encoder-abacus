import os
import sys
from pprint import pprint

import numpy as np
import torch
import wandb

from addition_generator import AdditionGenerator
from nano_abacus_transformer import (
    AbacusTransformerLMHead,
    TransformerConfig,
    flat_cross_entropy,
)
from util import Config, Environment, LRSchedule


def train(config: Config, env: Environment) -> None:
    print(f"{env.device=}, env.context={str(type(env.context))[8 : -2]}", end=", ")
    print(f"{env.pin_memory=}, {env.pin_memory_device=}, {env.compile_blocks=}")
    pprint(config.to_dict())

    run = wandb.init(
        dir=config.results_dir,
        project="encoder-abacus",
        config=config.to_dict(),
        name=config.name,
        resume=config.resume,
    )

    env.seed_everything(config.seed)

    data_generator = AdditionGenerator(config.n_digits_train, config.batch_size)

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

    optimizer = model.configure_optimizers(
        lr=config.min_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        device=env.device,
    )

    lr_schedule = LRSchedule(config)
    i = 0
    best_loss = float("inf")
    losses = []
    n_evals_without_improving = 0

    if config.resume:
        load_path = os.path.join(config.model_dir, config.checkpoint_name)
        checkpoint = torch.load(load_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        i = checkpoint["i"]
        best_loss = checkpoint["best_loss"]

    while True:
        i += 1

        model.train()

        lr = lr_schedule(i)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y, forward_idxs = data_generator.generate_batch()

        x = x.to(env.device)
        y = y.to(env.device)

        with env.context:
            logits = model(x, decoder=config.decoder, forward_idxs=forward_idxs)
            loss = flat_cross_entropy(logits[:, forward_idxs], y[:, forward_idxs])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

        if i % config.eval_interval == 0:
            train_loss = float(np.mean(losses[-config.eval_interval :]))
            wandb.log({"train_loss": train_loss}, step=i)

            if train_loss < best_loss:
                n_evals_without_improving = 0
                print(f"saved checkpoint    {f'{i=}':8}  {train_loss=:.3f}")
                best_loss = train_loss
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "i": i,
                    "best_loss": best_loss,
                }
                save_path = os.path.join(config.model_dir, config.checkpoint_name)
                torch.save(checkpoint, save_path)
            else:
                n_evals_without_improving += 1

        if i >= config.max_iters or (
            n_evals_without_improving >= config.max_evals_without_improving
            and best_loss < config.max_loss_for_early_stopping
        ):
            run.finish()
            return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python train.py <config-path>")
        exit(1)

    config = Config.from_json(sys.argv[1])
    env = Environment()

    train(config, env)
