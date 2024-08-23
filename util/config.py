import json


class Config:
    """Configuration for data generation, training, and evaluation."""

    batch_size: int = 64
    beta1: float = 0.9
    beta2: float = 0.99
    block_size: int = 64
    decoder: bool = True
    dropout: float = 0.0
    eval_interval: int = 100
    eval_stride: int = 1
    lr_decay_iters: int = 5000
    max_evals_without_improving: int = 1000
    max_iters: int = 5000
    max_loss_for_early_stopping: float = 1e9
    max_lr: float = 5e-4
    min_lr: float = 5e-5
    model_dir: str = "models"
    n_digits_train: int = 20
    n_digits_test: int = 100
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    name: str = ""
    results_dir: str = "results"
    resume: bool = False
    seed: int = 42
    test_batch_size: int = 1024
    use_wpe: bool = True
    warmup_iters: int = 100
    weight_decay: float = 0.1

    def __init__(self, config: dict[str, bool | int | float | str]) -> None:
        for k, v in config.items():
            setattr(self, k, v)

        assert self.name != ""

    def to_dict(self) -> dict[str, bool | int | float | str]:
        """Returns a `dict` with all configuration information."""
        d = {}
        for k in Config.__dict__.keys():
            if "__" not in k and k not in ("from_json", "to_dict"):
                d[k] = getattr(self, k)

        return d

    @property
    def checkpoint_name(self) -> str:
        return self.name + ".pt"

    @staticmethod
    def from_json(path: str) -> "Config":
        with open(path, "r") as f:
            config = json.load(f)

        return Config(config)
