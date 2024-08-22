import copy
import json

SMALL: dict[str, int] = {
    "n_layer": 3,
    "n_head": 3,
    "n_embd": 192,
}

MEDIUM: dict[str, int] = {
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
}

LARGE: dict[str, int] = {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
}

BASE_CONFIG: dict[str, bool | int | float | str] = {
    "block_size": 512,
    "n_digits_train": 20,
    "n_digits_test": 71,
    "max_iters": 25000,
    "lr_decay_iters": 25000,
    "decoder": True,
    "eval_interval": 100,
    "eval_stride": 10,
}

if __name__ == "__main__":
    config = copy.deepcopy(BASE_CONFIG | MEDIUM)
    config["name"] = "medium_20_digit"

    config_path = "configs/medium_20_digit.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
