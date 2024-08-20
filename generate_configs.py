import copy
import json

BASE_CONFIG: dict[str, bool | int | float | str] = {
    "block_size": 512,
    "n_digits_train": 10,
    "n_digits_test": 25,
    "max_iters": 100000,
    "lr_decay_iters": 50000,
    "decoder": True,
    "eval_interval": 100,
    "name": "test_decoder",
}

if __name__ == "__main__":
    config = copy.deepcopy(BASE_CONFIG)

    config_path = "configs/test.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
