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
    "n_digits_test": 80,
    "max_iters": 50000,
    "lr_decay_iters": 50000,
    "eval_interval": 100,
    "eval_stride": 5,
}

if __name__ == "__main__":
    for size, size_name in [(SMALL, "small"), (MEDIUM, "medium"), (LARGE, "large")]:
        for decoder in [True, False]:
            for n_digits_train in [10, 20]:
                name = f"{size_name}_{'decoder' if decoder else 'encoder'}_{n_digits_train}"
                config = copy.deepcopy(BASE_CONFIG | size)
                config["name"] = name
                config["decoder"] = decoder
                config["n_digits_train"] = n_digits_train

                config_path = f"configs/{name}.json"
                with open(config_path, "w") as f:
                    json.dump(config, f)
