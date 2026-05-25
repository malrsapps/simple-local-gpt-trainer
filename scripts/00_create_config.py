from pathlib import Path

import yaml


MODEL_PRESETS = {
    "micro": {
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 128,
        "dropout": 0.1,
    },
    "tiny": {
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "dropout": 0.1,
    },
    "small": {
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "dropout": 0.1,
    },
}

TRAINING_PRESETS = {
    "quick": {
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "learning_rate": 0.0003,
        "max_steps": 200,
        "warmup_steps": 20,
    },
    "balanced": {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0003,
        "max_steps": 1000,
        "warmup_steps": 50,
    },
    "longer": {
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 0.0002,
        "max_steps": 3000,
        "warmup_steps": 100,
    },
}


def ask_choice(title: str, choices: list[str], default: str) -> str:
    print(f"\n{title}")
    for i, choice in enumerate(choices, start=1):
        marker = "default" if choice == default else ""
        print(f"{i}. {choice} {marker}")

    raw = input(f"Select [{default}]: ").strip()

    if not raw:
        return default

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx]

    if raw in choices:
        return raw

    print("Invalid choice, using default.")
    return default


def safe_name(name: str) -> str:
    cleaned = name.strip().lower()
    cleaned = cleaned.replace(" ", "_")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch in "_-")
    return cleaned or "local_model"


def main():
    print("Simple Local GPT Trainer - Config Generator")
    print("=" * 48)

    project_name = input("\nProject name [my_local_model]: ").strip() or "my_local_model"
    project_name = safe_name(project_name)

    input_text = input("Input text path [data/input/corpus.txt]: ").strip() or "data/input/corpus.txt"

    context_choice = ask_choice(
        "Context length / block size",
        ["128", "256", "512"],
        "128",
    )
    block_size = int(context_choice)

    model_size = ask_choice(
        "Model size",
        ["micro", "tiny", "small"],
        "tiny",
    )

    training_mode = ask_choice(
        "Training mode",
        ["quick", "balanced", "longer"],
        "balanced",
    )

    vocab_choice = ask_choice(
        "Tokenizer vocabulary size",
        ["2000", "4000", "8000", "16000"],
        "4000",
    )
    vocab_size = int(vocab_choice)

    output_dir = f"outputs/{project_name}"

    cfg = {
        "project": {
            "name": project_name,
            "input_text": input_text,
            "output_dir": output_dir,
        },
        "tokenizer": {
            "vocab_size": vocab_size,
            "min_frequency": 1,
        },
        "dataset": {
            "block_size": block_size,
            "validation_ratio": 0.1,
        },
        "model": MODEL_PRESETS[model_size],
        "training": TRAINING_PRESETS[training_mode],
    }

    configs_dir = Path("configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    config_path = configs_dir / f"{project_name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print("\nConfig created:")
    print(config_path)
    print("\nNext steps:")
    print(f"python scripts/01_prepare.py --config {config_path}")
    print(f"python scripts/02_train.py --config {config_path}")
    print(f"python scripts/03_inference.py --config {config_path} --prompt \"Merhaba\"")


if __name__ == "__main__":
    main()
