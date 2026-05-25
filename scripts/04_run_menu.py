import subprocess
import sys
import json
from pathlib import Path

CONFIG = "configs/tiny_local.yaml"


def run(cmd):
    print("\nRunning:")
    print(" ".join(cmd))
    print("-" * 60)
    subprocess.run(cmd)


def select_config():
    global CONFIG

    configs_dir = Path("configs")
    configs = sorted(configs_dir.glob("*.yaml"))

    if not configs:
        print("No config files found in configs/")
        return

    print("\nAvailable configs:")
    for i, path in enumerate(configs, start=1):
        marker = "current" if str(path) == CONFIG else ""
        print(f"{i}. {path} {marker}")

    raw = input(f"\nSelect config [{CONFIG}]: ").strip()

    if not raw:
        return

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(configs):
            CONFIG = str(configs[idx])
            print(f"Active config: {CONFIG}")
            return

    print("Invalid selection. Config unchanged.")


def create_config():
    run([sys.executable, "scripts/00_create_config.py"])
    select_config()


def check_env():
    code = r"""
import sys

print("Environment Check")
print("=" * 32)

print(f"Python: OK ({sys.version.split()[0]})")

try:
    import torch
    print(f"Torch: OK ({torch.__version__})")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA: {'OK' if cuda_available else 'Not available'}")

    if cuda_available:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print("Recommendation: GPU training available.")
    else:
        print("Device: CPU only")
        print("Recommendation: Training will work, but it may be slow.")

except Exception as e:
    print("Torch: FAIL")
    print(f"Error: {e}")
    print("Recommendation: Install requirements or check your Python environment.")
"""
    run([sys.executable, "-c", code])


def prepare():
    run([sys.executable, "scripts/01_prepare.py", "--config", CONFIG])


def train():
    run([sys.executable, "scripts/02_train.py", "--config", CONFIG])


def inference():
    prompt = input("Prompt: ").strip()
    if not prompt:
        prompt = "Merhaba"

    run([
        sys.executable,
        "scripts/03_inference.py",
        "--config", CONFIG,
        "--prompt", prompt,
        "--max-new-tokens", "80"
    ])


def show_config():
    path = Path(CONFIG)
    print(f"\n--- Active Config: {CONFIG} ---")
    print(path.read_text(encoding="utf-8"))

def run_full_workflow():
    print("\nFull workflow will run:")
    print("1. Prepare dataset")
    print("2. Train model")
    print("3. Run inference")
    confirm = input("\nContinue? [y/N]: ").strip().lower()

    if confirm != "y":
        print("Cancelled.")
        return

    prepare()
    train()
    inference()

def show_project_status():
    import yaml

    cfg_path = Path(CONFIG)
    if not cfg_path.exists():
        print(f"Config not found: {CONFIG}")
        return

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    output_dir = Path(cfg["project"]["output_dir"])

    print("\nProject Status")
    print("=" * 32)
    print(f"Config: {CONFIG}")
    print(f"Project: {cfg['project']['name']}")
    print(f"Output dir: {output_dir}")

    prepare_summary = output_dir / "prepare_summary.json"
    train_summary = output_dir / "train_summary.json"
    final_model = output_dir / "final_model" / "model.safetensors"

    print(f"Prepared dataset: {'OK' if prepare_summary.exists() else 'Missing'}")
    print(f"Trained model: {'OK' if final_model.exists() else 'Missing'}")

    if prepare_summary.exists():
        data = json.loads(prepare_summary.read_text(encoding="utf-8"))
        print("\nDataset")
        print(f"- Total tokens: {data.get('total_tokens')}")
        print(f"- Block size: {data.get('block_size')}")
        print(f"- Train chunks: {data.get('train_chunks')}")
        print(f"- Val chunks: {data.get('val_chunks')}")

    if train_summary.exists():
        data = json.loads(train_summary.read_text(encoding="utf-8"))
        print("\nModel")
        print(f"- Parameters: {data.get('total_params'):,}")
        print(f"- Layers: {data.get('n_layer')}")
        print(f"- Heads: {data.get('n_head')}")
        print(f"- Embedding size: {data.get('n_embd')}")
        print(f"- Max steps: {data.get('max_steps')}")


def main():
    while True:
        print("\nSimple Local GPT Trainer")
        print("=" * 32)
        print(f"Active config: {CONFIG}")
        print()
        print("1. Select config")
        print("2. Create config")
        print("3. Check environment")
        print("4. Prepare dataset")
        print("5. Train model")
        print("6. Run inference")
        print("7. Show config")
        print("8. Show project status")
        print("9. Run full workflow")
        print("10. Exit")

        choice = input("\nSelect: ").strip()

        if choice == "1":
            select_config()
        elif choice == "2":
            create_config()
        elif choice == "3":
            check_env()
        elif choice == "4":
            prepare()
        elif choice == "5":
            train()
        elif choice == "6":
            inference()
        elif choice == "7":
            show_config()
        elif choice == "8":
            show_project_status()
        elif choice == "9":
            run_full_workflow()
        elif choice == "10":
            print("Goodbye.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
