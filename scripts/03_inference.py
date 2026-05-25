import argparse
import json
from pathlib import Path

import torch
import yaml
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2LMHeadModel


def read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tokenizer(tokenizer_dir: Path):
    vocab = str(tokenizer_dir / "vocab.json")
    merges = str(tokenizer_dir / "merges.txt")
    return ByteLevelBPETokenizer(vocab, merges)


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 80, temperature: float = 0.8):
    device = next(model.parameters()).device

    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", default="Merhaba")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    cfg = read_config(args.config)

    output_dir = Path(cfg["project"]["output_dir"])
    tokenizer_dir = output_dir / "tokenizer"
    model_dir = output_dir / "final_model"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_dir)

    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(str(model_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Device: {device}")
    print(f"Prompt: {args.prompt}")

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("\n--- Generated Text ---")
    print(text)


if __name__ == "__main__":
    main()
