import argparse
import json
import math
from pathlib import Path

import yaml
from tokenizers import ByteLevelBPETokenizer


def read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input text not found: {path}")
    text = path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text


def train_tokenizer(text_path: Path, tokenizer_dir: Path, vocab_size: int, min_frequency: int):
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(text_path)],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    tokenizer.save_model(str(tokenizer_dir))

    # Save simple metadata
    (tokenizer_dir / "tokenizer_meta.json").write_text(
        json.dumps(
            {
                "vocab_size": vocab_size,
                "min_frequency": min_frequency,
                "type": "ByteLevelBPE",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return tokenizer


def chunk_tokens(tokens: list[int], block_size: int) -> list[list[int]]:
    chunks = []
    for i in range(0, len(tokens) - block_size + 1, block_size):
        chunks.append(tokens[i : i + block_size])
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_config(args.config)

    input_text = Path(cfg["project"]["input_text"])
    output_dir = Path(cfg["project"]["output_dir"])

    tokenizer_dir = output_dir / "tokenizer"
    prepared_dir = output_dir / "prepared"

    vocab_size = int(cfg["tokenizer"]["vocab_size"])
    min_frequency = int(cfg["tokenizer"]["min_frequency"])
    block_size = int(cfg["dataset"]["block_size"])
    validation_ratio = float(cfg["dataset"]["validation_ratio"])

    prepared_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Reading corpus...")
    text = read_text(input_text)

    cleaned_path = prepared_dir / "cleaned_corpus.txt"
    cleaned_path.write_text(text, encoding="utf-8")

    print("[2/5] Training tokenizer...")
    tokenizer = train_tokenizer(cleaned_path, tokenizer_dir, vocab_size, min_frequency)

    print("[3/5] Encoding text...")
    encoded = tokenizer.encode(text)
    token_ids = encoded.ids

    total_tokens = len(token_ids)
    print(f"Total tokens: {total_tokens}")

    if total_tokens < block_size * 2:
        raise ValueError(
            f"Corpus too small. Need at least about {block_size * 2} tokens, got {total_tokens}."
        )

    print("[4/5] Creating fixed-length token chunks...")
    chunks = chunk_tokens(token_ids, block_size)

    split_index = max(1, int(len(chunks) * (1 - validation_ratio)))
    train_chunks = chunks[:split_index]
    val_chunks = chunks[split_index:]

    if not val_chunks:
        val_chunks = train_chunks[-1:]
        train_chunks = train_chunks[:-1]

    train_path = prepared_dir / "train.jsonl"
    val_path = prepared_dir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for chunk in train_chunks:
            f.write(json.dumps({"input_ids": chunk}, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for chunk in val_chunks:
            f.write(json.dumps({"input_ids": chunk}, ensure_ascii=False) + "\n")

    print("[5/5] Writing preparation summary...")
    summary = {
        "project": cfg["project"]["name"],
        "input_text": str(input_text),
        "output_dir": str(output_dir),
        "total_tokens": total_tokens,
        "block_size": block_size,
        "total_chunks": len(chunks),
        "train_chunks": len(train_chunks),
        "val_chunks": len(val_chunks),
        "vocab_size": vocab_size,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "tokenizer_dir": str(tokenizer_dir),
    }

    summary_path = output_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDone.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
