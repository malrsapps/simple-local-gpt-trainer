import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)


class JsonlTokenDataset(Dataset):
    def __init__(self, path: Path):
        self.samples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                ids = item["input_ids"]
                self.samples.append(torch.tensor(ids, dtype=torch.long))

        if not self.samples:
            raise ValueError(f"No samples found in {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids = self.samples[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_vocab(vocab_path: Path) -> int:
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    return len(vocab)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_config(args.config)

    output_dir = Path(cfg["project"]["output_dir"])
    prepared_dir = output_dir / "prepared"
    tokenizer_dir = output_dir / "tokenizer"
    model_dir = output_dir / "model"

    train_path = prepared_dir / "train.jsonl"
    val_path = prepared_dir / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing val file: {val_path}")

    vocab_size = count_vocab(tokenizer_dir / "vocab.json")

    block_size = int(cfg["dataset"]["block_size"])

    n_layer = int(cfg["model"]["n_layer"])
    n_head = int(cfg["model"]["n_head"])
    n_embd = int(cfg["model"]["n_embd"])
    dropout = float(cfg["model"]["dropout"])

    batch_size = int(cfg["training"]["batch_size"])
    grad_accum = int(cfg["training"]["gradient_accumulation_steps"])
    learning_rate = float(cfg["training"]["learning_rate"])
    max_steps = int(cfg["training"]["max_steps"])
    warmup_steps = int(cfg["training"]["warmup_steps"])

    print("Loading datasets...")
    train_dataset = JsonlTokenDataset(train_path)
    val_dataset = JsonlTokenDataset(val_path)

    print("Building GPT-style model from config...")
    model_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        bos_token_id=0,
        eos_token_id=2,
    )

    model = GPT2LMHeadModel(model_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Vocab size: {vocab_size}")
    print(f"Block size: {block_size}")
    print(f"Model params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    training_args = TrainingArguments(
        output_dir=str(model_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting training...")
    trainer.train()

    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving final model to: {final_dir}")
    trainer.save_model(str(final_dir))

    train_summary = {
        "project": cfg["project"]["name"],
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "dropout": dropout,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "max_steps": max_steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
    }

    summary_path = output_dir / "train_summary.json"
    summary_path.write_text(json.dumps(train_summary, indent=2), encoding="utf-8")

    print("Done.")
    print(json.dumps(train_summary, indent=2))


if __name__ == "__main__":
    main()
