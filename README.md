# Simple Local GPT Trainer

Train a tiny GPT-style model from your own text using a simple local workflow.

This project is a beginner-friendly local training kit. Put your text into `data/input/corpus.txt`, create or edit a config file, prepare the dataset, train a small GPT-style model, and run inference locally.

The goal is not to train a production-grade LLM. The goal is to make small-scale GPT-style experimentation understandable, editable, and runnable on a local machine.

---

## What This Project Does

```text
your text
        ↓
custom tokenizer
        ↓
train / validation dataset
        ↓
tiny GPT-style model
        ↓
local inference
```

---

## Features
- config-driven training
- simple terminal menu
- tokenizer training from your own text
- automatic train / validation dataset preparation
- tiny GPT-style model training from scratch
- local inference
- CPU fallback if CUDA is unavailable
- project status summary
- small preset model sizes: micro, tiny, small

---

## Project Structure

```text
simple-local-gpt-trainer/
  configs/
    tiny_local.yaml

  data/
    input/
      corpus.txt

  outputs/
    ...

  scripts/
    00_create_config.py
    01_prepare.py
    02_train.py
    03_inference.py
    04_run_menu.py

  docs/
  ```

---

## Quick Start

Install requirements:

```bash
pip install -r requirements.txt
```

Run the menu:

```bash
python scripts/04_run_menu.py
```

Recommended first flow:

```text
1. Check environment
2. Create config
3. Prepare dataset
4. Train model
5. Run inference
6. Show project status
```

---

## Input Text

Place your own text here:

```text
data/input/corpus.txt
```

The text can be plain .txt content.

For the first test, use a small file. Larger and cleaner datasets usually produce better results, but this project is designed to start simple.

---

## Config Example

```text
project:
  name: tiny_local_test
  input_text: data/input/corpus.txt
  output_dir: outputs/tiny_local_test

tokenizer:
  vocab_size: 4000
  min_frequency: 1

dataset:
  block_size: 128
  validation_ratio: 0.1

model:
  n_layer: 4
  n_head: 4
  n_embd: 256
  dropout: 0.1

training:
  batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 0.0003
  max_steps: 1000
  warmup_steps: 50
```

---

## Model Presets

The config generator currently supports:

```text
micro
tiny
small
```

These presets change the approximate model size by adjusting:

- number of layers
- embedding size
- attention heads
- training steps

Small models are easier to train locally, but they may overfit quickly on small datasets.

---

## Important Notes

This project is experimental.

Models trained on small datasets may:

- memorize the training text
- repeat phrases
- produce unstable outputs
- generate incorrect information
- fail outside the training domain

This is expected.

The purpose of this project is to make the training process visible and understandable.

---

## Suggested Use Cases

- learning how GPT-style training works
- experimenting with custom text
- testing small model sizes
- understanding tokenizer + model interaction
- local AI experimentation
- educational demos

---

## Not Intended For

This project is not intended to be:

- a production LLM system
- a commercial AI assistant
- a factual question-answering engine
- a replacement for large pretrained models

---

## License

This repository is released under the Apache 2.0 License.

---

**Created by MA**