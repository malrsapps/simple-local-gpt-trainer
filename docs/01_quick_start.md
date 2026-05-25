# Quick Start

This guide shows the shortest path for running Simple Local GPT Trainer.

## 1. Install requirements

```bash
pip install -r requirements.txt
```

---

## 2. Add your text

Put your plain text into:

```text
data/input/corpus.txt
```

---

## 3. Run the menu

```text
python scripts/04_run_menu.py
```

---

## 4. Recommended first flow

From the menu:

```text
3. Check environment
4. Prepare dataset
5. Train model
6. Run inference
8. Show project status
```

---

## 5. Create a custom config

Use:

```text
2. Create config
```

This lets you choose:

- project name
- context length
- model size
- training mode
- tokenizer vocabulary size

---

## Notes

The first training run is expected to be small and experimental.

Small datasets may produce repetitive or memorized output. This is normal for tiny GPT-style models.