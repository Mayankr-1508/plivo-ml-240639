# Plivo AI/ML Track: 2,000 Step LLM Speedrun (`plivo-ml-240639`)

This repository contains the complete solution, model architecture, training scripts, evaluation harness, and trained checkpoint for the Plivo AI/ML Internship LLM Speedrun assignment.

---

## Performance & Architecture Summary

- **Model Type:** Compact GPT-style decoder-only transformer
- **Parameter Count:** 1,875,840 parameters (Strictly under the 2M cap)
- **Layers / Heads / Dim:** 4 Layers | 4 Attention Heads | 192 Embedding Dimension
- **Training Steps:** 2,000 steps
- **Tokens Scored:** 158,976 tokens
- **Average Loss:** 1.7944
- **Bits-Per-Byte (BPB):** **2.5887**

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `model.py` | Defines the decoder-only transformer architecture and attention blocks. |
| `train.py` | Training script implementing optimizer loops, warmup, and cosine decay schedule. |
| `evaluate.py` | Evaluation harness used to compute Bits-Per-Byte (BPB) and loss metrics. |
| `tokenizer.py` | Byte-level tokenization implementation. |
| `ckpt.pt` | Trained model weights checkpoint file. |
| `NOTES.md` | Summary of hyperparameters, training strategy, and architectural decisions. |
| `RUNLOG.md` | Step-by-step log of experimental runs and optimizations. |
| `SUMMARY.html` | Interactive HTML dashboard detailing performance metrics. |

---

## Quick Start & Evaluation

To run evaluation using the provided checkpoint and development evaluation corpus:

```bash
python evaluate.py --text_file dev_eval.txt

##Contributor - Mayank Raj 240639
