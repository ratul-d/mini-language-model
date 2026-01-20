# Minimal Transformer Language Model

## About

A minimal **decoder-only Transformer language model** trained from scratch to generate **Shakespeare-like text**.
The model is implemented in PyTorch and trained at the **character level**.

* **Architecture:** Decoder-only Transformer (GPT-style)
* **Parameters:** ~10M
* **Training Hardware:** NVIDIA A100 GPU
* **Dataset:** Shakespeare text
* **Framework:** PyTorch

The model captures Shakespearean style, vocabulary, and dialogue structure, but due to its small size and character-level training, the generated text often lacks long-range coherence and semantic consistency. Outputs resemble Shakespeare in form and tone, but may contain repetition, grammatical drift, or nonsensical passages at longer lengths.

The project focuses on clarity and correctness, implementing causal self-attention, multi-head attention, residual connections, and layer normalization.

---

## Architecture

* **6 Transformer blocks**
* **Multi-head self-attention with 6 heads per block**
* **Embedding size:** 384
* **Context length:** 256 characters
* **Feed-forward expansion:** 4× embedding size
* **Causal self-attention (autoregressive)**

The model predicts the next character given a fixed-length context in a left-to-right, autoregressive manner.

---

## Usage

The model is **already trained**. To test text generation, simply run:

```bash
python models/Char-Transformer/transformer-generate.py
```

This will load the pretrained checkpoint and generate Shakespeare-like text directly in the terminal.

---

## Acknowledgements

This project follows [**Andrej Karpathy’s** tutorial](https://youtu.be/kCc8FmEb1nY?si=5Y1JoPRQO4rtUHs6) on building a GPT-style language model from scratch, where he explains the Transformer architecture, self-attention, and training process in a clear and intuitive way.

---