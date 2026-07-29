# Technical Notes

## Architectural Decisions
1. **Parameter Maximization**: Configured the model depth and embedding space (`n_layer=6`, `n_embd=384`) to maximize capacity right up to the 2M parameter limit without crossing it.
2. **Weight Tying**: Shared weights between the token embedding layer and the final language model head to save massive parameter overhead, reallocating budget toward deeper attention layers.
3. **Regularization**: Integrated Dropout (`0.1`) and AdamW weight decay to prevent overfitting on the mixed English/Hindi training corpus.
4. **Learning Rate Schedule**: Implemented a 100-step linear warmup followed by cosine annealing to stabilize early training dynamics.
