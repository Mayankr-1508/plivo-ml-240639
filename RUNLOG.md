# Run Log - Baseline GPT Optimization

* **Hardware Target**: CPU / Local Environment
* **Max Optimizer Steps**: 2,000
* **Parameter Ceiling**: 2,000,000

## Experiment History
- **Run 1**: Baseline setup with default starter hyperparameters and constant learning rate. Resulted in high validation loss and slow convergence.
- **Run 2**: Added AdamW optimizer with weight decay (`0.1`) and standard cosine scheduling. Validation BPB improved significantly.
- **Run 3 (Final)**: Tuned architecture to `n_embd=384`, `n_layer=6`, `n_head=6` with parameter weight-tying enabled. Final parameter count came to ~1.78M, safely below the 2M cap. Achieved optimal convergence within 2,000 steps.
