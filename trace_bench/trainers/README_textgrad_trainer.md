# TextGradTrainer

`TextGradTrainer` is an external Trace-Bench trainer wrapper for `opto.optimizers.textgrad.TextGrad`.

- Thin wrapper around NewTrace TextGrad.
- Supports `mode=stub` and `mode=real`.
- Uses trainable Trace parameters only.
- Can optionally keep only improving updates (`ensure_improvement=True`).
