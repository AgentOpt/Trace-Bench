# Adding a Trainer

Trainers are optimization algorithms that drive the training loop. Trace-Bench discovers them automatically from OpenTrace.

## How Trainers Are Discovered

The registry scans two package trees:

1. `opto.trainer.algorithms` -- core algorithms shipped with OpenTrace
2. `opto.features` -- extension algorithms (GEPA variants, PrioritySearch examples)

Any class that subclasses the trainer base class (`Trainer` / `Algorithm` from `opto.trainer.algorithms.algorithm`) is registered. The discovery uses both runtime introspection and AST parsing as a fallback for modules that fail to import.

## Trainer Aliases

Some trainers have friendly aliases:

| Class Name | Alias |
|-----------|-------|
| `GEPAAlgorithmBase` | `GEPA-Base` |
| `GEPAUCBSearch` | `GEPA-UCB` |
| `GEPABeamPareto` | `GEPA-Beam` |

Use either the alias or the class name in configs. `PrioritySearch` is the most commonly used trainer and is the default if no trainer is specified.

## The train() Method Contract

Every trainer must implement `train()`. The runner calls it as:

```python
trainer.train(
    guide,
    train_dataset,
    **resolved_trainer_kwargs,
)
```

Where:
- `guide` -- the task's Guide instance
- `train_dataset` -- dict with `inputs` and `infos`
- `**resolved_trainer_kwargs` -- merged kwargs from config `params_variants`, `trainer_kwargs`, and task defaults

The trainer is instantiated with the agent (`param`) and optimizer kwargs:

```python
trainer_cls(
    param,
    algorithm=resolved_algorithm,
    optimizer=optimizer_class,
    optimizer_kwargs=merged_optimizer_kwargs,
    logger=logger_instance,
    **extra_kwargs,
)
```

## Trainer Configuration

In a YAML config, trainers are specified with optional params variants:

```yaml
trainers:
  - id: PrioritySearch
    optimizer: OptoPrime
    optimizer_kwargs:
      memory_size: 10
    params_variants:
      - ps_steps: 3
        ps_batches: 2
        ps_candidates: 4
      - ps_steps: 5
        ps_batches: 3
        ps_candidates: 8

  - id: GEPA-UCB
    params_variants:
      - gepa_iters: 5
        gepa_train_bs: 4
```

Each entry in `params_variants` creates a separate job in the matrix. This lets you sweep hyperparameters within a single run.

## Allowed Trainer kwargs

The following kwargs are allowed in `params_variants`:

| Category | Keys |
|----------|------|
| General | `verbose`, `threads`, `num_threads` |
| Iteration control | `num_epochs`, `num_steps`, `num_batches`, `num_candidates`, `num_proposals`, `num_iters`, `num_search_iterations`, `train_batch_size` |
| PrioritySearch | `ps_steps`, `ps_batches`, `ps_candidates`, `ps_proposals`, `ps_mem_update` |
| GEPA | `gepa_iters`, `gepa_train_bs`, `gepa_merge_every`, `gepa_pareto_subset` |
| Merge | `merge_every`, `pareto_subset_size` |

Unknown kwargs will cause validation errors in strict mode.

## PrioritySearch Example Trainers

OpenTrace ships example trainers built on PrioritySearch:

- `SequentialUpdate`
- `SequentialSearch`
- `BeamSearch`

These require a specific positional-args fix in OpenTrace. The registry automatically detects whether the fix is present and marks them as available or unavailable accordingly.

## Adding a New Trainer to OpenTrace

To make a new trainer available in Trace-Bench:

1. Implement it as a subclass of `Trainer` or `Algorithm` in `opto.trainer.algorithms` or `opto.features`.
2. Ensure it has a `train(guide, train_dataset, **kwargs)` method.
3. Install / update your OpenTrace package.
4. Verify discovery: `trace-bench list-trainers --all`.

No changes to Trace-Bench code are needed. Discovery is automatic.

## Guide and Logger Overrides

Trainers can specify guide and logger overrides:

```yaml
trainers:
  - id: PrioritySearch
    guide: MyCustomGuide          # override the task's guide
    guide_kwargs: {}
    logger: ConsoleLogger         # or TensorboardLogger, WandbLogger, none
    logger_kwargs: {}
```

The CLI also supports a global logger override: `--logger none` disables logging for all trainers.

## Reference Trainers

- **PrioritySearch** -- the default and most widely tested trainer
- **GEPA-Base**, **GEPA-UCB**, **GEPA-Beam** -- Genetic/Evolutionary Population-based Algorithms
- **SequentialUpdate**, **SequentialSearch**, **BeamSearch** -- PrioritySearch examples (when supported)
