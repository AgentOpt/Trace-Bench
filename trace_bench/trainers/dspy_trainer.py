"""DSPy trainer liaison for Trace-Bench.

``DSPyTrainer`` wraps four DSPy teleprompt optimisers — COPRO, MIPROv2, SIMBA,
and GEPA — inside a single Trace-Bench ``Trainer`` interface.  The ``train()``
method is the single entry point and accepts the union of all four optimisers'
constructor arguments.

Framework compatibility check
------------------------------
DSPy trainers can only optimise ``dspy.Module`` agents.  If a non-DSPy agent
is passed (e.g. a Trace ``ParameterNode`` / ``Module``), ``train()`` raises a
``TypeError`` immediately with a clear message.

The check is duck-typed: it looks for ``named_predictors`` (present on every
``dspy.Module``) and ``to_examples`` (a Trace-Bench convention on all agents).

Supported optimisers
--------------------
Pass ``dspy_optimizer`` in ``params_variants`` to select one:

* ``"mipro"`` (default) — MIPROv2, the recommended general-purpose optimiser.
* ``"copro"``           — COPRO, depth-first instruction search.
* ``"simba"``           — SIMBA, mini-batch stochastic optimiser.
* ``"gepa"``            — GEPA (requires ``reflection_lm``).

Usage in a YAML config::

    trainers:
      - id: DSPyTrainer
        params_variants:
          - dspy_optimizer: mipro
            auto: light
            num_threads: 8

    tasks:
      - id: hf:bbeh/boolean_expressions
        eval_kwargs:
          framework: dspy
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import dspy as _dspy
    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Base class — graceful fallback if OpenTrace is not installed
# ---------------------------------------------------------------------------
try:
    from opto.trainer.algorithms.algorithm import Trainer as _TrainerBase
except Exception:
    _TrainerBase = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Compatibility helpers (module-level so they can be imported independently)
# ---------------------------------------------------------------------------

def _is_dspy_agent(param: Any) -> bool:
    """Return True if *param* looks like a ``dspy.Module``."""
    return hasattr(param, "named_predictors") and callable(
        getattr(param, "named_predictors")
    )


def _check_dspy_agent(param: Any, trainer_name: str) -> None:
    """Raise ``TypeError`` if *param* is not a DSPy agent."""
    if not _is_dspy_agent(param):
        is_trace = hasattr(param, "parameters") and callable(
            getattr(param, "parameters")
        )
        hint = (
            " It looks like a Trace agent — set framework='dspy' in "
            "eval_kwargs to get a DSPy agent instead."
            if is_trace
            else ""
        )
        raise TypeError(
            f"{trainer_name} requires a DSPy agent (dspy.Module with "
            f"'named_predictors'). Got: {type(param).__name__}.{hint}"
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DSPyTrainer(_TrainerBase):
    """Trainer that optimises a DSPy agent using one of four DSPy optimisers.

    Select the optimiser with the ``dspy_optimizer`` kwarg in ``params_variants``
    (default: ``"mipro"``).  All optimiser constructor arguments are accepted by
    ``train()``; unknown kwargs are silently ignored rather than raising, so you
    can safely pass a superset.
    """

    EXTERNAL_REQUIRES: List[str] = ["dspy-ai"]

    # ------------------------------------------------------------------
    # Framework check
    # ------------------------------------------------------------------

    def _check_agent(self) -> None:
        """Raise TypeError / AttributeError if self.param is not a suitable DSPy agent."""
        _check_dspy_agent(self.param, type(self).__name__)
        if not hasattr(type(self.param), "to_examples"):
            raise AttributeError(
                f"{type(self.param).__name__} must implement a "
                "'to_examples(inputs, infos)' classmethod. "
                "See DSPyHotpotQAAgent or DSPyBBEHAgent for reference."
            )

    # ------------------------------------------------------------------
    # Metric
    # ------------------------------------------------------------------

    def _make_metric(self, guide: Any):
        """Return a single metric function compatible with all four optimisers.

        Signature: ``metric(gold, pred, trace=None, pred_name=None, pred_trace=None)``

        - The 5-arg signature satisfies GEPA's validation
          (``inspect.signature(metric).bind(None, None, None, None, None)``).
        - COPRO / MIPROv2 / SIMBA call it with 3 positional args; the last two
          default to ``None`` and the function returns a ``float``.
        - When ``pred_name`` is not ``None`` (GEPA per-predictor path), the
          function returns ``dspy.Prediction(score=..., feedback=...)`` for richer
          optimiser guidance.

        Raw task / info objects stored under ``_task`` and ``_info`` in the
        DSPy Example (set by each agent's ``to_examples()``) are used to call
        ``guide.get_feedback`` without any task-format-specific logic here.
        """
        def metric(
            gold: Any,
            pred: Any,
            trace: Any = None,
            pred_name: Any = None,
            pred_trace: Any = None,
        ):
            response = getattr(pred, "answer", str(pred))

            # Prefer the raw objects stashed by to_examples().
            task = getattr(gold, "_task", None)
            info = getattr(gold, "_info", None)

            # Fallback: reconstruct from Example fields.
            if task is None:
                if hasattr(gold, "context"):
                    from types import SimpleNamespace
                    task = SimpleNamespace(
                        question=gold.question,
                        context=gold.context,
                        answer=gold.answer,
                    )
                    info = task
                else:
                    task = gold.question
                    info = gold.answer

            score, feedback = guide.get_feedback(task, response, info)

            if pred_name is not None and _DSPY_AVAILABLE:
                # GEPA path: return rich Prediction so GEPA can use the feedback.
                return _dspy.Prediction(score=float(score), feedback=feedback)
            return float(score)

        return metric

    # ------------------------------------------------------------------
    # Optimiser instantiation helpers
    # ------------------------------------------------------------------

    def _build_copro(self, metric, prompt_model, breadth, depth, init_temperature, track_stats, **_):
        from dspy.teleprompt import COPRO
        return COPRO(
            prompt_model=prompt_model,
            metric=metric,
            breadth=breadth,
            depth=depth,
            init_temperature=init_temperature,
            track_stats=track_stats,
        )

    def _build_mipro(
        self, metric, prompt_model, task_model, num_threads, auto,
        num_candidates, max_bootstrapped_demos, max_labeled_demos,
        seed, init_temperature, verbose, track_stats, log_dir,
        teacher_settings, **_,
    ):
        from dspy.teleprompt import MIPROv2
        return MIPROv2(
            metric=metric,
            prompt_model=prompt_model,
            task_model=task_model,
            teacher_settings=teacher_settings,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            auto=auto,
            num_candidates=num_candidates,
            num_threads=num_threads,
            seed=seed,
            init_temperature=init_temperature,
            verbose=verbose,
            track_stats=track_stats,
            log_dir=log_dir,
        )

    def _build_simba(
        self, metric, prompt_model, num_threads, num_candidates,
        max_steps, bsize, max_demos, teacher_settings, **_,
    ):
        from dspy.teleprompt import SIMBA
        return SIMBA(
            metric=metric,
            bsize=bsize,
            num_candidates=num_candidates,
            max_steps=max_steps,
            max_demos=max_demos,
            prompt_model=prompt_model,
            teacher_settings=teacher_settings,
            num_threads=num_threads,
        )

    def _build_gepa(
        self, metric, reflection_lm, auto, max_metric_calls, max_full_evals,
        reflection_minibatch_size, num_threads, track_stats, log_dir,
        seed, **_,
    ):
        from dspy.teleprompt import GEPA
        if reflection_lm is None:
            raise ValueError(
                "DSPyTrainer with dspy_optimizer='gepa' requires a "
                "'reflection_lm' argument (e.g. a dspy.LM instance). "
                "Add it to params_variants in your config."
            )
        # GEPA requires exactly one budget param; prefer auto if nothing else set.
        budget_count = sum([
            auto is not None,
            max_metric_calls is not None,
            max_full_evals is not None,
        ])
        if budget_count == 0:
            auto = "light"
        return GEPA(
            metric=metric,
            auto=auto,
            max_metric_calls=max_metric_calls,
            max_full_evals=max_full_evals,
            reflection_minibatch_size=reflection_minibatch_size,
            reflection_lm=reflection_lm,
            num_threads=num_threads,
            track_stats=track_stats,
            log_dir=log_dir,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Compile helpers
    # ------------------------------------------------------------------

    def _run_compile(
        self,
        dspy_optimizer: str,
        optimizer: Any,
        agent: Any,
        trainset: List[Any],
        valset: List[Any],
        seed: int,
        num_threads: Optional[int],
    ) -> None:
        """Call the appropriate compile() signature for each optimiser.

        COPRO and SIMBA do not accept a separate validation set.  When the
        Trace-Bench bundle provides one we merge it into the training set so
        the extra labelled data is not wasted.  MIPROv2 and GEPA receive
        trainset and valset separately.
        """
        # Merge val into train for optimisers that have no valset param.
        # Guard: only merge when valset is a *distinct* list (not the trainset
        # fallback assigned when no val split exists).
        def _merged(ts, vs):
            return ts + vs if vs is not ts else ts

        if dspy_optimizer == "copro":
            eval_kwargs: Dict[str, Any] = {"display_progress": True, "display_table": 0}
            if num_threads is not None:
                eval_kwargs["num_threads"] = num_threads
            optimizer.compile(agent, trainset=_merged(trainset, valset), eval_kwargs=eval_kwargs)
        elif dspy_optimizer == "simba":
            optimizer.compile(agent, trainset=_merged(trainset, valset), seed=seed)
        else:
            # MIPROv2, GEPA — valset is kept separate for Pareto / trial scoring
            optimizer.compile(agent, trainset=trainset, valset=valset)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(
        self,
        guide: Any,
        train_dataset: Dict[str, Any],
        # ----- which optimiser -----
        dspy_optimizer: str = "mipro",
        # ----- LM settings (union) -----
        prompt_model: Any = None,
        task_model: Any = None,
        reflection_lm: Any = None,
        num_threads: Optional[int] = None,
        teacher_settings: Optional[Dict[str, Any]] = None,
        # ----- budget (union, each ignored by optimisers that don't use it) -----
        auto: Optional[str] = "light",
        num_candidates: int = 6,
        # SIMBA
        max_steps: int = 8,
        bsize: int = 32,
        max_demos: int = 4,
        # COPRO
        breadth: int = 10,
        depth: int = 3,
        # GEPA
        max_metric_calls: Optional[int] = None,
        max_full_evals: Optional[int] = None,
        reflection_minibatch_size: int = 3,
        # MIPROv2
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        # ----- general -----
        seed: int = 0,
        init_temperature: float = 1.0,
        track_stats: bool = False,
        log_dir: Optional[str] = None,
        verbose: bool = False,
        # absorb unknown Trace-Bench kwargs (num_steps, etc.) silently
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a DSPy optimisation loop.

        Args:
            guide:           Task guide (provides ``get_feedback``).
            train_dataset:   Dict with ``"inputs"`` and ``"infos"`` lists.
            dspy_optimizer:  One of ``"mipro"`` (default), ``"copro"``,
                             ``"simba"``, ``"gepa"``.
            prompt_model:    LM used for instruction generation (COPRO, MIPROv2,
                             SIMBA).  Defaults to the globally configured LM.
            task_model:      LM used for task execution (MIPROv2 only).
            reflection_lm:   LM used for reflection (GEPA; **required**).
            num_threads:     Parallel threads for evaluation.
            teacher_settings: Teacher LM settings (MIPROv2, SIMBA).
            auto:            Budget preset (``"light"``/``"medium"``/``"heavy"``)
                             for MIPROv2 and GEPA.  Ignored by COPRO and SIMBA.
            num_candidates:  Candidates per step (MIPROv2, SIMBA).
            max_steps:       SIMBA optimisation steps.
            bsize:           SIMBA mini-batch size.
            max_demos:       SIMBA max demo count per predictor.
            breadth:         COPRO breadth (new prompts per iteration).
            depth:           COPRO depth (optimisation iterations).
            max_metric_calls: GEPA budget in metric calls (mutually exclusive
                              with ``auto`` and ``max_full_evals``).
            max_full_evals:  GEPA budget in full evaluations.
            reflection_minibatch_size: GEPA reflection mini-batch size.
            max_bootstrapped_demos: MIPROv2 few-shot bootstrap limit.
            max_labeled_demos:      MIPROv2 labeled demo limit.
            seed:            Random seed.
            init_temperature: Temperature for instruction generation (COPRO,
                              MIPROv2).
            track_stats:     Record optimisation statistics (COPRO, MIPROv2,
                             GEPA).
            log_dir:         Directory for optimisation logs (MIPROv2, GEPA).
            verbose:         Verbose output (MIPROv2).

        Returns:
            ``{"status": "ok", "optimizer": dspy_optimizer}``
        """
        if not _DSPY_AVAILABLE:
            raise ImportError(
                "DSPyTrainer requires the `dspy` package. "
                "Install it with: pip install dspy-ai"
            )

        dspy_optimizer = dspy_optimizer.lower().strip()
        _SUPPORTED = {"mipro", "copro", "simba", "gepa"}
        if dspy_optimizer not in _SUPPORTED:
            raise ValueError(
                f"Unknown dspy_optimizer: {dspy_optimizer!r}. "
                f"Supported: {sorted(_SUPPORTED)}"
            )

        # --- framework compatibility check ---
        self._check_agent()

        # --- dataset ---
        inputs = train_dataset["inputs"]
        infos = train_dataset.get("infos", train_dataset.get("info", []))
        val_data = train_dataset.get("validate_dataset") or {}
        val_inputs = val_data.get("inputs", [])
        val_infos = val_data.get("infos", [])

        to_examples = type(self.param).to_examples
        trainset = to_examples(inputs, infos)
        valset = to_examples(val_inputs, val_infos) if val_inputs else trainset

        # SIMBA requires len(effective trainset) >= bsize.
        # The effective set is train+val (merged in _run_compile), so check that.
        simba_effective = len(trainset) + (len(valset) if valset is not trainset else 0)
        if dspy_optimizer == "simba" and simba_effective < bsize:
            raise ValueError(
                f"SIMBA requires at least {bsize} training examples "
                f"(got {simba_effective} after merging train+val). "
                "Reduce 'bsize' or increase 'num_train'/'num_validate'."
            )

        # --- metric (one function for all four optimisers) ---
        metric = self._make_metric(guide)

        # --- build init kwargs shared across all helpers ---
        init_kw = dict(
            metric=metric,
            prompt_model=prompt_model,
            task_model=task_model,
            reflection_lm=reflection_lm,
            num_threads=num_threads,
            teacher_settings=teacher_settings,
            auto=auto,
            num_candidates=num_candidates,
            max_steps=max_steps,
            bsize=bsize,
            max_demos=max_demos,
            breadth=breadth,
            depth=depth,
            max_metric_calls=max_metric_calls,
            max_full_evals=max_full_evals,
            reflection_minibatch_size=reflection_minibatch_size,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            seed=seed,
            init_temperature=init_temperature,
            track_stats=track_stats,
            log_dir=log_dir,
            verbose=verbose,
        )

        _builders = {
            "copro": self._build_copro,
            "mipro": self._build_mipro,
            "simba": self._build_simba,
            "gepa":  self._build_gepa,
        }
        optimizer = _builders[dspy_optimizer](**init_kw)

        # --- compile ---
        self._run_compile(dspy_optimizer, optimizer, self.param, trainset, valset, seed, num_threads)

        if hasattr(self, "logger") and self.logger is not None:
            self.logger.log({"dspy_optimizer": dspy_optimizer, "status": "ok"})

        return {"status": "ok", "optimizer": dspy_optimizer}
