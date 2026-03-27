"""No-op trainer — a minimal working example and copy-paste template.

``NoOpTrainer`` runs the agent on every sample and collects feedback, but never
calls the optimizer.  The parameter is returned unchanged after training.  This
is useful as:

* a **baseline** (measures how well the un-tuned agent already performs),
* a **smoke test** (verifies the task bundle wires up correctly), and
* a **template** for external-library trainers (copy this file, replace the
  ``# TODO`` section with your library's update call, and add the optional
  dependency to ``extras_require`` in setup.py).

Usage in a YAML config::

    trainers:
      - id: NoOpTrainer
"""
from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Base class — graceful fallback if OpenTrace is not installed
# ---------------------------------------------------------------------------
try:
    from opto.trainer.algorithms.algorithm import Trainer as _TrainerBase
except Exception:
    _TrainerBase = object  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class NoOpTrainer(_TrainerBase):
    """Trainer that never updates the parameter.

    Iterates over the training dataset, calls the agent, and asks the guide for
    feedback — but makes no optimizer call, leaving ``self.param`` unchanged.

    To turn this into a real external-library trainer:

    1. Add an optional-import guard at the top (``try: import mylib``).
    2. Replace the ``# TODO`` comment below with your library's update call.
    3. Add the library to ``extras_require`` in ``setup.py``.
    4. Set ``EXTERNAL_REQUIRES`` to the pip package name(s).
    """

    EXTERNAL_REQUIRES: List[str] = []  # no extra dependencies needed

    def train(
        self,
        guide: Any,
        train_dataset: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Iterate over data, collect feedback, skip the optimizer step.

        Args:
            guide: The task's Guide instance (provides ``get_feedback``).
            train_dataset: Dict with ``"inputs"`` and ``"infos"`` lists.
            **kwargs: Resolved trainer kwargs from config (``num_steps``, etc.).

        Returns:
            ``{"status": "ok", "scores": [...]}``
        """
        inputs = train_dataset["inputs"]
        infos = train_dataset.get("infos", train_dataset.get("info", []))
        num_steps = kwargs.get("num_steps", len(inputs))

        scores: List[Any] = []
        for step, (x, ref) in enumerate(zip(inputs, infos)):
            if step >= num_steps:
                break

            y = self.param(x)
            score, _feedback = guide.get_feedback(x, y, ref)
            scores.append(score)

            # TODO: replace this comment with your external library's update call.
            #   e.g.  mylib.step(param=self.param, score=score, feedback=_feedback)

            if hasattr(self, "logger") and self.logger is not None:
                self.logger.log({"step": step, "score": score})

        return {"status": "ok", "scores": scores}
