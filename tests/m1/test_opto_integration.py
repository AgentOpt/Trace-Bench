"""Tests for the generic opto/OpenTrace bundle-evaluation integration."""
from __future__ import annotations

import pytest

from trace_bench.integrations.opto import BundleEval, evaluate_bundle
from trace_bench.registry import normalize_task_id


class _Node:
    """Node-like param: _extract_response reads .data (production shape)."""
    def __init__(self, data="GOOD"): self.data = data


class _Guide:
    # Rewards the known-good response; mirrors a real guide scoring a fixed
    # artifact response against each example.
    def __call__(self, x, response, info):
        return (0.5 if response == "GOOD" else 0.0), f"resp={response}"

    def get_score_dict(self, x, response, info):
        return {"accuracy": 1.0 if response == "GOOD" else 0.0, "cost": 0.1}


def _bundle(**over):
    base = {"param": _Node("GOOD"), "guide": _Guide(),
            "train_dataset": {"inputs": ["a", "b"], "infos": [{}, {}]}}
    base.update(over)
    return base


def test_evaluate_bundle_returns_mean_reward_and_score_dicts():
    mean_reward, evals = evaluate_bundle(_bundle(), max_examples=2)
    assert mean_reward == 0.5
    assert [type(e) for e in evals] == [BundleEval, BundleEval]
    assert evals[0].score_dict == {"accuracy": 1.0, "cost": 0.1}


def test_evaluate_bundle_prefers_validation_split_and_caps_examples():
    bundle = _bundle(validate_dataset={"inputs": ["a"], "infos": [{}]})
    mean_reward, evals = evaluate_bundle(bundle, max_examples=5)
    assert len(evals) == 1 and mean_reward == 0.5      # validate split used, capped


def test_evaluate_bundle_unwraps_trainable_param():
    mean_reward, evals = evaluate_bundle(_bundle(param=_Node("GOOD")), max_examples=1)
    assert mean_reward == 0.5 and evals[0].feedback == "resp=GOOD"


def test_evaluate_bundle_raises_on_empty_dataset():
    with pytest.raises(ValueError, match="no evaluable examples"):
        evaluate_bundle(_bundle(train_dataset={"inputs": [], "infos": []}))


def test_normalize_task_id_is_public_and_stable():
    assert normalize_task_id("example:foo") == "trace_examples:foo"
    assert normalize_task_id("hf:GSM8K") == "hf:GSM8K"
    assert normalize_task_id("online_bin_packing") == "llm4ad:online_bin_packing"


def test_evaluate_bundle_strict_score_dict_reraises():
    class _BadDict(_Guide):
        def get_score_dict(self, x, response, info):
            raise RuntimeError("scorer unavailable")

    bundle = _bundle(guide=_BadDict())
    # default: swallow -> score_dict None, reward still present
    mean_reward, evals = evaluate_bundle(bundle, max_examples=1)
    assert evals[0].score_dict is None and mean_reward == 0.5
    # strict: re-raise so a silent None can't masquerade as a real vector
    with pytest.raises(RuntimeError, match="scorer unavailable"):
        evaluate_bundle(bundle, max_examples=1, strict_score_dict=True)


def test_evaluate_bundle_mean_reward_matches_runner_mean():
    # parity guard: aggregate reward equals the runner's own mean over the split
    from trace_bench.runner import _score_dataset, _evaluation_dataset
    bundle = _bundle()
    name, dataset = _evaluation_dataset(bundle)
    runner_summary = _score_dataset(bundle, dataset, name)
    mean_reward, _ = evaluate_bundle(bundle, max_examples=len(dataset["inputs"]))
    assert mean_reward == pytest.approx(runner_summary["score"])


def test_integrations_package_exports_public_api():
    import trace_bench.integrations as integ
    assert set(integ.__all__) == {"BundleEval", "evaluate_bundle", "load_and_evaluate"}
