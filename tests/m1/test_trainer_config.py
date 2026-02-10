import pytest

from trace_bench.config import RunConfig


def test_trainer_params_variants_parsed():
    cfg = RunConfig.from_dict(
        {
            "trainers": [
                {
                    "id": "PrioritySearch",
                    "params_variants": [{"ps_steps": 2}],
                }
            ]
        }
    )
    assert cfg.trainers[0].params_variants[0]["ps_steps"] == 2


def test_trainer_missing_id_raises():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"trainers": [{"params_variants": [{}]}]})
