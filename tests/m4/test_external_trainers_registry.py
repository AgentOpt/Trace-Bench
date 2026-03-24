from trace_bench.registry import discover_trainers


def test_external_trainers_are_listed():
    ids = {spec.id for spec in discover_trainers()}
    assert "DSPy-MIPROv2" in ids
    assert "DSPy-BootstrapFewShot" in ids
    assert "TextGrad-TGD" in ids
    assert "OpenEvolve" in ids
