from trace_bench.trainers._external_utils import apply_parameter_updates


class _ReadOnlyDataParam:
    def __init__(self, value: str) -> None:
        self._data = value

    @property
    def data(self) -> str:
        return self._data


def test_apply_parameter_updates_falls_back_to_private_data_slot_when_data_property_has_no_setter() -> None:
    parameter = _ReadOnlyDataParam("before")

    apply_parameter_updates({parameter: "after"})

    assert parameter.data == "after"
