from trace_bench.terminal_bench.atif import (
    summarize_atif,
    atif_to_trace_like,
    atif_feedback_snippet,
)


def test_atif_helpers_smoke():
    payload = {
        "steps": [
            {
                "role": "assistant",
                "content": "Plan: list files, then run tests.",
                "tool_calls": [{"name": "bash", "args": {"cmd": "ls -la"}}],
            },
            {
                "role": "tool",
                "content": "file1\nfile2\n",
            },
        ]
    }

    summary = summarize_atif(payload, last_n=1)
    assert summary.n_steps == 2
    assert len(summary.last_steps) == 1

    trace_like = atif_to_trace_like(payload, last_n=10)
    assert trace_like["format"] == "trace_bench.atif.v1"
    assert trace_like["n_steps"] == 2
    assert len(trace_like["events"]) == 2

    snippet = atif_feedback_snippet(payload, max_chars=2000, last_n=2)
    assert "ATIF" in snippet
    assert "steps=2" in snippet
