from __future__ import annotations

from trace_bench.results import build_leaderboard_rows


def test_build_leaderboard_rows_excludes_failed_jobs():
    rows = [
        {"task_id": "t1", "suite": "s", "job_id": "j1", "trainer_id": "A", "status": "failed", "score_best": 100, "time_seconds": 1},
        {"task_id": "t1", "suite": "s", "job_id": "j2", "trainer_id": "B", "status": "ok", "score_best": 2.0, "time_seconds": 2},
        {"task_id": "t2", "suite": "s", "job_id": "j3", "trainer_id": "C", "status": "ok", "score_best": 1.0, "time_seconds": 3},
    ]
    leaderboard = build_leaderboard_rows(rows)
    assert [r["task_id"] for r in leaderboard] == ["t1", "t2"]
    assert leaderboard[0]["job_id"] == "j2"
