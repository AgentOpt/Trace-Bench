from __future__ import annotations

from types import SimpleNamespace

import pytest

import benchmarks.hf_qa.hf_qa_loader as loader
from benchmarks.hf_qa import drop_qa, gsm8k, mcqa, musique, qasper
from benchmarks.hf_qa.common import HFQATask, response_text


def test_yaml_entries_exist() -> None:
    """New HF task ids resolve to the expected task semantics."""
    expected = {
        "musique": "musique",
        "gsm8k": "gsm8k",
        "arc_challenge": "mcqa",
        "qasc": "mcqa",
        "drop": "drop",
        "quality": "mcqa",
        "qasper": "qasper",
    }
    for task_id, task_type in expected.items():
        assert loader._load_task_config(task_id)["task_type"] == task_type


def test_partition_loader_same_split_nonoverlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partitions sharing a split use disjoint contiguous slices."""
    calls = []

    def fake_load(_cfg: dict, split: str, n: int) -> list[HFQATask]:
        calls.append((split, n))
        return [
            HFQATask(question=f"{split}-{index}", context="", answer=str(index))
            for index in range(n)
        ]

    monkeypatch.setattr(loader, "_load_hf_data", fake_load)
    parts = loader._load_hf_partitions(
        {},
        [("train", "train", 2), ("validate", "train", 2), ("test", "train", 1)],
    )

    assert calls == [("train", 5)]
    assert [task.answer for task in parts["train"]] == ["0", "1"]
    assert [task.answer for task in parts["validate"]] == ["2", "3"]
    assert [task.answer for task in parts["test"]] == ["4"]


def test_resolve_count_rejects_invalid_values() -> None:
    """Dataset sizes fail fast on unsafe values."""
    with pytest.raises(ValueError, match="non-negative"):
        loader._resolve_count("num_train", -1, {}, 10)
    with pytest.raises(TypeError, match="integer"):
        loader._resolve_count("num_train", True, {}, 10)


def test_response_text_extracts_chat_message_content() -> None:
    """LLM scoring uses assistant content rather than response object repr."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="final answer"))]
    )

    assert response_text(response) == "final answer"


def test_gsm8k_guide_numeric_match_and_miss() -> None:
    """GSM8K scoring extracts the final numeric answer."""
    row = {"question": "How many?", "answer": "Natalia sold 48+24 = 72. #### 72"}
    task = gsm8k.row_to_tasks(
        row, {"question_field": "question", "answer_field": "answer"}
    )[0]
    guide = gsm8k.make_guide()

    score, _ = guide.get_feedback(task.question, "Reasoning... #### 72", task)
    assert score == 1.0
    score, _ = guide.get_feedback(task.question, "Reasoning... #### 71", task)
    assert score == 0.0


def test_mcqa_arc_like_row() -> None:
    """MCQ rows with labeled choice dicts preserve labels and option text."""
    row = {
        "question": "Which factor increased?",
        "choices": {"label": ["A", "B"], "text": ["Shady areas", "Food sources"]},
        "answerKey": "B",
        "combinedfact": "Food can increase populations.",
    }
    task = mcqa.row_to_tasks(
        row,
        {
            "question_field": "question",
            "options_field": "choices",
            "answer_field": "answerKey",
            "context_field": "combinedfact",
        },
    )[0]

    assert "(A) Shady areas" in task.question
    assert task.meta["gold_label"] == "B"
    guide = mcqa.make_guide()
    score, _ = guide.get_feedback(task.question, "I choose (B).", task)
    assert score == 1.0


def test_mcqa_quality_one_based_answer() -> None:
    """QuALITY-style one-based answers map to A/B/C/D labels."""
    row = {
        "question": "Why does he retire?",
        "options": [
            "For honor",
            "Because he has enough money",
            "Because the ship broke",
            "To become mayor",
        ],
        "answer": 2,
        "article": "Long story...",
    }
    task = mcqa.row_to_tasks(
        row,
        {
            "question_field": "question",
            "options_field": "options",
            "answer_field": "answer",
            "answer_index_base": 1,
            "context_field": "article",
        },
    )[0]

    assert task.meta["gold_label"] == "B"


def test_mcqa_rejects_missing_options() -> None:
    """MCQ conversion reports malformed rows instead of silently continuing."""
    with pytest.raises(ValueError, match="no options"):
        mcqa.row_to_tasks({"question": "Q?", "answer": "A"}, {})


def test_drop_aliases_and_guide() -> None:
    """DROP scoring accepts aliases embedded in a sentence."""
    row = {
        "question": "Who scored first?",
        "passage": "Some passage",
        "answers_spans": {"spans": ["Chaz Schilens", "Schilens"]},
    }
    task = drop_qa.row_to_tasks(
        row,
        {
            "question_field": "question",
            "context_field": "passage",
            "answer_field": "answers_spans",
        },
    )[0]
    guide = drop_qa.make_guide()

    score, _ = guide.get_feedback(task, "The first scorer was Chaz Schilens.", task)
    assert score == 1.0


def test_musique_alias_support() -> None:
    """MuSiQue conversion keeps answer aliases for scoring."""
    row = {
        "question": "Where did he die?",
        "paragraphs": [{"title": "Doc", "paragraph_text": "He died in Copenhagen."}],
        "answer": "Copenhagen",
        "answer_aliases": ["Copenhagen", "Kobenhavn"],
    }
    task = musique.row_to_tasks(
        row, {"question_field": "question", "context_field": "paragraphs"}
    )[0]

    assert "Copenhagen" in task.meta["aliases"]


def test_qasper_row_expansion() -> None:
    """Original nested Qasper rows expand to one task per question."""
    row = {
        "full_text": {"section_name": ["Intro"], "paragraphs": [["A long paragraph."]]},
        "qas": {
            "question": ["What is proposed?", "Is it supervised?"],
            "question_id": ["q1", "q2"],
            "answers": [
                [
                    {
                        "answer": [
                            {
                                "free_form_answer": "A new method",
                                "extractive_spans": [],
                                "evidence": ["A long paragraph."],
                            }
                        ]
                    }
                ],
                [
                    {
                        "answer": [
                            {
                                "free_form_answer": "",
                                "extractive_spans": [],
                                "yes_no": True,
                                "evidence": ["A long paragraph."],
                            }
                        ]
                    }
                ],
            ],
        },
    }

    tasks = qasper.row_to_tasks(row, {})

    assert len(tasks) == 2
    assert tasks[0].meta["question_id"] == "q1"
    assert "A new method" in tasks[0].meta["aliases"]
    assert "yes" in [alias.lower() for alias in tasks[1].meta["aliases"]]


def test_qasper_converted_row() -> None:
    """Converted Qasper mirror rows split question and paper text."""
    row = {
        "id": "paper",
        "pid": "paper_0",
        "input": "Q: What is proposed?\nText: Intro\nA long paragraph.",
        "output": "A new method",
    }
    task = qasper.row_to_tasks(
        row, {"question_field": "input", "answer_field": "output"}
    )[0]

    assert task.question == "What is proposed?"
    assert task.context.startswith("Intro")
    assert task.answer == "A new method"
