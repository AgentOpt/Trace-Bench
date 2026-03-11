"""Multi-objective GSM8K task (error + tokens_in + tokens_out).

Uses BasicLearner (trainable system prompt -> LLM -> math answer) with
UsageTrackingLLM for token counting.

Three objectives (all minimize):
    error      -- 0.0 correct, 1.0 wrong
    tokens_in  -- prompt tokens
    tokens_out -- completion tokens

Guide architecture (from Xavier's design):
    GSM8KGuide            -- scores by exact-match on extracted final answer
    TokenUsageAugmentingGuide -- wraps GSM8KGuide + UsageTrackingLLM to add
                                 token counts to the score dict
"""
from __future__ import annotations

import contextvars
import copy
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from opto.trainer.guide import Guide
from opto.trainer.objectives import ObjectiveConfig
from opto.utils.llm import LLM
from opto.features.predefined_agents import BasicLearner


# ---------------------------------------------------------------------------
# Constants / answer extraction (from Xavier's diff)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


def _extract_final_answer(text: str) -> Optional[str]:
    """Extract GSM8K final answer.

    Prefer the canonical ``#### <answer>`` format; otherwise fall back to
    the last number-like span in the text.
    """
    if text is None:
        return None
    s = str(text)

    m = re.search(r"####\s*(.*)", s)
    if m:
        tail = m.group(1).strip()
        n = _NUM_RE.search(tail)
        if n:
            return n.group(0).replace(",", "").strip()

    nums = _NUM_RE.findall(s)
    if not nums:
        return None
    return nums[-1].replace(",", "").strip()


def _normalize_answer(ans: Optional[str]) -> Optional[str]:
    """Normalize simple numeric formatting differences (e.g. '18.0' -> '18')."""
    if ans is None:
        return None
    a = ans.strip().replace(",", "")
    if re.fullmatch(r"-?\d+", a):
        return str(int(a))
    if re.fullmatch(r"-?\d+\.\d+", a):
        try:
            f = float(a)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
    return a


# ---------------------------------------------------------------------------
# Embedded GSM8K examples (fallback when `datasets` is not installed)
# ---------------------------------------------------------------------------

_EMBEDDED_EXAMPLES: List[Dict[str, str]] = [
    {
        "question": (
            "Natalia sold clips to 48 of her friends in April, and then she "
            "sold half as many clips in May. How many clips did Natalia sell "
            "altogether in April and May?"
        ),
        "answer": (
            "Natalia sold 48/2 = 24 clips in May.\n"
            "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
            "#### 72"
        ),
    },
    {
        "question": (
            "Weng earns $12 an hour for babysitting. Yesterday, she just did "
            "50 minutes of babysitting. How much did she earn?"
        ),
        "answer": (
            "Weng earns 12/60 = $0.2 per minute.\n"
            "Since she babysat for 50 minutes, she earned 0.2 x 50 = $10.\n"
            "#### 10"
        ),
    },
    {
        "question": (
            "Betty is saving money for a new wallet which costs $100. Betty "
            "has only half of the money she needs. Her parents decided to give "
            "her $15 for that purpose, and her grandparents twice as much as "
            "her parents. How much more money does Betty need to buy the wallet?"
        ),
        "answer": (
            "In the beginning, Betty has only 100 / 2 = $50.\n"
            "Betty's grandparents gave her 15 * 2 = $30.\n"
            "This means, Betty needs 100 - 50 - 30 - 15 = $5 more.\n"
            "#### 5"
        ),
    },
    {
        "question": (
            "Julie is reading a 120-page book. Yesterday, she was able to "
            "read 12 pages and today, she read twice as many pages as "
            "yesterday. If she wants to read half of the remaining pages "
            "tomorrow, how many pages should she read?"
        ),
        "answer": (
            "Julie read 12 x 2 = 24 pages today.\n"
            "So she was able to read a total of 12 + 24 = 36 pages since yesterday.\n"
            "There are 120 - 36 = 84 pages left to be read.\n"
            "Since she wants to read half of the remaining pages tomorrow, "
            "she should read 84 / 2 = 42 pages.\n"
            "#### 42"
        ),
    },
    {
        "question": (
            "James writes a 3-page letter to 2 different friends twice a "
            "week. How many pages does he write a year?"
        ),
        "answer": (
            "He writes each friend 3*2=6 pages a week.\n"
            "So he writes 6*2=12 pages every week.\n"
            "That means he writes 12*52=624 pages a year.\n"
            "#### 624"
        ),
    },
    {
        "question": (
            "Albert is wondering how much pizza he can eat in one day. He "
            "buys 2 large pizzas and 2 small pizzas. A large pizza has 16 "
            "slices and a small pizza has 8 slices. If he eats it all, how "
            "many pieces does he eat that day?"
        ),
        "answer": (
            "He eats 2*16=32 slices of large pizza.\n"
            "He eats 2*8=16 slices of small pizza.\n"
            "He eats a total of 32+16=48 slices.\n"
            "#### 48"
        ),
    },
    {
        "question": (
            "Ken created a care package to send to his brother, who was away "
            "at boarding school. Ken placed a box on a scale, and then he "
            "poured into the box enough jelly beans to bring the weight to 2 "
            "pounds. Then, he added enough brownies to cause the weight to "
            "triple. Next, he added another 2 pounds of jelly beans. And "
            "finally, he added enough gummy worms to double the weight once "
            "again. What was the final weight of the box of goodies, in pounds?"
        ),
        "answer": (
            "After adding the jelly beans, the weight was 2 pounds.\n"
            "After adding the brownies, the weight tripled to 2*3=6 pounds.\n"
            "After adding more jelly beans, the weight was 6+2=8 pounds.\n"
            "After adding the gummy worms, the weight doubled to 8*2=16 pounds.\n"
            "#### 16"
        ),
    },
    {
        "question": (
            "A farmer has 3 fields. The first field has 20 rows of corn with "
            "8 stalks in each row. The second field has 15 rows with 12 "
            "stalks each. The third field has 10 rows with 6 stalks each. "
            "How many stalks of corn does the farmer have in total?"
        ),
        "answer": (
            "First field: 20 * 8 = 160 stalks.\n"
            "Second field: 15 * 12 = 180 stalks.\n"
            "Third field: 10 * 6 = 60 stalks.\n"
            "Total: 160 + 180 + 60 = 400 stalks.\n"
            "#### 400"
        ),
    },
    {
        "question": (
            "Lisa has 5 times as many marbles as Tom. Tom has 3 times as "
            "many marbles as Alex. If Alex has 4 marbles, how many marbles "
            "do they have altogether?"
        ),
        "answer": (
            "Tom has 3 * 4 = 12 marbles.\n"
            "Lisa has 5 * 12 = 60 marbles.\n"
            "Altogether: 4 + 12 + 60 = 76 marbles.\n"
            "#### 76"
        ),
    },
    {
        "question": (
            "A store sold 65 apples on Monday, twice as many on Tuesday, "
            "and 15 fewer than Tuesday on Wednesday. How many apples were "
            "sold in the three days?"
        ),
        "answer": (
            "Tuesday: 65 * 2 = 130 apples.\n"
            "Wednesday: 130 - 15 = 115 apples.\n"
            "Total: 65 + 130 + 115 = 310 apples.\n"
            "#### 310"
        ),
    },
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_gsm8k_data(
    n_train: int = 10,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Load GSM8K data from HuggingFace or embedded fallback."""
    try:
        import datasets  # type: ignore

        ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(ds))[:n_train].tolist()
        return [
            {"question": ds[int(i)]["question"], "answer": ds[int(i)]["answer"]}
            for i in indices
        ]
    except Exception:
        rng = np.random.RandomState(seed)
        items = list(_EMBEDDED_EXAMPLES)
        indices = list(range(len(items)))
        rng.shuffle(indices)
        return [items[i] for i in indices[:n_train]]


# ---------------------------------------------------------------------------
# UsageTrackingLLM (from Xavier's diff)
# ---------------------------------------------------------------------------

class UsageTrackingLLM:
    """LLM wrapper that records token usage from provider responses.

    Uses ``response.usage.{prompt_tokens,completion_tokens}`` when available.
    Falls back to conservative whitespace-split estimates when usage metadata
    is missing (to avoid treating 'missing' as 'free').

    Thread-safe via ``contextvars.ContextVar``.
    """

    def __init__(self, base_llm: Any):
        self._base = base_llm
        self._last_usage: contextvars.ContextVar[Optional[Dict[str, int]]] = (
            contextvars.ContextVar("last_llm_usage", default=None)
        )

    def __deepcopy__(self, memo):
        """ContextVar cannot be deepcopied; create a fresh one."""
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        new._base = copy.deepcopy(self._base, memo)
        new._last_usage = contextvars.ContextVar("last_llm_usage", default=None)
        return new

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", None)

        resp = self._base(*args, **kwargs)

        # Try to read provider-supplied usage metadata.
        usage = None
        if hasattr(resp, "usage"):
            usage = getattr(resp, "usage")
        elif isinstance(resp, dict):
            usage = resp.get("usage")

        prompt_toks: Optional[int] = None
        completion_toks: Optional[int] = None
        if usage is not None:
            if isinstance(usage, dict):
                prompt_toks = usage.get("prompt_tokens")
                completion_toks = usage.get("completion_tokens")
            else:
                prompt_toks = getattr(usage, "prompt_tokens", None)
                completion_toks = getattr(usage, "completion_tokens", None)

        # Conservative fallback: estimate from messages + completion.
        if prompt_toks is None:
            if isinstance(messages, list):
                prompt_text = " ".join(
                    str(m.get("content", ""))
                    for m in messages
                    if isinstance(m, dict)
                )
                prompt_toks = len(prompt_text.split())
            else:
                prompt_toks = 0

        if completion_toks is None:
            completion_text = None
            try:
                completion_text = resp.choices[0].message.content
            except Exception:
                completion_text = None
            completion_toks = (
                len(str(completion_text).split()) if completion_text else 0
            )

        self._last_usage.set(
            {"tokens_in": int(prompt_toks or 0), "tokens_out": int(completion_toks or 0)}
        )
        return resp

    def last_usage(self) -> Dict[str, int]:
        """Return token counts from the most recent call."""
        u = self._last_usage.get()
        if not u:
            return {"tokens_in": 0, "tokens_out": 0}
        return {
            "tokens_in": int(u.get("tokens_in", 0)),
            "tokens_out": int(u.get("tokens_out", 0)),
        }


# ---------------------------------------------------------------------------
# GSM8KGuide (from Xavier's diff)
# ---------------------------------------------------------------------------

class GSM8KGuide(Guide):
    """Scores GSM8K by exact match on extracted final answer."""

    def _error(self, response: str, reference: str) -> float:
        pred = _normalize_answer(_extract_final_answer(response))
        gold = _normalize_answer(_extract_final_answer(reference))
        return 0.0 if (pred is not None and gold is not None and pred == gold) else 1.0

    def get_feedback(
        self, query, response, reference=None, **kwargs,
    ) -> Tuple[float, str]:
        err = self._error(str(response), str(reference))
        reward = -float(err)  # higher is better
        feedback = (
            f"expected={_extract_final_answer(str(reference))} "
            f"predicted={_extract_final_answer(str(response))}"
        )
        return reward, feedback

    def get_score_dict(
        self, query, response, reference=None, **kwargs,
    ) -> Dict[str, float]:
        err = self._error(str(response), str(reference))
        return {"error": float(err)}


# ---------------------------------------------------------------------------
# TokenUsageAugmentingGuide (from Xavier's diff)
# ---------------------------------------------------------------------------

class TokenUsageAugmentingGuide(Guide):
    """Wrap another guide and add token-usage objectives (tokens_in, tokens_out).

    ``get_score_dict()`` returns the base guide's metrics plus
    ``tokens_in`` and ``tokens_out`` read from a shared ``UsageTrackingLLM``.
    """

    def __init__(self, base_guide: Guide, token_llm: UsageTrackingLLM):
        self._base = base_guide
        self._token_llm = token_llm

    def get_feedback(
        self, query, response, reference=None, **kwargs,
    ) -> Tuple[float, str]:
        reward, feedback = self._base.get_feedback(
            query, response, reference=reference, **kwargs,
        )
        u = self._token_llm.last_usage()
        feedback = (
            f"{feedback} tokens_in={u['tokens_in']} tokens_out={u['tokens_out']}"
        )
        return float(reward), feedback

    def get_score_dict(
        self, query, response, reference=None, **kwargs,
    ) -> Dict[str, float]:
        sd = dict(
            self._base.get_score_dict(
                query, response, reference=reference, **kwargs,
            )
        )
        u = self._token_llm.last_usage()
        sd["tokens_in"] = float(u["tokens_in"])
        sd["tokens_out"] = float(u["tokens_out"])
        return sd


# ---------------------------------------------------------------------------
# Trace-Bench entry point
# ---------------------------------------------------------------------------

def build_trace_problem(**eval_kwargs):
    """Build the GSM8K multi-objective task bundle.

    Keyword args (via ``eval_kwargs`` in YAML config):
        n_train        : int  -- training examples   (default 10)
        seed           : int  -- random seed          (default 42)
        model          : str  -- LLM model name       (default: LLM() default)
        objective_mode : str  -- "weighted" (default) or "pareto"
    """
    n_train = eval_kwargs.get("n_train", 10)
    seed = eval_kwargs.get("seed", 42)
    model_name = eval_kwargs.get("model", None)
    objective_mode = eval_kwargs.get("objective_mode", "weighted")

    examples = _load_gsm8k_data(n_train, seed)

    # LLM wrapped in UsageTrackingLLM for token counting.
    base_llm = LLM(model=model_name) if model_name else LLM()
    tracked_llm = UsageTrackingLLM(base_llm)

    # BasicLearner: trainable system_prompt -> LLM -> math answer.
    agent = BasicLearner(
        system_prompt=(
            "You are a math problem solver. Solve the problem step by step. "
            "End your answer with #### followed by the final numeric answer."
        ),
        user_prompt_template="{message}",
        llm=tracked_llm,
    )

    # Two-layer guide: GSM8KGuide for correctness, augmented with token counts.
    base_guide = GSM8KGuide()
    guide = TokenUsageAugmentingGuide(base_guide=base_guide, token_llm=tracked_llm)

    train_dataset = dict(
        inputs=[ex["question"] for ex in examples],
        infos=[ex["answer"] for ex in examples],
    )

    objective_config = ObjectiveConfig(
        mode=objective_mode,
        weights={"error": 1.0, "tokens_in": 1e-3, "tokens_out": 1e-3},
        minimize=frozenset({"error", "tokens_in", "tokens_out"}),
        seed=seed,
    )

    return dict(
        param=agent,
        guide=guide,
        train_dataset=train_dataset,
        optimizer_kwargs=dict(
            objective=(
                "Improve the system prompt to solve math word problems accurately "
                "and concisely. End with #### <number>."
            ),
            memory_size=10,
        ),
        metadata=dict(
            benchmark="multiobjective",
            entry="gsm8k",
            objective_mode=objective_mode,
            n_train=n_train,
        ),
        objective_config=objective_config,
    )


__all__ = [
    "build_trace_problem",
    "GSM8KGuide",
    "TokenUsageAugmentingGuide",
    "UsageTrackingLLM",
]
