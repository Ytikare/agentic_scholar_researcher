"""Dollar cost tracking for every LLM call, per model and per graph node, with a budget.

Attach a CostTracker as a callback: `graph.invoke(input, {"callbacks": [tracker]})`.
LangChain reports every finished LLM call to it, including calls made inside nodes and
inside the researchers' own ReAct loops, from any thread.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float  # 5-minute cache writes cost 1.25x input


# Anthropic first-party prices, checked 2026-10-11. Thinking tokens bill as output.
PRICES = {
    "claude-opus-5-5": Price(4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125),
}
# Haiku 5.5 charges more for long prompts.
LONG_PROMPT_TOKENS = 100_000
LONG_PROMPT_PRICES = {"claude-haiku-5-5": Price(0.50, 2.50, 0.05, 0.625)}


def price_for(model: str, prompt_tokens: int) -> Price | None:
    for name, price in PRICES.items():
        if model.startswith(name):  # tolerate dated or suffixed model names
            if prompt_tokens > LONG_PROMPT_TOKENS and name in LONG_PROMPT_PRICES:
                return LONG_PROMPT_PRICES[name]
            return price
    return None


def call_cost(model: str, usage: dict) -> float | None:
    """Cost of one LLM call from LangChain's `usage_metadata`. None if the model is unpriced."""
    details = usage.get("input_token_details") or {}
    cache_read = details.get("cache_read") or 0
    cache_write = details.get("cache_creation") or 0
    # LangChain's input_tokens already includes the cached tokens.
    uncached = usage["input_tokens"] - cache_read - cache_write
    price = price_for(model, usage["input_tokens"])
    if price is None:
        return None
    return (
        uncached * price.input
        + cache_read * price.cache_read
        + cache_write * price.cache_write
        + usage["output_tokens"] * price.output
    ) / 1_000_000


class BudgetExceededError(RuntimeError):
    pass


class CostTracker(BaseCallbackHandler):
    """Adds up tokens and dollars per model and per node; optionally enforces a budget."""

    # Without this, LangChain logs exceptions raised in callbacks and carries on, and the
    # budget would never stop anything.
    raise_error = True

    def __init__(self, budget_usd: float | None = None):
        self.budget_usd = budget_usd
        self.total_usd = 0.0
        self.calls = 0
        self.by_model: dict[str, dict] = defaultdict(lambda: {"input": 0, "output": 0, "usd": 0.0})
        self.by_node: dict[str, float] = defaultdict(float)
        self.unpriced: set[str] = set()
        self._node_of_run: dict = {}
        self._lock = threading.Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        # e.g. "researcher:<id>|agent:<id>" -> "researcher" (the top-level graph node)
        ns = (metadata or {}).get("langgraph_checkpoint_ns", "")
        self._node_of_run[run_id] = ns.split(":")[0] or "other"

    def on_llm_end(self, response, *, run_id, **kwargs):
        node = self._node_of_run.pop(run_id, "other")
        try:
            message = response.generations[0][0].message
        except (IndexError, AttributeError):
            return
        if not isinstance(message, AIMessage) or not message.usage_metadata:
            return
        model = message.response_metadata.get("model_name", "unknown")
        usage = message.usage_metadata
        cost = call_cost(model, usage)

        with self._lock:
            self.calls += 1
            m = self.by_model[model]
            m["input"] += usage["input_tokens"]
            m["output"] += usage["output_tokens"]
            if cost is None:
                self.unpriced.add(model)
                return
            m["usd"] += cost
            self.by_node[node] += cost
            self.total_usd += cost
            over = self.budget_usd is not None and self.total_usd > self.budget_usd

        if over:
            raise BudgetExceededError(
                f"Spent ${self.total_usd:.3f}, over the ${self.budget_usd:.2f} budget"
            )

    def summary(self) -> dict:
        return {
            "usd": round(self.total_usd, 4),
            "calls": self.calls,
            "by_model": {k: {**v, "usd": round(v["usd"], 4)} for k, v in self.by_model.items()},
            "by_node": {k: round(v, 4) for k, v in sorted(self.by_node.items())},
            "unpriced_models": sorted(self.unpriced),
        }
