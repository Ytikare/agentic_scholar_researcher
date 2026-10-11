"""Cost tracking: price math, per-node attribution through the graph, and the budget stop."""

import pytest

from scholar_agent.costs import BudgetExceededError, CostTracker, call_cost
from scholar_agent.graph import build_graph
from scholar_agent.state import Review
from tests.fakes import FakeLLM, research_llm, writer_llm


def usage(inp, out, cache_read=0):
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out,
        "input_token_details": {"cache_read": cache_read},
    }


def test_call_cost():
    assert call_cost("claude-sonnet-5-5", usage(1_000_000, 0)) == pytest.approx(2.00)
    assert call_cost("claude-sonnet-5-5", usage(0, 1_000_000)) == pytest.approx(10.00)
    # Half the prompt read from cache: 0.5M x $2 + 0.5M x $0.20
    assert call_cost("claude-sonnet-5-5", usage(1_000_000, 0, 500_000)) == pytest.approx(1.10)


def test_haiku_long_prompts_cost_more():
    assert call_cost("claude-haiku-5-5", usage(100_000, 0)) == pytest.approx(0.01)
    assert call_cost("claude-haiku-5-5", usage(200_000, 0)) == pytest.approx(0.10)


def test_unknown_model_is_not_priced():
    assert call_cost("some-other-model", usage(1000, 100)) is None


def priced(fake: FakeLLM, model: str) -> FakeLLM:
    """Make every reply report 1,000 input / 100 output tokens on `model`."""
    inner = fake.respond

    def respond(messages):
        message = inner(messages)
        message.usage_metadata = usage(1000, 100)
        message.response_metadata = {"model_name": model}
        return message

    fake.respond = respond
    return fake


def priced_graph():
    return build_graph(
        priced(research_llm(), "claude-haiku-5-5"),
        priced(writer_llm("X works well [2401.00001].", [Review(issues=[])]), "claude-sonnet-5-5"),
    )


@pytest.mark.usefixtures("fake_arxiv")
def test_costs_are_attributed_to_graph_nodes():
    tracker = CostTracker()
    priced_graph().invoke({"question": "Does X work?"}, {"callbacks": [tracker]})
    s = tracker.summary()

    # 2 researchers x 2 Haiku calls (search, then notes) and 1 Sonnet writer call.
    assert s["calls"] == 5
    assert s["by_node"]["researcher"] == pytest.approx(4 * 0.00015)
    assert s["by_node"]["writer"] == pytest.approx(0.003)
    assert s["usd"] == pytest.approx(0.0036)
    assert s["by_model"]["claude-haiku-5-5"]["input"] == 4000


@pytest.mark.usefixtures("fake_arxiv")
def test_budget_stops_the_run():
    tracker = CostTracker(budget_usd=0.001)  # researchers fit, the writer pushes it over

    with pytest.raises(BudgetExceededError, match="over the \\$0.00 budget"):
        priced_graph().invoke({"question": "Does X work?"}, {"callbacks": [tracker]})
    assert tracker.total_usd > 0.001
