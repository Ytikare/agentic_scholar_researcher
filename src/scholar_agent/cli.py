"""Command-line entry point: `uv run scholar "your research question"`."""

import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown

from scholar_agent.graph import build_graph

console = Console()


def main() -> None:
    question = " ".join(sys.argv[1:]) or console.input("[bold]Research question:[/] ")
    graph = build_graph()
    final: AIMessage | None = None

    # stream_mode="updates" yields {node_name: state_update} after each node runs,
    # so we can show the agent's progress live.
    for update in graph.stream(
        {"messages": [HumanMessage(question)]},
        config={"recursion_limit": 25},  # hard cap on agent<->tools loops
        stream_mode="updates",
    ):
        for node, data in update.items():
            for msg in data["messages"]:
                if isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        console.print(f"[cyan]→ {call['name']}[/] {call['args']}")
                    if not msg.tool_calls:
                        final = msg
                elif isinstance(msg, ToolMessage):
                    n = msg.content.count("\nURL: ")
                    console.print(f"[dim]  ← {node}: {n} papers[/]")

    if final is not None:
        console.rule("Answer")
        console.print(Markdown(final.text))
        usage = final.usage_metadata or {}
        tokens_in, tokens_out = usage.get("input_tokens"), usage.get("output_tokens")
        console.print(f"[dim]last call tokens: in={tokens_in} out={tokens_out}[/]")


if __name__ == "__main__":
    main()
