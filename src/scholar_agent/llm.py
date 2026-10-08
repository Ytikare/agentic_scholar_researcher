"""Factory for the chat models used by the agent nodes."""

from langchain_anthropic import ChatAnthropic

from scholar_agent.config import get_settings


def get_llm(model: str | None = None) -> ChatAnthropic:
    s = get_settings()
    extra = {"effort": s.llm_effort} if s.llm_effort else {}
    return ChatAnthropic(
        model=model or s.llm_model,
        api_key=s.anthropic_api_key,
        max_tokens=16000,
        # If a safety classifier declines a request, let the API retry it on
        # a suitable fallback model instead of failing the whole run.
        betas=["server-side-fallback-2026-07-01"],
        model_kwargs={"fallbacks": "default"},
        **extra,
    )
