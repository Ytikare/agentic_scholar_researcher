"""CLI progress output."""

from scholar_agent.cli import show_progress


def test_progress_survives_brackets_in_llm_text(capsys):
    # Rich would read "[/bold]" as a closing markup tag and crash if markup weren't off.
    show_progress("review_plan", {"sub_questions": ["Is [bold] X better?"]})
    show_progress("fact_checker", {"issues": ['"claim" [2401.00001]: overstated [/bold]']})

    out = capsys.readouterr().out
    assert "Plan approved: 1 sub-question" in out
    assert "Is [bold] X better?" in out
    assert "overstated [/bold]" in out
