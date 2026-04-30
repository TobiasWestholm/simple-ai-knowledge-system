from __future__ import annotations

from typer.testing import CliRunner

import ai_ks.cli as cli
from ai_ks.evaluation import EvaluateResponse, ToolBehaviorSummary


def test_cli_eval_prints_json(monkeypatch: object) -> None:
    class FakeEvaluationService:
        def __init__(self, settings: object) -> None:
            self.settings = settings

        def evaluate(self, request: object) -> EvaluateResponse:
            return EvaluateResponse(
                evaluation_id="eval-cli",
                suites=["tool_behavior"],
                tool_behavior=ToolBehaviorSummary(
                    total_cases=1,
                    passed_cases=1,
                    failed_cases=0,
                    cases=[],
                ),
            )

    monkeypatch.setattr(cli, "EvaluationService", FakeEvaluationService)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["eval", "--suite", "tool_behavior"])

    assert result.exit_code == 0
    assert '"evaluation_id": "eval-cli"' in result.stdout
