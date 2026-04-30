import json
from pathlib import Path
from typing import Annotated

import typer

from ai_ks.config import get_settings
from ai_ks.evaluation import EvaluateRequest, EvaluationService

app = typer.Typer(help="AI Knowledge System CLI")
SourcesPathOption = Annotated[
    Path | None,
    typer.Option(
        "--sources-path",
        help="Override the default sources.yaml file.",
    ),
]
RecreateCollectionOption = Annotated[
    bool,
    typer.Option(
        "--recreate-collection/--append-to-collection",
        help="Recreate the Qdrant collection before ingesting.",
    ),
]
SuiteOption = Annotated[
    list[str] | None,
    typer.Option(
        "--suite",
        help="Evaluation suite to run. Repeat the flag to run multiple suites.",
    ),
]


@app.command("ping")
def ping() -> None:
    typer.echo("pong")


@app.command("ingest")
def ingest(
    sources_path: SourcesPathOption = None,
    recreate_collection: RecreateCollectionOption = True,
) -> None:
    from ai_ks.ingestion import IngestionService

    settings = get_settings()
    if sources_path is not None:
        settings.sources_path = sources_path

    result = IngestionService(settings).run(recreate_collection=recreate_collection)
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command("eval")
def run_eval(
    suite: SuiteOption = None,
    case_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--case-id",
            help="Run only the specified tool behavior case ids. Repeat the flag as needed.",
        ),
    ] = None,
) -> None:
    settings = get_settings()
    request = EvaluateRequest.model_validate(
        {
            "suites": suite or ["tool_behavior", "failure", "timing"],
            "case_ids": case_ids or [],
        }
    )
    result = EvaluationService(settings).evaluate(request)
    typer.echo(json.dumps(result.model_dump(), indent=2))

if __name__ == "__main__":
    app()
