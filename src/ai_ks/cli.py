import json
from pathlib import Path
from typing import Annotated

import typer

from ai_ks.config import get_settings
from ai_ks.ingestion import IngestionService

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


@app.command("ping")
def ping() -> None:
    typer.echo("pong")


@app.command("ingest")
def ingest(
    sources_path: SourcesPathOption = None,
    recreate_collection: RecreateCollectionOption = True,
) -> None:
    settings = get_settings()
    if sources_path is not None:
        settings.sources_path = sources_path

    result = IngestionService(settings).run(recreate_collection=recreate_collection)
    typer.echo(json.dumps(result.to_dict(), indent=2))

if __name__ == "__main__":
    app()
