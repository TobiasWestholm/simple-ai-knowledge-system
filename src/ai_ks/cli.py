import typer

app = typer.Typer(help="AI Knowledge System CLI")


@app.command("ping")
def ping() -> None:
    typer.echo("pong")
