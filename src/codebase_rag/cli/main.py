"""Command-line interface."""

from pathlib import Path

import typer

app = typer.Typer(
    help="Local-first RAG assistant for software repositories.",
    no_args_is_help=True,
)


@app.command()
def index(repository: Path) -> None:
    """Index a local repository."""
    typer.echo(f"Indexing is not implemented yet: {repository}")


@app.command()
def ask(question: str) -> None:
    """Ask a question about the active indexed project."""
    typer.echo(f"Querying is not implemented yet: {question}")


if __name__ == "__main__":
    app()
