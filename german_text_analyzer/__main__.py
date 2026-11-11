"""CLI entry point for German Text Analyzer."""

from __future__ import annotations

import json
import sys

try:
    import click
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print("CLI requires: pip install click rich")
    sys.exit(1)

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """German Text Analyzer -- NLP pipeline for German text."""
    pass


@cli.command()
@click.option("--input", "-i", "input_file", type=click.Path(exists=True), help="Input text file")
@click.option("--text", "-t", help="Direct text input")
@click.option("--output", "-o", help="Output JSON file")
@click.option("--no-sentiment", is_flag=True, help="Skip sentiment analysis")
@click.option("--no-ner", is_flag=True, help="Skip named entity recognition")
def analyze(input_file, text, output, no_sentiment, no_ner):
    """Analyze German text."""
    from german_text_analyzer import GermanAnalyzer

    if not input_file and not text:
        console.print("[red]Provide --input FILE or --text TEXT[/red]")
        sys.exit(1)

    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

    analyzer = GermanAnalyzer(
        enable_sentiment=not no_sentiment,
        enable_ner=not no_ner,
    )

    with console.status("Analyzing..."):
        result = analyzer.analyze(text)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        console.print(f"[green]Results saved to {output}[/green]")
    else:
        _display_results(result)


@cli.command()
def interactive():
    """Interactive analysis mode."""
    from german_text_analyzer import GermanAnalyzer

    console.print("[bold]German Text Analyzer[/bold] -- Interactive Mode")
    console.print("Type German text and press Enter. Type 'quit' to exit.\n")

    analyzer = GermanAnalyzer()

    while True:
        try:
            text = console.input("[bold blue]> [/bold blue]")
        except (EOFError, KeyboardInterrupt):
            break

        if text.lower() in ("quit", "exit", "q"):
            break

        if not text.strip():
            continue

        result = analyzer.analyze(text)
        _display_results(result)
        console.print()


def _display_results(result):
    """Display analysis results in a formatted table."""
    console.print(f"\n[bold]Analysis Summary[/bold]")
    console.print(result.summary())

    if result.entities:
        table = Table(title="Named Entities")
        table.add_column("Text", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Description")

        for entity in result.entities:
            table.add_row(entity.text, entity.label, entity.description)

        console.print(table)

    if result.compounds:
        table = Table(title="Compound Words")
        table.add_column("Word", style="cyan")
        table.add_column("Components", style="yellow")

        for compound in result.compounds:
            table.add_row(
                compound.word,
                " + ".join(compound.components),
            )

        console.print(table)


if __name__ == "__main__":
    cli()
