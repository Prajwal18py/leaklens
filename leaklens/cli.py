"""Optional CLI: pip install leaklens[cli]
Usage: leaklens check train.csv test.csv --target price --report report.html
"""
import pandas as pd

try:
    import typer
except ImportError:
    typer = None

from .core import LeakLens

if typer is not None:
    app = typer.Typer(help="LeakLens — leakage & drift QA for ML datasets.")

    @app.command()
    def check(
        train_path: str = typer.Argument(..., help="Path to train CSV"),
        test_path: str = typer.Argument(None, help="Path to test CSV (optional)"),
        target: str = typer.Option(None, help="Target column name"),
        report: str = typer.Option(None, help="Path to write an HTML report"),
        json_out: str = typer.Option(None, "--json", help="Path to write a JSON report"),
    ):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path) if test_path else None
        result = LeakLens(train, test, target=target).run()
        result.summary()
        if report:
            result.to_html(report)
            typer.echo(f"\nHTML report saved to {report}")
        if json_out:
            result.to_json(json_out)
            typer.echo(f"JSON report saved to {json_out}")

    def main():
        app()
else:
    def main():
        raise ImportError("The CLI requires the 'cli' extra: pip install leaklens[cli]")

if __name__ == "__main__":
    main()
