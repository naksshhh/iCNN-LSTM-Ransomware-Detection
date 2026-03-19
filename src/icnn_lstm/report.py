from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _safe_float(v: float) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _build_summary(df: pd.DataFrame, run: dict) -> dict:
    summary = {
        "num_incremental_batches": int(run.get("num_incremental_batches", len(df))),
        "initial_train_seconds": _safe_float(run.get("initial_train_seconds", 0.0)),
        "total_runtime_seconds": _safe_float(run.get("total_runtime_seconds", 0.0)),
        "mean_f2": _safe_float(df["f2"].mean() if len(df) else 0.0),
        "median_f2": _safe_float(df["f2"].median() if len(df) else 0.0),
        "max_f2": _safe_float(df["f2"].max() if len(df) else 0.0),
        "min_f2": _safe_float(df["f2"].min() if len(df) else 0.0),
        "mean_recall": _safe_float(df["recall"].mean() if len(df) else 0.0),
        "mean_precision": _safe_float(df["precision"].mean() if len(df) else 0.0),
        "mean_accuracy": _safe_float(df["accuracy"].mean() if len(df) else 0.0),
        "mean_batch_seconds": _safe_float(df.get("batch_seconds", pd.Series(dtype=float)).mean() if len(df) else 0.0),
    }

    if len(df):
        best = df.loc[df["f2"].idxmax()]
        worst = df.loc[df["f2"].idxmin()]
        summary["best_batch_id"] = int(best["batch_id"])
        summary["best_batch_f2"] = _safe_float(best["f2"])
        summary["worst_batch_id"] = int(worst["batch_id"])
        summary["worst_batch_f2"] = _safe_float(worst["f2"])
    else:
        summary["best_batch_id"] = 0
        summary["best_batch_f2"] = 0.0
        summary["worst_batch_id"] = 0
        summary["worst_batch_f2"] = 0.0

    return summary


def _plot_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df["batch_id"], df["f2"], label="F2", linewidth=2)
    plt.plot(df["batch_id"], df["recall"], label="Recall", linewidth=2)
    plt.plot(df["batch_id"], df["precision"], label="Precision", linewidth=2)
    plt.ylim(0, 1.05)
    plt.xlabel("Incremental Batch")
    plt.ylabel("Score")
    plt.title("iCNN-LSTM+ Incremental Performance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_over_batches.png", dpi=180)
    plt.close()

    if "batch_seconds" in df.columns:
        plt.figure(figsize=(10, 4))
        plt.bar(df["batch_id"], df["batch_seconds"], color="#2f7ed8")
        plt.xlabel("Incremental Batch")
        plt.ylabel("Seconds")
        plt.title("Per-Batch Update + Validation Time")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "batch_runtime_seconds.png", dpi=180)
        plt.close()


def _write_markdown(summary: dict, run: dict, output_dir: Path) -> None:
    init = run.get("initial_metrics", {})
    lines = [
        "# iCNN-LSTM+ Incremental Learning Report",
        "",
        "## Executive Summary",
        f"- Incremental batches evaluated: {summary['num_incremental_batches']}",
        f"- Mean F2 score: {summary['mean_f2']:.4f}",
        f"- Mean recall: {summary['mean_recall']:.4f}",
        f"- Mean precision: {summary['mean_precision']:.4f}",
        f"- Mean accuracy: {summary['mean_accuracy']:.4f}",
        f"- Best batch (F2): Batch {summary['best_batch_id']} ({summary['best_batch_f2']:.4f})",
        f"- Worst batch (F2): Batch {summary['worst_batch_id']} ({summary['worst_batch_f2']:.4f})",
        f"- Initial training time (s): {summary['initial_train_seconds']:.2f}",
        f"- Total run time (s): {summary['total_runtime_seconds']:.2f}",
        "",
        "## Initial Model Metrics (before incremental updates)",
        f"- Accuracy: {_safe_float(init.get('accuracy', 0.0)):.4f}",
        f"- Precision: {_safe_float(init.get('precision', 0.0)):.4f}",
        f"- Recall: {_safe_float(init.get('recall', 0.0)):.4f}",
        f"- F2: {_safe_float(init.get('f2', 0.0)):.4f}",
        "",
        "## Output Artifacts",
        "- metrics_by_batch.csv",
        "- run_summary.json",
        "- metrics_over_batches.png",
        "- batch_runtime_seconds.png",
        "- dashboard.html",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard_html(df: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    rows = "\n".join(
        f"<tr><td>{int(r.batch_id)}</td><td>{r.f2:.4f}</td><td>{r.recall:.4f}</td><td>{r.precision:.4f}</td><td>{r.accuracy:.4f}</td><td>{getattr(r, 'batch_seconds', 0.0):.2f}</td></tr>"
        for r in df.itertuples(index=False)
    )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>iCNN-LSTM+ Dashboard</title>
  <style>
    body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 24px; background: #f7fafc; color: #1f2937; }}
    h1 {{ margin: 0 0 8px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .card {{ background: #ffffff; border: 1px solid #dbe3ea; border-radius: 12px; padding: 12px; }}
    .k {{ font-size: 12px; color: #6b7280; }}
    .v {{ font-size: 22px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dbe3ea; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ width: 100%; max-width: 1000px; border: 1px solid #dbe3ea; border-radius: 10px; margin-bottom: 14px; background: #fff; }}
  </style>
</head>
<body>
  <h1>iCNN-LSTM+ Incremental Dashboard</h1>
  <div class=\"grid\">
    <div class=\"card\"><div class=\"k\">Mean F2</div><div class=\"v\">{summary['mean_f2']:.4f}</div></div>
    <div class=\"card\"><div class=\"k\">Mean Recall</div><div class=\"v\">{summary['mean_recall']:.4f}</div></div>
    <div class=\"card\"><div class=\"k\">Best Batch F2</div><div class=\"v\">{summary['best_batch_f2']:.4f}</div></div>
    <div class=\"card\"><div class=\"k\">Total Runtime (s)</div><div class=\"v\">{summary['total_runtime_seconds']:.1f}</div></div>
  </div>

  <h2>Performance Curves</h2>
  <img src=\"metrics_over_batches.png\" alt=\"Metrics over batches\" />
  <img src=\"batch_runtime_seconds.png\" alt=\"Runtime per batch\" />

  <h2>Per-Batch Metrics</h2>
  <table>
    <thead>
      <tr><th>Batch</th><th>F2</th><th>Recall</th><th>Precision</th><th>Accuracy</th><th>Seconds</th></tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""
    (output_dir / "dashboard.html").write_text(html, encoding="utf-8")


def generate_report(metrics_json_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    history = run.get("history", [])
    df = pd.DataFrame(history)

    metrics_csv_path = output_dir / "metrics_by_batch.csv"
    df.to_csv(metrics_csv_path, index=False)

    summary = _build_summary(df, run)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_metrics(df, output_dir)
    _write_markdown(summary, run, output_dir)
    _write_dashboard_html(df, summary, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report/dashboard from incremental metrics JSON")
    parser.add_argument("--metrics-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_report(Path(args.metrics_json), Path(args.output_dir))
    print(f"Report artifacts generated in: {args.output_dir}")


if __name__ == "__main__":
    main()
