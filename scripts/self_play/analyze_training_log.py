#!/usr/bin/env python3
"""Parse veRL-style training logs and render a lightweight HTML dashboard.

This script is designed for raw text logs that contain lines like:

    (TaskRunner pid=10338) step:5 - actor/kl_loss:0.0285 - ...

It extracts per-step scalar metrics, summarizes the run, exports CSV/JSON,
and generates a self-contained HTML report with inline SVG charts.

Usage:
    python3 scripts/self_play/analyze_training_log.py \
        outputs/training_20260319.txt
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


STEP_RE = re.compile(r"^\(TaskRunner pid=\d+\) step:(\d+)\s+-\s+(.*)$")
PROGRESS_RE = re.compile(
    r"^Training Progress:\s+([0-9]+)%\|.*?\|\s+([0-9]+)/([0-9]+)\s+\[(.*?)<(.*?),\s+([0-9.]+)s/it\]"
)


PREFERRED_METRICS = [
    "critic/score/mean",
    "critic/returns/mean",
    "val-core/medec_selfplay/reward/mean@1",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/ppo_kl",
    "actor/entropy",
    "actor/grad_norm",
    "actor/pg_clipfrac",
    "response_length/mean",
    "prompt_length/mean",
    "global_seqlen/mean",
    "perf/throughput",
    "timing_s/step",
    "perf/mfu/actor",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "perf/cpu_memory_used_gb",
    "response_length/clip_ratio",
]


def parse_number(text: str):
    text = text.strip()
    lowered = text.lower()
    if lowered in {"nan", "+nan", "-nan"}:
        return math.nan
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def split_metric_segment(segment: str):
    if ":" not in segment:
        return None, None
    key, value = segment.rsplit(":", 1)
    return key.strip(), parse_number(value)


def parse_log(log_path: Path):
    steps: dict[int, dict] = {}
    progress_rows = []
    warnings = []

    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        step_match = STEP_RE.match(line)
        if step_match:
            step = int(step_match.group(1))
            metrics_blob = step_match.group(2)
            row = steps.setdefault(step, {"step": step})
            for part in metrics_blob.split(" - "):
                key, value = split_metric_segment(part)
                if key is None:
                    continue
                row[key] = value
            continue

        progress_match = PROGRESS_RE.match(line)
        if progress_match:
            percent, done, total, elapsed, remaining, sec_per_it = progress_match.groups()
            progress_rows.append(
                {
                    "percent": int(percent),
                    "done": int(done),
                    "total": int(total),
                    "elapsed": elapsed,
                    "remaining": remaining,
                    "seconds_per_iteration": float(sec_per_it),
                }
            )
            continue

        if "WARNING:" in line:
            warnings.append(line)

    ordered_steps = [steps[idx] for idx in sorted(steps)]
    return ordered_steps, progress_rows, warnings


def numeric_metric_names(rows: list[dict]) -> list[str]:
    names = set()
    for row in rows:
        for key, value in row.items():
            if key == "step":
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                names.add(key)
    return sorted(names)


def metric_series(rows: list[dict], metric: str) -> list[tuple[int, float]]:
    series = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and math.isnan(value):
                continue
            series.append((int(row["step"]), float(value)))
    return series


def summarize_metric(rows: list[dict], metric: str):
    series = metric_series(rows, metric)
    if not series:
        return None
    xs, ys = zip(*series)
    return {
        "count": len(series),
        "first_step": xs[0],
        "last_step": xs[-1],
        "first": ys[0],
        "last": ys[-1],
        "min": min(ys),
        "max": max(ys),
        "mean": statistics.fmean(ys),
        "delta": ys[-1] - ys[0],
    }


def summarize_run(rows: list[dict], progress_rows: list[dict], warnings: list[str]):
    metric_names = numeric_metric_names(rows)
    metric_summaries = {
        name: summarize_metric(rows, name)
        for name in metric_names
    }
    metric_summaries = {k: v for k, v in metric_summaries.items() if v is not None}

    last_row = rows[-1] if rows else {}
    validation_steps = [
        int(row["step"])
        for row in rows
        if "val-core/medec_selfplay/reward/mean@1" in row
    ]

    warning_counts = Counter()
    for warning in warnings:
        if "Standalone LLM call failed" in warning:
            warning_counts["Standalone LLM call failed"] += 1
        elif "Entity extraction: LLM returned empty" in warning:
            warning_counts["Entity extraction returned empty"] += 1
        else:
            warning_counts["Other warnings"] += 1

    return {
        "log_file": str(last_row.get("_log_file", "")),
        "num_steps": len(rows),
        "first_step": rows[0]["step"] if rows else None,
        "last_step": rows[-1]["step"] if rows else None,
        "validation_steps": validation_steps,
        "warning_count": len(warnings),
        "warning_breakdown": dict(warning_counts),
        "latest_metrics": {
            key: value
            for key, value in last_row.items()
            if key not in {"step", "_log_file"}
        },
        "progress_last": progress_rows[-1] if progress_rows else None,
        "metric_summaries": metric_summaries,
    }


def write_csv(rows: list[dict], output_path: Path):
    fieldnames = ["step"] + numeric_metric_names(rows)
    for row in rows:
        for key in row:
            if key not in fieldnames and key != "_log_file":
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def format_number(value):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if not isinstance(value, float):
        return str(value)
    if math.isnan(value):
        return "nan"
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:,.1f}"
    if abs_value >= 10:
        return f"{value:.2f}"
    if abs_value >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def choose_metrics(rows: list[dict]) -> list[str]:
    available = set(numeric_metric_names(rows))
    chosen = [metric for metric in PREFERRED_METRICS if metric in available]
    if len(chosen) >= 12:
        return chosen
    for metric in sorted(available):
        if metric not in chosen:
            chosen.append(metric)
        if len(chosen) >= 12:
            break
    return chosen


def svg_line_chart(rows: list[dict], metric: str, width: int = 360, height: int = 180) -> str:
    series = metric_series(rows, metric)
    if not series:
        return '<div class="empty-chart">No data</div>'

    pad_left = 44
    pad_right = 14
    pad_top = 18
    pad_bottom = 30
    inner_width = width - pad_left - pad_right
    inner_height = height - pad_top - pad_bottom

    xs, ys = zip(*series)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_y == max_y:
        spread = abs(min_y) * 0.05 or 1.0
        min_y -= spread
        max_y += spread
    y_pad = (max_y - min_y) * 0.08
    min_y -= y_pad
    max_y += y_pad

    def sx(value: float) -> float:
        if max_x == min_x:
            return pad_left + inner_width / 2
        return pad_left + (value - min_x) / (max_x - min_x) * inner_width

    def sy(value: float) -> float:
        return pad_top + inner_height - (value - min_y) / (max_y - min_y) * inner_height

    points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in series)
    last_x, last_y = series[-1]
    first_x, first_y = series[0]

    y_ticks = []
    for idx in range(3):
        tick_value = min_y + (max_y - min_y) * idx / 2
        py = sy(tick_value)
        y_ticks.append(
            f'<line x1="{pad_left}" y1="{py:.1f}" x2="{width - pad_right}" y2="{py:.1f}" class="grid" />'
            f'<text x="{pad_left - 8}" y="{py + 4:.1f}" class="axis-label" text-anchor="end">{html.escape(format_number(tick_value))}</text>'
        )

    x_labels = (
        f'<text x="{pad_left}" y="{height - 8}" class="axis-label" text-anchor="start">step {first_x}</text>'
        f'<text x="{width - pad_right}" y="{height - 8}" class="axis-label" text-anchor="end">step {last_x}</text>'
    )

    return f"""
    <svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="{html.escape(metric)}">
      <rect x="0" y="0" width="{width}" height="{height}" rx="12" class="chart-bg"></rect>
      {''.join(y_ticks)}
      <line x1="{pad_left}" y1="{pad_top + inner_height:.1f}" x2="{width - pad_right}" y2="{pad_top + inner_height:.1f}" class="axis" />
      <polyline fill="none" stroke="var(--accent)" stroke-width="2.5" points="{points}" />
      <circle cx="{sx(first_x):.1f}" cy="{sy(first_y):.1f}" r="3.2" class="point point-first" />
      <circle cx="{sx(last_x):.1f}" cy="{sy(last_y):.1f}" r="4.0" class="point point-last" />
      {x_labels}
    </svg>
    """


def metric_card(rows: list[dict], metric: str) -> str:
    summary = summarize_metric(rows, metric)
    if summary is None:
        return ""

    delta = summary["delta"]
    delta_prefix = "+" if delta > 0 else ""
    return f"""
    <section class="panel">
      <div class="panel-header">
        <h3>{html.escape(metric)}</h3>
        <div class="metric-meta">
          <span>last {html.escape(format_number(summary["last"]))}</span>
          <span>Δ {delta_prefix}{html.escape(format_number(delta))}</span>
        </div>
      </div>
      {svg_line_chart(rows, metric)}
    </section>
    """


def summary_cards(summary: dict) -> str:
    latest = summary["latest_metrics"]
    cards = [
        ("Steps Parsed", summary["num_steps"]),
        ("Last Step", summary["last_step"]),
        ("Latest Score", latest.get("critic/score/mean")),
        ("Latest Val Reward", latest.get("val-core/medec_selfplay/reward/mean@1")),
        ("Latest KL Loss", latest.get("actor/kl_loss")),
        ("Latest Throughput", latest.get("perf/throughput")),
        ("Max GPU Memory", latest.get("perf/max_memory_allocated_gb")),
        ("Warnings", summary["warning_count"]),
    ]
    rendered = []
    for label, value in cards:
        rendered.append(
            f"""
            <div class="summary-card">
              <div class="summary-label">{html.escape(label)}</div>
              <div class="summary-value">{html.escape(format_number(value))}</div>
            </div>
            """
        )
    return "".join(rendered)


def warning_list(warnings: list[str]) -> str:
    if not warnings:
        return "<p class=\"muted\">No warnings found in the log.</p>"
    items = "\n".join(f"<li>{html.escape(line)}</li>" for line in warnings[:20])
    more = ""
    if len(warnings) > 20:
        more = f"<p class=\"muted\">Showing first 20 of {len(warnings)} warnings.</p>"
    return f"<ul class=\"warnings\">{items}</ul>{more}"


def validation_table(rows: list[dict]) -> str:
    validation_rows = [
        row for row in rows if "val-core/medec_selfplay/reward/mean@1" in row
    ]
    if not validation_rows:
        return "<p class=\"muted\">No validation checkpoints found.</p>"

    body = []
    for row in validation_rows:
        body.append(
            "<tr>"
            f"<td>{row['step']}</td>"
            f"<td>{html.escape(format_number(row.get('val-core/medec_selfplay/reward/mean@1')))}</td>"
            f"<td>{html.escape(format_number(row.get('val-aux/medec_selfplay/score/mean@1')))}</td>"
            f"<td>{html.escape(format_number(row.get('timing_s/testing')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Step</th><th>Val Reward</th><th>Val Score</th><th>Testing Time (s)</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def build_html(log_path: Path, rows: list[dict], progress_rows: list[dict], warnings: list[str], summary: dict) -> str:
    chosen_metrics = choose_metrics(rows)
    panels = "".join(metric_card(rows, metric) for metric in chosen_metrics)
    progress_note = ""
    if progress_rows:
        last_progress = progress_rows[-1]
        progress_note = (
            f"Last tqdm progress snapshot: {last_progress['done']}/{last_progress['total']} "
            f"({last_progress['percent']}%), {format_number(last_progress['seconds_per_iteration'])} s/it."
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Training Dashboard</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf1;
      --ink: #1d2a30;
      --muted: #62727b;
      --grid: #d7ddd5;
      --accent: #0f766e;
      --accent-soft: #d8efe9;
      --warn: #b45309;
      --border: #d8cfbf;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      background:
        radial-gradient(circle at top left, #fff7de 0%, rgba(255,247,222,0.2) 32%, transparent 55%),
        linear-gradient(180deg, #f6f0e7 0%, #efe7d9 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    .hero {{
      padding: 24px 28px;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: rgba(255, 250, 241, 0.92);
      box-shadow: 0 12px 28px rgba(72, 57, 35, 0.08);
    }}
    h1, h2, h3, p {{
      margin: 0;
    }}
    h1 {{
      font-size: clamp(2rem, 4vw, 3.6rem);
      line-height: 1;
      letter-spacing: -0.04em;
      margin-bottom: 10px;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 1.02rem;
      max-width: 980px;
      line-height: 1.5;
    }}
    .meta {{
      margin-top: 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .chip {{
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.92rem;
      color: var(--muted);
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 14px;
      margin-top: 24px;
    }}
    .summary-card {{
      background: linear-gradient(180deg, #fffdf8 0%, #f7f2e8 100%);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      min-height: 96px;
    }}
    .summary-label {{
      color: var(--muted);
      font-size: 0.88rem;
      margin-bottom: 10px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .summary-value {{
      font-size: 1.7rem;
      line-height: 1.05;
      font-weight: 700;
    }}
    .section-title {{
      margin: 30px 0 16px;
      font-size: 1.25rem;
    }}
    .panel-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    .panel {{
      background: rgba(255, 250, 241, 0.95);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 16px;
      box-shadow: 0 10px 22px rgba(60, 44, 25, 0.06);
    }}
    .panel-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .panel-header h3 {{
      font-size: 1rem;
      line-height: 1.2;
      word-break: break-word;
    }}
    .metric-meta {{
      display: flex;
      gap: 10px;
      color: var(--muted);
      font-size: 0.84rem;
      white-space: nowrap;
    }}
    .chart-svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .chart-bg {{
      fill: var(--panel);
      stroke: rgba(0, 0, 0, 0);
    }}
    .grid {{
      stroke: var(--grid);
      stroke-width: 1;
      stroke-dasharray: 4 4;
    }}
    .axis {{
      stroke: #8999a1;
      stroke-width: 1;
    }}
    .axis-label {{
      fill: var(--muted);
      font-size: 11px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .point {{
      fill: var(--accent);
      stroke: white;
      stroke-width: 1.5;
    }}
    .point-first {{
      fill: #6b7280;
    }}
    .point-last {{
      fill: #b91c1c;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-top: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.9rem;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    .warnings {{
      margin: 0;
      padding-left: 18px;
      color: var(--warn);
      line-height: 1.45;
    }}
    .muted {{
      color: var(--muted);
      line-height: 1.5;
    }}
    @media (max-width: 960px) {{
      .two-col {{
        grid-template-columns: 1fr;
      }}
      .panel-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Training Log Dashboard</h1>
      <p class="subtitle">
        Parsed raw scalar output from <code>{html.escape(str(log_path))}</code> into a veRL-style dashboard.
        This is intended to mirror the GRPO engineering-post workflow: inspect reward/score quality,
        policy stability, token growth, and system health from text logs without requiring W&B.
      </p>
      <div class="meta">
        <span class="chip">steps {summary["first_step"]} to {summary["last_step"]}</span>
        <span class="chip">validation checkpoints {len(summary["validation_steps"])}</span>
        <span class="chip">warnings {summary["warning_count"]}</span>
        <span class="chip">{html.escape(progress_note or "No tqdm progress lines parsed.")}</span>
      </div>
      <div class="summary-grid">
        {summary_cards(summary)}
      </div>
    </section>

    <h2 class="section-title">Core Metrics</h2>
    <div class="panel-grid">
      {panels}
    </div>

    <div class="two-col">
      <section class="panel">
        <div class="panel-header">
          <h3>Validation Checkpoints</h3>
        </div>
        {validation_table(rows)}
      </section>

      <section class="panel">
        <div class="panel-header">
          <h3>Warnings</h3>
        </div>
        {warning_list(warnings)}
      </section>
    </div>
  </div>
</body>
</html>
"""


def print_console_summary(summary: dict):
    latest = summary["latest_metrics"]
    print(f"Parsed {summary['num_steps']} step lines from step {summary['first_step']} to {summary['last_step']}.")
    if summary["validation_steps"]:
        print(f"Validation checkpoints at steps: {', '.join(str(s) for s in summary['validation_steps'])}")
    print(f"Warnings: {summary['warning_count']}")
    interesting = [
        "critic/score/mean",
        "val-core/medec_selfplay/reward/mean@1",
        "actor/kl_loss",
        "actor/grad_norm",
        "response_length/mean",
        "perf/throughput",
        "perf/max_memory_allocated_gb",
    ]
    for metric in interesting:
        if metric in latest:
            print(f"  {metric}: {format_number(latest[metric])}")


def main():
    parser = argparse.ArgumentParser(description="Parse and analyze veRL-style training logs.")
    parser.add_argument("log_file", type=Path, help="Path to the raw training log text file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated artifacts. Defaults to <log_dir>/analysis_<stem>/",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix for generated files. Defaults to the log stem.",
    )
    args = parser.parse_args()

    log_path = args.log_file.resolve()
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = log_path.parent / f"analysis_{log_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or log_path.stem
    rows, progress_rows, warnings = parse_log(log_path)
    if not rows:
        raise SystemExit("No '(TaskRunner pid=...) step:' lines found in the log.")

    for row in rows:
        row["_log_file"] = str(log_path)

    summary = summarize_run(rows, progress_rows, warnings)
    summary["log_file"] = str(log_path)

    csv_path = output_dir / f"{prefix}_metrics.csv"
    json_path = output_dir / f"{prefix}_summary.json"
    html_path = output_dir / f"{prefix}_dashboard.html"

    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    html_path.write_text(build_html(log_path, rows, progress_rows, warnings, summary), encoding="utf-8")

    print_console_summary(summary)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
