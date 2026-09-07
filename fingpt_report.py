"""Offline HTML presentation only; multidimensional scores need future validation."""
from html import escape
from pathlib import Path
import math

import pandas as pd


HTML_REPORT_FILE = "fingpt_smoke_test_report.html"
LABELS = ("positive", "neutral", "negative")
TABLE_COLUMNS = (
    "date", "source_type", "text_preview", "sentiment_label", "sentiment_score",
    "fingpt_raw_output", "fingpt_inference_seconds", "fingpt_error",
)


def display_text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def summarize_results(outputs: pd.DataFrame) -> dict:
    records = outputs.to_dict(orient="records")
    successful = [
        row for row in records
        if not display_text(row.get("fingpt_error")).strip()
        and display_text(row.get("fingpt_raw_output")).strip()
    ]
    counts = {label: 0 for label in LABELS}
    seconds = []
    for row in successful:
        label = display_text(row.get("sentiment_label")).strip().lower()
        if label in counts:
            counts[label] += 1
        try:
            value = float(row.get("fingpt_inference_seconds"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            seconds.append(value)
    classified = sum(counts.values())
    return {
        "total": len(records),
        "success": len(successful),
        "failed": len(records) - len(successful),
        "unparsed": len(successful) - classified,
        "counts": counts,
        "percentages": {
            label: count / classified * 100 if classified else 0.0
            for label, count in counts.items()
        },
        "average_seconds": sum(seconds) / len(seconds) if seconds else None,
    }


def write_html_report(report: dict, outputs: pd.DataFrame,
                      path: str | Path = HTML_REPORT_FILE) -> None:
    """Escape all data; never modify or truncate the source dataframe."""
    summary = summarize_results(outputs)
    esc = lambda value: escape(display_text(value), quote=True)
    metadata = {
        "Run ID": report.get("run_id"),
        "Started at": report.get("started_at"),
        "Finished at": report.get("finished_at"),
        "Status": report.get("status"),
        "model_profile": report.get("model_profile"),
        "model_name": report.get("model_name"),
        "quantization": report.get("quantization"),
    }
    details = "".join(
        f"<div><dt>{esc(key)}</dt><dd>{esc(value)}</dd></div>"
        for key, value in metadata.items()
    )
    average = summary["average_seconds"]
    cards = "".join(
        f'<div class="card"><span>{title}</span><strong>{value}</strong></div>'
        for title, value in (
            ("Total rows", summary["total"]),
            ("Successful rows", summary["success"]),
            ("Failed / incomplete rows", summary["failed"]),
            ("Average inference", f"{average:.4f} s" if average is not None else "N/A"),
        )
    )
    bars = ""
    for label in LABELS:
        count = summary["counts"][label]
        percent = summary["percentages"][label]
        bars += (
            f'<div class="distribution"><span class="badge {label}">{label}</span>'
            f'<div class="track"><div class="bar {label}" style="width:{percent:.4f}%"></div></div>'
            f"<span>{count} ({percent:.1f}%)</span></div>"
        )
    table_rows = []
    for row in outputs.to_dict(orient="records"):
        preview = " ".join(display_text(row.get("text")).split())
        if len(preview) > 240:
            preview = preview[:239] + "?"
        label = display_text(row.get("sentiment_label")).strip().lower()
        error = display_text(row.get("fingpt_error")).strip()
        incomplete = not display_text(row.get("fingpt_raw_output")).strip()
        tone = "error" if error or incomplete else label if label in LABELS else "unparsed"
        cells = []
        for column in TABLE_COLUMNS:
            value = preview if column == "text_preview" else row.get(column)
            content = esc(value)
            if column == "sentiment_label":
                content = f'<span class="badge {tone}">{content or "?"}</span>'
            cells.append(f'<td class="{column}">{content}</td>')
        table_rows.append(f'<tr class="row-{tone}">{"".join(cells)}</tr>')
    if not table_rows:
        table_rows.append(f'<tr><td colspan="{len(TABLE_COLUMNS)}">No rows available.</td></tr>')
    headers = "".join(f'<th scope="col">{column}</th>' for column in TABLE_COLUMNS)
    error_block = ""
    if display_text(report.get("fingpt_error")):
        error_block = f'<section class="failure"><h2>Run error</h2><pre>{esc(report["fingpt_error"])}</pre></section>'
    if display_text(report.get("traceback")):
        error_block += f'<details><summary>Full traceback</summary><pre>{esc(report["traceback"])}</pre></details>'
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FinGPT smoke test results</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #f4f6f8; color: #182332; font: 15px/1.5 system-ui, sans-serif; }}
main {{ max-width: 1600px; padding: 32px 24px; margin: auto; }}
h1 {{ margin: 0; font-size: 30px; }} h2 {{ font-size: 20px; }}
.subtitle, .note {{ color: #536172; }}
section, details {{ background: white; border: 1px solid #dce2e8; border-radius: 10px; padding: 20px; margin: 20px 0; }}
dl {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); gap: 16px; }}
dt {{ color: #536172; font-size: 13px; }} dd {{ margin: 3px 0 0; overflow-wrap: anywhere; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 14px; }}
.card {{ background: white; border: 1px solid #dce2e8; border-radius: 10px; padding: 20px; }}
.card span {{ display: block; color: #536172; }} .card strong {{ font-size: 28px; }}
.distribution {{ display: grid; grid-template-columns: 100px 1fr 120px; gap: 14px; align-items: center; margin: 16px 0; }}
.track {{ background: #edf0f3; border-radius: 6px; height: 18px; overflow: hidden; }}
.bar {{ height: 100%; }}
.badge {{ display: inline-block; padding: 3px 9px; border-radius: 5px; font-weight: 600; }}
.positive {{ background: #dcfce7; color: #166534; }} .bar.positive {{ background: #22834a; }}
.neutral {{ background: #e5e7eb; color: #374151; }} .bar.neutral {{ background: #6b7280; }}
.negative {{ background: #fee2e2; color: #991b1b; }} .bar.negative {{ background: #c73535; }}
.error, .unparsed {{ background: #ffedd5; color: #9a3412; }}
.row-error {{ background: #fff7ed; }} .failure {{ border-left: 5px solid #c65d12; }}
.table-scroll {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 12px; border-bottom: 1px solid #dce2e8; text-align: left; vertical-align: top; }}
th {{ background: #edf1f5; white-space: nowrap; }}
td {{ overflow-wrap: anywhere; min-width: 100px; max-width: 380px; }}
.text_preview {{ min-width: 280px; }}
.fingpt_error {{ color: #9a3412; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
@media print {{ body {{ background: white; }} main {{ padding: 0; }} .table-scroll {{ overflow: visible; }} }}
</style>
</head>
<body><main>
<h1>FinGPT smoke test</h1>
<p class="subtitle">Sentiment results ? self-contained offline report</p>
<section><dl>{details}</dl></section>
<div class="cards">{cards}</div>
{error_block}
<section><h2>Sentiment distribution</h2>
<p class="note">Percentages use successfully parsed positive / neutral / negative rows only.
Unparsed successful outputs: {summary["unparsed"]}. Error rows are never counted as neutral.</p>
{bars}
</section>
<section><h2>Results</h2>
<p class="note">Successful = nonempty output with no fingpt_error. Failed / incomplete includes rows
not attempted after an earlier failure. Average uses available successful-row inference times.
Text previews are limited to 240 characters; the CSV retains full text.</p>
<div class="table-scroll"><table><thead><tr>{headers}</tr></thead>
<tbody>{"".join(table_rows)}</tbody></table></div>
</section>
<p class="note">Only sentiment is validated. Multidimensional outputs such as risk, growth_outlook
and uncertainty will be designed and validated in a later phase; no such scores are generated here.</p>
</main></body></html>
"""
    Path(path).write_text(document, encoding="utf-8")
