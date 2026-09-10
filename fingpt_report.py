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
    if report.get("analysis_mode", "sentiment") == "multidimensional":
        return write_multidimensional_html_report(report, outputs, path)
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
<p class="subtitle">Sentiment results &middot; self-contained offline report</p>
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


def write_multidimensional_html_report(report, outputs, path):
    """Render only validated rows in aggregates; keep error rows visible in the table."""
    import json
    from fingpt_multidimensional import NUMERIC_RANGES, EVENT_TYPES, SCHEMA_FIELDS, EXPERIMENT_NOTICE
    records = outputs.to_dict(orient="records")

    def esc(value):
        if isinstance(value, list):
            value = json.dumps(value, ensure_ascii=False)
        return escape(display_text(value), quote=True)

    def valid(row):
        return (
            not display_text(row.get("fingpt_error")).strip()
            and bool(display_text(row.get("fingpt_raw_output")).strip())
            and row.get("event_type") in EVENT_TYPES
            and all(display_text(row.get(field)) != "" for field in NUMERIC_RANGES)
        )

    successful = [row for row in records if valid(row)]
    cards = []
    for field, bounds in NUMERIC_RANGES.items():
        values = [float(row[field]) for row in successful]
        mean = f"{sum(values)/len(values):.3f}" if values else "N/A"
        cards.append(f'<div class="card"><span>{field} ({bounds[0]}..{bounds[1]})</span><strong>{mean}</strong></div>')
    events = []
    for event in EVENT_TYPES:
        count = sum(row["event_type"] == event for row in successful)
        percent = count/len(successful)*100 if successful else 0
        events.append(f'<div class="event"><span>{event}</span><div class="track">'
                      f'<div class="bar" style="width:{percent:.3f}%"></div></div><span>{count}</span></div>')
    columns = ["date", "source_type", "text_preview", *SCHEMA_FIELDS,
               "fingpt_raw_output", "fingpt_inference_seconds", "fingpt_error"]
    table = []
    for row in records:
        preview = " ".join(display_text(row.get("text")).split())
        preview = preview if len(preview) <= 240 else preview[:237] + "..."
        cells = []
        for column in columns:
            value = preview if column == "text_preview" else row.get(column)
            content = esc(value)
            if column in NUMERIC_RANGES and content:
                number = float(value)
                if column in {"risk", "uncertainty", "price_narrative"}:
                    tone = "concern" if number > 0 else "neutral"
                elif column == "relevance":
                    tone = "neutral"
                else:
                    tone = "positive" if number > 0 else "negative" if number < 0 else "neutral"
                content = f'<span class="{tone}">{content}</span>'
            if column == "fingpt_raw_output":
                content = f"<details><summary>Raw output</summary><pre>{content}</pre></details>"
            cells.append(f'<td class="{column}">{content or "null"}</td>')
        table.append(f'<tr class="{"success" if valid(row) else "error"}">{"".join(cells)}</tr>')
    if not table:
        table.append(f'<tr><td colspan="{len(columns)}">No rows available.</td></tr>')
    metadata = "".join(f"<dt>{key}</dt><dd>{esc(report.get(key))}</dd>" for key in
                       ["run_id", "started_at", "finished_at", "model_profile", "model_name", "quantization", "status"])
    error = esc(report.get("fingpt_error"))
    traces = esc(report.get("traceback"))
    times = [float(row["fingpt_inference_seconds"]) for row in successful
             if display_text(row.get("fingpt_inference_seconds"))]
    average = f"{sum(times)/len(times):.4f} s" if times else "N/A"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FinGPT experimental multidimensional extraction</title>
<style>
body {{margin:0;background:#f4f6f8;color:#182332;font:15px/1.5 system-ui,sans-serif}}
main {{padding:28px;max-width:1700px;margin:auto}}
section {{background:white;border:1px solid #dce2e8;border-radius:10px;padding:20px;margin:20px 0}}
dl {{display:grid;grid-template-columns:150px 1fr;gap:8px}} dd {{margin:0;overflow-wrap:anywhere}}
.cards {{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.card {{padding:16px;background:white;border:1px solid #dce2e8;border-radius:8px}}
.card span {{display:block;color:#536172}} .card strong {{font-size:25px}}
.note {{color:#536172}} .event {{display:grid;grid-template-columns:150px 1fr 40px;gap:15px;margin:8px 0}}
.track {{height:15px;background:#edf0f3}} .bar {{height:100%;background:#4466a0}}
.table-scroll {{overflow:auto}} table {{border-collapse:collapse;font-size:13px}}
th,td {{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #dce2e8}}
th {{background:#edf1f5;white-space:nowrap}} td {{min-width:90px;max-width:400px;overflow-wrap:anywhere}}
.text_preview,.summary,.evidence,.positive_factors,.potential_concerns {{min-width:230px}}
.positive {{color:#166534}} .negative {{color:#991b1b}} .neutral {{color:#526071}}
.concern,.fingpt_error {{color:#9a3412}} .error {{background:#fff0df}}
pre {{white-space:pre-wrap;overflow-wrap:anywhere;min-width:240px;max-width:450px}}
</style></head><body><main>
<h1>FinGPT multidimensional results</h1>
<p class="note">{esc(EXPERIMENT_NOTICE)}</p>
<section><dl>{metadata}</dl>
<p>Total: {len(records)} &middot; Successful: {len(successful)} &middot;
Failed: {len(records)-len(successful)} &middot; Average successful inference: {average}</p>
<p class="fingpt_error">{error}</p>
<details><summary>Traceback</summary><pre>{traces}</pre></details></section>
<h2>Mean dimension values</h2>
<p class="note">Only successfully validated rows are included. No results: N/A (not zero).</p>
<div class="cards">{"".join(cards)}</div>
<section><h2>Event counts</h2>{"".join(events)}</section>
<section><h2>Per-row results</h2>
<p class="note">CSV retains full text. HTML text_preview is limited to 240 characters.
Invalid extraction fields remain null; error rows are excluded from averages and counts.</p>
<div class="table-scroll"><table><thead><tr>
{"".join(f'<th scope="col">{column}</th>' for column in columns)}
</tr></thead><tbody>{"".join(table)}</tbody></table></div></section>
</main></body></html>"""
    Path(path).write_text(document, encoding="utf-8")
