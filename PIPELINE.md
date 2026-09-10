# Full pipeline for RTX 4090

The smoke test remains unchanged. The research pipeline scaffold entry point is `run_pipeline.py`.
It has offline synthetic coverage but has not been validated on the complete real corpus.
No model, document, or price download happens from imports, --help, or the offline tests.

## Stages

raw CSVs -> prepare -> sentiment -> features -> dataset -> train/evaluate

- audit: inventory raw inputs and their SHA-256 hashes.
- prepare: preserve document bodies, assign stable IDs, check quality, optionally retrieve HTML/PDF bodies.
- sentiment: tokenize and split full supplied bodies, run FinGPT, save chunk and document checkpoints.
- features: aggregate documents independently by source and by day/week; align monthly Trends.
- prices: import local adjusted daily closes or explicitly download using yfinance.
- dataset: join features to future-price labels without using future prices as features.
- train: chronological Transformer training, validation-based checkpoint selection, held-out evaluation.
- all: run all stages. Stops on errors unless --allow-partial is supplied.

Existing raw CSVs, historical archives, the smoke-test CLI are unchanged. Dataset consolidation requires an explicit `--apply` acknowledgement.

## Start with a local audit

From WSL:

~~~bash
conda activate fingpt4090
cd /path/to/us-stock-transformer-4090
python run_pipeline.py audit
~~~

The command prints an absolute run directory:
`data/pipeline_runs/YYYY-MM-DD_HHMMSS_identifier/`.

To inspect preparation without network or a model:

~~~bash
python run_pipeline.py prepare --run-dir /absolute/path/to/run
~~~

It writes documents.csv, preparation_errors.csv and preparation_summary.json.
Nonzero exit is expected when bodies or verified publication dates are missing.
The preparation output is an inventory of usable/excluded documents, not a claim that all bodies are complete.

## Prepare full documents

Start a new run with fetching enabled:

~~~bash
python run_pipeline.py prepare --fetch-body \
  --user-agent "YourName your-contact@example.com" \
  --date-overrides /absolute/path/publication_dates.csv
~~~

Omit --date-overrides if no reviewed dates are available yet; unverified CFO commentary remains excluded.
Build the override CSV from the document IDs in the offline preparation inventory:

~~~csv
document_id,available_date,evidence
<document ID>,2020-05-21,<source supporting actual publication date>
~~~

Use actual reviewed values, not the placeholder above.
Fiscal quarter labels are not interpreted as publication dates. No dates are inferred from the current webpage.

- Fetching is opt-in and cached under data/cache/article_bodies.
- PDF extraction requires optional pypdf; the default HTML extractor uses the standard library.
- URLs that fail, are blocked or return too little text are recorded as errors.
- No paywall/CAPTCHA bypass is attempted; manually provide a corrected content column when needed.
- Automated extraction still needs sample-based quality review, including completeness and article identity.
- Full-body retrieval does not repair a wrong source URL or historical publication date.
- Downloaded text is stored in the run/cache; raw files are not overwritten.
- --retry-errors retries failed fetch/inference caches.
- Changes to data/model settings require a new run. Existing run settings are reused on resume.
- --limit N restricts each source and explicitly marks the run as a sample.

Explicit exploratory options exist: --allow-title-only, --allow-incomplete-text,
--allow-unverified-dates. These are not enabled by default and do not establish data validity.
Title-only mode applies only when no news body exists and fetching is disabled.

## Analyze and build features

Use the directory printed by prepare for all subsequent commands:

~~~bash
RUN_DIR=/absolute/path/to/run
python run_pipeline.py sentiment --run-dir "$RUN_DIR"
python run_pipeline.py features --run-dir "$RUN_DIR"
~~~

The sentiment stage loads the configured FinGPT base model and adapter. This is a real workload;
it was NOT executed on the real corpus during implementation.

Defaults: sentiment-llama2-13b, NF4 4-bit, one chunk per inference call.
Existing smoke-test inference/loading is reused.

- Text is normalized for inference and split to fit both 1,800 characters and the exact 512-token prompt.
- All text is covered by consecutive chunks; there is no silent head-only truncation.
- Individual chunks are independently classified; this does not guarantee whole-document discourse understanding.
- Chunk labels map to -1/0/1. Unparsed/ambiguous output is an error, not neutral.
- A document is usable only if every chunk succeeded.
- Document score is the equal-weight mean of chunk scores; sign determines the document label.
  A zero mean can mean mixed positive/negative chunks, not necessarily neutral wording.
- Features weight documents equally, so long documents do not count as many separate news items.
- These are sentiment measures, not calibrated confidence or risk scores.
- Caches record outputs and errors; successful chunks can be reused after an interrupted document.
- To force a fresh run after model weights/code change, use a new run directory.

--allow-partial permits downstream stages to use successful documents only.
Errors remain recorded; exit code stays nonzero for a run containing exclusions.
A dataset with no successful sentiment documents is refused.
Error counts and missing indicators remain explicit features.

## Price data and targets

No stock-price raw CSV was present in the imported project. Supply your own:

~~~bash
python run_pipeline.py prices --run-dir "$RUN_DIR" --prices /absolute/path/NVDA_prices.csv
~~~

Required columns: date,close. Dates must be unique; closes must be finite, positive,
split-adjusted daily closes. Include at least the requested horizon beyond the final feature cutoff.
The local-price schema cannot itself verify adjustment provenance or a complete exchange calendar.

Alternatively, after separately making optional yfinance available:

~~~bash
python run_pipeline.py prices --run-dir "$RUN_DIR" --download-prices
~~~

The downloader requests adjusted closes and extra calendar days for future labels.
No dependencies are installed by the pipeline.

~~~bash
python run_pipeline.py dataset --run-dir "$RUN_DIR"
python run_pipeline.py train --run-dir "$RUN_DIR" --device cuda --epochs 20 --lookback 8
~~~

Defaults are weekly features and 5 future trading observations.
Choose --frequency daily and/or --horizon on the FIRST command for a new run.

Weekly cutoffs are Sunday; daily cutoffs are day end.
Entry price is the first trading close strictly AFTER the feature cutoff.
Exit is horizon observations after entry; label=1 for positive return, otherwise 0.
Unavailable targets stay missing and are excluded from supervised samples.
This is a close-to-close research label, not an executable trading strategy.

Monthly Trends values are made available only from the next month start.
They are carried forward rather than assigning zero to all intervening weeks.
Unavailable observations get a missing indicator. Partial months are excluded.
Historical Trends normalization is still not a point-in-time vintage dataset.

## Train/evaluate

- Lookback defaults to 8 periods; Transformer width=32, heads=4, layers=2.
- Time-ordered 70/15/15 splits, with target-horizon purging at split boundaries.
- Standardization is fit exclusively on rows in training windows.
- Best checkpoint is chosen using validation loss, never test accuracy.
- Test metrics include accuracy, confusion matrix and a training-majority baseline.
- At least 20 labeled sequences and adequate purged split sizes are required.
- Test windows may use past context from earlier splits; future rows never enter those windows.
- CPU is the training default; --device cuda is explicit. FinGPT still uses its existing CUDA loader.
- No predictive performance claim follows from passing the synthetic integration test.

## Outputs and resume

Every run contains manifest.json with settings, stage states, errors/tracebacks and artifact hashes.
Upstream reruns invalidate downstream stage status; changed artifacts require regeneration.

Key artifacts:
- inventory.json
- documents.csv, preparation_errors.csv, preparation_summary.json
- sentiment_chunks.csv, sentiment_documents.csv, sentiment_report.html
- chunk_cache/, inference_cache/, inference_config.json
- features.csv, feature_schema.json
- prices.csv, prices.metadata.json
- dataset.csv, dataset_summary.json
- transformer.pt, transformer_predictions.csv, metrics.json
- training_history.csv, training_config.json

The checkpoint includes model weights, feature ordering, scaling parameters and architecture settings.
Pipeline outputs are ignored by Git. Smoke-test CSV/Markdown/HTML outputs remain independent.

## One command (after source quality and dependencies are ready)

~~~bash
python run_pipeline.py all --fetch-body \
  --user-agent "YourName your-contact@example.com" \
  --date-overrides /absolute/path/publication_dates.csv \
  --prices /absolute/path/NVDA_prices.csv \
  --frequency weekly --horizon 5 --device cuda
~~~

## Offline validation

~~~bash
python -m py_compile pipeline_documents.py pipeline_sentiment.py pipeline_transformer.py run_pipeline.py
python run_pipeline.py --help
python -m unittest -v test_pipeline test_fingpt_report
~~~

Tests use synthetic documents/prices, mock FinGPT and a tiny CPU Transformer training pass.
They do not fetch documents, download prices, or load pretrained weights.

Optional dependency list: requirements-pipeline-optional.txt (pypdf, yfinance).
Do not reinstall Torch/CUDA or change the working FinGPT stack to run these tests.

API references:
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html
- https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
