# Experimental FinGPT multidimensional extraction

This mode extracts text features only. It does not use FinGPT Forecaster, predict stock prices,
modify the Transformer, add clustering, assess priced_in, or perform fine-tuning.

## Commands

The default remains the validated sentiment mode:

~~~bash
python run_fingpt_smoke_test.py \
  --model-profile sentiment-llama2-13b --quantization 4bit \
  --text "NVIDIA reported strong revenue growth."
~~~

Explicit --analysis-mode sentiment has the same behavior.
The experimental mode requires the MT profile and batch size 1:

~~~bash
python run_fingpt_smoke_test.py \
  --analysis-mode multidimensional \
  --model-profile mt-llama2-7b \
  --quantization 4bit \
  --text "NVIDIA reported strong demand for AI chips while warning that export restrictions may affect China sales."
~~~

These are future real-inference commands; no pretrained model was loaded during implementation.
Omitting --text uses the existing CSV sample inputs. News currently uses the existing title input;
earnings_call and ten_k use the existing title plus content input. No article fetching is added here.

## Model pairing

All pairs are defined in config.py; fingpt_sentiment.py re-exports them for existing callers.

- sentiment-llama2-13b: NousResearch/Llama-2-13b-hf + FinGPT/fingpt-sentiment_llama2-13b_lora (unchanged).
- mt-llama2-7b: meta-llama/Llama-2-7b-hf + FinGPT/fingpt-mt_llama2-7b_lora.
- sentiment-chatglm2-6b remains available for sentiment.

During the original implementation-stage offline check, no local MT adapter_config.json was found.
Only small public text metadata/code was consulted online on 2026-09-09:
- https://huggingface.co/FinGPT/fingpt-mt_llama2-7b_lora/raw/main/adapter_config.json
  reports base_model_name_or_path = base_models/Llama-2-7b-hf (a training-local path).
- https://raw.githubusercontent.com/AI4Finance-Foundation/FinGPT/master/fingpt/FinGPT_Benchmark/utils.py
  maps that local llama2 path to meta-llama/Llama-2-7b-hf for remote use.
- https://huggingface.co/FinGPT/fingpt-mt_llama2-7b_lora
  describes Llama2-7B multi-task training (sentiment, relation extraction, headlines, NER).

The previous NousResearch 7B choice was replaced with that explicit official remote mapping;
the successfully used 13B pair was not changed. Subsequent independent probes loaded the paired MT adapter,
but arbitrary JSON schemas remained unreliable. That evidence does not validate this experimental 15-field extractor for pipeline use.

## Schema and validation

All 15 keys are required; unknown keys (including priced_in) and duplicate JSON keys are errors.

| Fields | Allowed values |
|---|---|
| sentiment, growth_outlook, financial_strength, fundamental_impact, short_term_impact, long_term_impact | integers -2 through 2 |
| risk, uncertainty, price_narrative, relevance | integers 0 through 2 |
| event_type | earnings, guidance, product, demand, supply_chain, regulation, competition, management, partnership, market_price, macroeconomic, litigation, other |
| positive_factors, potential_concerns, evidence | lists of at most 4 nonempty strings, each at most 240 characters |
| summary | string, at most 600 characters and two sentences |

The exact schema/example and field-specific instructions live in fingpt_multidimensional.py.
Booleans, floats (including 1.0), numeric strings, NaN and Infinity are rejected.
Evidence must occur verbatim in the original input, ignoring case. It is not checked against the prompt instructions.
Summary sentence counting is punctuation-based; abbreviation-heavy summaries can require review.
Prompts require neutral values when information is unsupported and do not authorize invented facts.
Schema/evidence checks cannot establish that the model's interpretation is semantically correct.

The source-specific instructions cover direct_text, news, earnings_call and ten_k, using one common schema.
The target company and ticker come from config.py.

## Generation and failures

- Existing NF4 4-bit, BF16 compute and device_map=auto loading are reused without changes.
- Greedy generation: do_sample=False, num_beams=1, no temperature sampling.
- At most 1,024 new tokens; prompt+input at most 3,072 tokens and within the model context limit.
- Overlong input fails explicitly instead of silently truncating; this smoke mode does not split documents.
- JSON can be recovered from Markdown fences or surrounding prose.
- The first complete object is validated. Invalid fields are never replaced with zeros.
- Failed rows retain full fingpt_raw_output and a specific fingpt_error; all 15 extraction fields are null.
- Other rows continue; any failed row makes the execution exit nonzero.
- Tracebacks appear in the terminal and Markdown report.

Greedy decoding reduces sampling variability but is not a promise of bitwise reproducibility across GPU/software versions.

## Output

Existing timestamped archives and fixed latest Markdown/HTML paths are retained.
model_config JSON includes analysis_mode.
CSV contains the 15 extraction fields plus source/date/full text/model/timing/error metadata.
List fields are JSON strings; failed null fields are empty CSV cells (CSV has no native null type).
Direct-text terminal output and Markdown JSON examples represent failed values as JSON null.

Multidimensional HTML contains dimension means, all event counts, per-row scores, factors,
concerns, summary, evidence and expandable raw outputs. Error rows are orange and excluded from means.
Text previews are limited to 240 characters; original CSV text is unchanged.
The report is a single offline UTF-8 HTML file with embedded CSS, no CDN or JavaScript.
It explicitly labels scores as experimental model-generated discrete labels, not calibrated probabilities.

The original sentiment HTML branch is preserved. Production pipeline/Transformer integration of these new
features is deliberately deferred until the extraction quality and aggregation rules have been validated.

## Tests

~~~bash
python -m py_compile config.py fingpt_sentiment.py fingpt_multidimensional.py run_fingpt_smoke_test.py fingpt_report.py test_fingpt_multidimensional.py
python -m unittest -v
python run_fingpt_smoke_test.py --help
git diff --check
~~~

New tests use fake tensors and mocked model loading; no pretrained weights or network retrieval.
Existing pipeline tests also use mock FinGPT and a tiny synthetic CPU Transformer test.
No package installs or upgrades are required by this addition.

Next research step: after manual review, evaluate a small labeled sample from each source for JSON validity,
evidence fidelity, dimension consistency and reviewer agreement. The MT adapter's published tasks do not
establish competence on this custom schema; passing mock tests only verifies the implementation.
