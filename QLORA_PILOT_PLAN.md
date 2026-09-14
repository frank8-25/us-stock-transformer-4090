# FinGPT MT 7B custom multidimensional QLoRA pilot plan

## Recommendation

For a 300–500 row pilot, tentatively start from a separately copied FinGPT/fingpt-mt_llama2-7b_lora adapter, load that copy as trainable, and continue training the single adapter against meta-llama/Llama-2-7b-hf with a low learning rate. This must still pass a real smoke-training and reload test; it is not yet proven. Never overwrite the official adapter. This retains useful financial instruction initialization while keeping rollback and side-by-side evaluation possible. It is adapter continuation, not an assumption that arbitrary adapters can be stacked.

Do not train until labels, event-group splits, licensing, memory settings, and a frozen evaluation suite are approved.

## Single-dimension training format

Annotate each source text once with the complete ten-dimension schema and evidence. Before any task expansion, split source records by time and event group into train, validation and test. Only then expand each source record into ten independent instruction examples. This guarantees that different dimensions from one text cannot cross splits. Keep the same news cluster, adjacent sections from one earnings call, and nearby passages from one 10-K in one split.

Each instruction asks exactly one dimension and the model outputs exactly one allowed label. Do not train generation of a complete ten-field JSON object. Python assembles the ten outputs into JSON or CSV after inference. A pilot of 300–500 reviewed source texts therefore yields about 3,000–5,000 custom task instructions before rehearsal examples are added.

Example task rows from one source record:

    {"instruction":"Determine the growth outlook for NVDA. Choose exactly one of negative, neutral, positive, or insufficient_information.","input":"canonical input text","output":"negative","dimension":"growth_outlook","source_sample_id":"sample-001","split":"train"}

    {"instruction":"Does the text describe a material downside risk to NVDA? Choose exactly one of yes or no.","input":"canonical input text","output":"yes","dimension":"risk_presence","source_sample_id":"sample-001","split":"train"}

insufficient_information is a first-class output label. It must remain missing with applicability=0 at the Transformer boundary and must never be converted to neutral.

## Three approaches

| Approach | Official capability | Forgetting risk | PEFT complexity | RTX 4090 24 GB | Inference and artifacts |
|---|---|---|---|---|---|
| New custom LoRA from base | Official MT behavior is not inherited and must be learned through custom or rehearsal data. | No overwrite of MT skills, but 300–500 rows may be too small to recreate them. | Lowest: quantized base, prepare_model_for_kbit_training, new LoraConfig. | Feasible as a cautious 7B QLoRA pilot with short sequences, batch 1, accumulation and checkpointing; measure before scaling. | Load base plus custom adapter. Save adapter, tokenizer/config, manifest and hashes. |
| Continue a copy of MT LoRA | Begins with the official adapter’s observed instruction behavior. | Highest direct forgetting risk if custom labels dominate. | Medium: load the adapter trainable and verify target modules, parameters and active adapter. Do not implicitly add a second adapter. | Likely feasible with cautious QLoRA settings; confirm optimizer and sequence memory in a later tiny dry run. | Load base plus the custom continuation adapter. Preserve the untouched official adapter and revision. |
| Merge MT adapter, then train a new LoRA | MT changes become part of a derived base. | Merge is hard to reverse; later training can still forget. | Highest: merge against a compatible non-quantized base, verify numerics, save a derived base, then quantize and attach a new adapter. Merging directly into 4-bit weights must not be assumed safe. | New-LoRA training may fit, but merge/export needs more CPU/GPU RAM and disk. | Load derived base plus new adapter. Provenance, hashes and licensing are more complex. |

Compare the recommended continuation with approach 1 on identical frozen splits. Use approach 1 if continuation is unstable, metadata is incompatible, or official-task regression is excessive. Approach 3 is not recommended for the pilot.

## Installed API compatibility

Read-only checks on 2026-09-10 found:

- torch 2.14.0+cu130
- transformers 5.16.1
- peft 0.20.0
- accelerate 1.14.0
- bitsandbytes 0.50.2
- BitsAndBytesConfig, PeftModel, LoraConfig, prepare_model_for_kbit_training, and get_peft_model import successfully.
- PeftModel.from_pretrained exposes is_trainable=True. The generic PeftModel class does not itself advertise merge_and_unload; merge support must be verified on the actual loaded adapter subclass.

This proves API presence, not end-to-end training compatibility. Before training, pin these exact versions in the run manifest and perform a tiny disposable dry run. Verify the adapter-declared base, target modules, active adapter, trainable-parameter count, NF4, BF16 support, gradient flow, checkpoint reload, and held-out inference. Do not alter Torch or CUDA for this planning step.

## Pilot controls

- Use 300–500 human-confirmed rows covering News, Earnings Calls and 10-K.
- Split first by time or event group. Keep one news cluster, adjacent EC chunks, and nearby 10-K sections together.
- Keep the test set sealed and manually reviewed. Never select checkpoints using test performance.
- Never derive text labels from future five-day returns, later filings, or later outcomes.
- Model suggestions remain candidates until a human confirms every label and evidence span.
- Start with NF4 4-bit base loading, BF16 compute, batch size 1, gradient accumulation, gradient checkpointing, a paged optimizer where supported, and a conservative sequence limit. Measure VRAM before scaling.
- Record base/adapter revisions, schema version, data hashes, seeds, hyperparameters, package versions, GPU, VRAM, elapsed time and evaluation results.

## Preventing catastrophic forgetting

Preserve the official adapter unchanged, write every new checkpoint to a different versioned directory, and train only the copy. Do not use adapter stacking and do not merge the base with an adapter in phase one. Mix a reviewed rehearsal subset from official MT task formats with custom examples; begin with roughly 20–30% rehearsal and tune using held-out results. Use a low learning rate, few epochs, and early stopping. At every epoch evaluate both custom validation tasks and the frozen official-task suite. Before and after training, run frozen sentiment, headline, NER, relation/task-format probes plus the new dimensions. Reject a checkpoint if custom Macro-F1 does not improve or official-task capability materially regresses. Keep intermediate checkpoints so the tradeoff remains inspectable.

## Evaluation

Report every dimension separately: accuracy, Macro-F1, Matthews correlation coefficient (MCC), confusion matrix, label-selection counts and collapse detection. Also report evidence-valid rate and the retained capability rate on frozen official MT rehearsal tasks. Include per-source and per-label breakdowns where sample counts permit. Overall accuracy alone is not an acceptance metric because an imbalanced task can look accurate while always selecting one label.

## Future training and loading

A future continuation run should load the base in 4-bit NF4 with BF16 compute, call prepare_model_for_kbit_training, and load a copied MT adapter with is_trainable enabled. Device_map auto remains appropriate for the current inference probes; training must instead verify Accelerate-managed placement or an explicit single-GPU device map supported by the installed Trainer stack. Confirm exact PEFT arguments against version 0.20.0 and the adapter config in the dry run. Save only the new adapter, tokenizer files, schema, manifest and evaluation report unless licensing and storage plans explicitly permit a derived merged checkpoint.

Recommended inference loads the official base and exactly one versioned custom continuation adapter, then verifies the active adapter. Loading official and custom adapters together, weighted composition, or sequential stacking is outside this plan until independently tested.

Llama 2 access and redistribution remain subject to the Meta license and Hugging Face terms. Review the FinGPT adapter and dataset obligations before distributing a derivative. Never store access tokens in the repository.

## Transformer interface plan

No Transformer code changes are part of this pilot. A future validated extractor should emit, for each dimension:

- a categorical value;
- one relative score per allowed label;
- top-versus-second score margin;
- an applicability mask;
- an evidence-validation flag.

For directional dimensions, map negative, neutral and positive to -1, 0 and 1. Set the numeric value to null and applicable to 0 for insufficient_information, while optionally retaining its candidate score as a diagnostic. For binary dimensions, map no and yes to 0 and 1. Include every candidate score.

Relative scores are normalized only within one prompt’s candidate set. They are not calibrated probabilities and must never be named or interpreted as true probabilities.

Example fields are growth_outlook_value, growth_outlook_negative_score, growth_outlook_neutral_score, growth_outlook_positive_score, growth_outlook_insufficient_information_score, growth_outlook_margin, growth_outlook_applicable, and growth_outlook_evidence_valid.

Feature aggregation must retain published_at, the computation cutoff, source identity and error/missing masks. At time t, include only texts and news-cluster metadata available by t. Fit score calibration and downstream scalers on the training split only. Evidence-invalid or failed rows remain missing and must not become neutral.

## Acceptance gates

Require schema-valid data, adjudicated evidence, acceptable inter-annotator agreement, non-collapsed labels, source coverage, leakage-safe grouped splits, a reloadable tiny dry-run checkpoint, and no official-task regression beyond a predeclared tolerance. Passing the synthetic fixtures validates tooling only.
