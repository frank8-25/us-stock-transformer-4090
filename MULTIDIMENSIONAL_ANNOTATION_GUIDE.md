# FinGPT MT 7B multidimensional annotation guide (v1)

## Scope and annotation unit

Each JSONL row is one financial-text fragment publicly available at published_at. Allowed sources are news, earnings_call, and ten_k. Annotators may use only the row and information already public at that timestamp. Future prices, later filings, later news, and hindsight are prohibited. Synthetic examples validate this specification; they are not training data.

The exact model input and evidence coordinate system is:

    title + "\n" + text   when title is nonempty
    text                  when title is empty

Strings are used exactly as stored. Offsets are zero-based Python slices [start:end]. Prefer the shortest quote that supports the label. Do not paraphrase evidence. The validator requires canonical_input[start:end] == quote.

## Shared label rules

The six directional dimensions use negative, neutral, positive, or insufficient_information.

- neutral means the text addresses that dimension and states a balanced, unchanged, immaterial, or explicitly neutral effect. It requires evidence.
- insufficient_information means the text does not support a judgment for that dimension. It must have no evidence for that dimension.
- A mixed passage is neutral only when supported positive and negative implications reasonably balance.
- If timing differs, label short-term and long-term separately.
- Every directional label except insufficient_information requires evidence.

The four judgment dimensions use only yes and no. Each question asks whether a described condition exists, so absence is a meaningful no; a third label would blur absence with ambiguity. Every yes requires evidence. A no may omit evidence because absence usually cannot be quoted.

## Directional dimensions

### 1. sentiment

- Definition: The overall financial sentiment and impact direction of the information specifically toward the target company NVDA. It is not the prose tone and not sentiment about another company.
- Question: Does the information imply a negative, balanced or unchanged, positive, or unsupported financial effect for NVDA?
- Labels: negative, neutral, positive, insufficient_information.
- Positive example: “NVIDIA reported record demand and raised guidance.”
- Negative example: “NVIDIA orders fell and management cut guidance.”
- Boundary: A negative event at another company with no NVDA connection is relevance=no and sentiment=insufficient_information. A broad-market decline explicitly affecting NVDA is labeled from the stated NVDA evidence. Optimistic prose with no substantive NVDA information is not positive merely because of its style. Mixed NVDA benefits and harms can be neutral when they balance.
- Source notes: News labels follow target-specific information rather than sensational wording. EC labels reflect implications of management statements for NVDA. 10-K boilerplate risk can be negative but is not automatically a realized loss.

### 2. growth_outlook

- Definition: Direction of future revenue, demand, adoption, capacity-supported sales, or business growth.
- Question: What direction does the text support for future business growth?
- Labels: the four directional labels.
- Positive example: Raised future revenue guidance after new commitments.
- Negative example: Lowered outlook because customers delayed orders.
- Boundary: Strong current revenue without a future statement is insufficient; offsetting drivers and constraints may be neutral.
- Source notes: News needs a future mechanism. EC guidance is direct evidence. Historical 10-K growth alone does not imply future growth.

### 3. financial_strength

- Definition: Effect on profitability, margins, cash flow, liquidity, leverage, or balance-sheet resilience.
- Question: Does the text indicate weaker, balanced or unchanged, stronger, or unsupported financial strength?
- Labels: the four directional labels.
- Positive example: Higher operating cash flow and expanding gross margin.
- Negative example: A write-down and lower gross margin.
- Boundary: Revenue growth without margin, cash, or balance-sheet information may be insufficient.
- Source notes: EC non-GAAP claims need their stated basis. 10-K facts are direct evidence. Share-price movement does not establish financial strength.

### 4. fundamental_impact

- Definition: Direction of impact on operations, products, demand, supply, costs, or financial fundamentals.
- Question: What fundamental business effect is supported?
- Labels: the four directional labels.
- Positive example: Cloud providers commit to deploy a new NVIDIA platform.
- Negative example: Export controls interrupt production or sales.
- Boundary: Pure price movement is insufficient; a stated immaterial operating effect is neutral.
- Source notes: Verify the target entity in news. Distinguish EC plans from completed actions and 10-K risk disclosure from realized events.

### 5. short_term_impact

- Definition: Direction of business impact over the next days to several weeks.
- Question: What near-term effect is supported without predicting stock returns?
- Labels: the four directional labels.
- Positive example: Raised current-quarter shipments after capacity became available.
- Negative example: A current-quarter order delay.
- Boundary: Long-run opportunity without a near-term consequence is insufficient; balanced immediate effects can be neutral.
- Source notes: Use publication-time horizons. EC quarterly guidance is often relevant. Annual 10-K risks need a near-term trigger.

### 6. long_term_impact

- Definition: Direction over months or longer for competitiveness, durable demand, capacity, or growth.
- Question: What durable business implication does the text support?
- Labels: the four directional labels.
- Positive example: Long-term platform commitments expand the ecosystem.
- Negative example: Persistent export controls shrink addressable markets.
- Boundary: A one-day price move is insufficient; a temporary charge with no durable consequence may be neutral.
- Source notes: News needs a durable mechanism. EC strategy is not a completed result. 10-K structural risks and advantages must be company-specific.

## Judgment dimensions

### 7. risk_presence

- Definition and question: Does the text describe a concrete downside risk that could harm operations, financial performance, or outlook?
- Labels: yes, no. There is no third label because this detects a described risk, not its probability.
- Positive example: Confirmed cost increases reduce margin.
- Negative example: A routine reporting date with no adverse mechanism.
- Boundary: Pending policy can be both risk and uncertainty; generic boilerplate needs a concrete exposure.
- Source notes: Distinguish news speculation, EC qualifiers, and formal 10-K Risk Factors. Do not infer severity.

### 8. uncertainty_presence

- Definition and question: Does the text state that an outcome, timing, policy, demand level, or execution result is unresolved?
- Labels: yes, no. Uncertainty is either expressed or absent in the supplied text.
- Positive example: Regulators have not decided the rule’s scope or date.
- Negative example: A final charge with a fixed amount and payment date.
- Boundary: A known negative fact is risk but not necessarily uncertainty. Routine future scheduling is not uncertainty.
- Source notes: EC hedging counts only when a material result remains unresolved. 10-K “may” language requires context.

### 9. relevance

- Definition and question: Is the information materially related to the target company’s operations, finance, competitiveness, or outlook?
- Labels: yes, no. Target relevance is decidable from the canonical input.
- Positive example: NVIDIA commitments affecting demand.
- Negative example: An unrelated retailer’s advertising campaign.
- Boundary: A stock-price-only article is relevance=no for this fundamental-feature definition even if it names NVDA.
- Source notes: Resolve entity identity in news. EC and 10-K fragments still need substantive company context.

### 10. price_narrative

- Definition and question: Is the passage mainly about observed share-price movement, technical momentum, or market performance without new fundamentals?
- Labels: yes, no. The condition is directly observable in the text.
- Positive example: Shares reached a high on technical momentum with no business update.
- Negative example: Raised revenue guidance supported by new customer orders.
- Boundary: A story with price reaction and material new guidance is no because price is not the only substantive information.
- Source notes: Most EC and 10-K text should be no. Never infer that information is “priced in.”

## Evidence and review workflow

Evidence keys may only name the ten dimensions. Each item contains exactly quote, start, and end. Multiple spans are allowed when a balanced label needs both sides. Evidence fidelity does not prove interpretive correctness. Evidence emitted by a model is never trusted directly; programmatic validation must confirm every quote and offset against canonical input.

Annotators assign labels and evidence independently, then resolve disagreements without future outcomes. Freeze labels before model evaluation. Models may suggest candidates, but output is not ground truth until a human confirms every label and span.

Split by publication time or event group before export. The same news cluster, adjacent chunks from one earnings call, and nearby sections from one 10-K must remain in one split. as_of_date equals the UTC date of published_at. Exact cross-split duplicates are errors; normalized similarity at or above 0.92 is a warning.
