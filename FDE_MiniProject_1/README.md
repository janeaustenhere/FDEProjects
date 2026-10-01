# Dhaga & Co. Returns Intelligence

Streamlit MVP that combines existing structured return reasons with AI interpretation of free-text comments recorded as `Other`.

## Processing rules

- Existing reasons such as `Size Issue`, `Defective Pieces`, `Not Delivered`, `Late Delivery`, and `Wrong Item` are accepted without an LLM call.
- Only `Other` comments enter AI interpretation. Every AI candidate is evaluated before automatic acceptance; difficult or evaluator-rejected bulk results are escalated to the strong model and evaluated again.
- AI results are aligned to an existing business reason where meanings match.
- A clear reason outside the existing categories is displayed as a new category with its reason description.
- Vague or unsupported comments enter human review instead of being guessed.
- Python calculates all counts and percentages.

## Dashboard outputs

- Overall return-reason percentage pie chart and table
- SKU/vendor/return-reason percentages
- AI classification coverage for `Other`
- Complete AI-classified `Other` table
- Newly discovered category and description table
- Human-review table with return ID, SKU ID, vendor, comment, suggestion, confidence, and review reason

Classification coverage is not model accuracy. Accuracy requires human-labelled ground truth.

## Required CSV columns

```text
return_id, sku_id, category, return_reason, return_comment
```

Add `vendor` for vendor-level analysis. Files without it use `Unknown Vendor`. Optional `total_orders`, `is_returned`, or `order_status` fields allow the app to calculate an overall return rate; a return-only file without a denominator displays `Not available`.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

AI configuration is environment-only and never appears in the client UI. Configure `.env` or deployment secrets:

```text
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4.1-mini
OPENROUTER_BULK_MODEL=openai/gpt-4.1-mini
OPENROUTER_STRONG_MODEL=openai/gpt-4.1
OPENROUTER_EVALUATOR_MODEL=openai/gpt-4.1-mini
OPENROUTER_TEMPERATURE=0.1
OPENROUTER_HTTP_REFERER=
OPENROUTER_APP_TITLE=Dhaga Returns Intelligence
LLM_PROVIDER_NAME=OpenRouter

CONFIDENCE_THRESHOLD=0.75
MAX_CONCURRENT_REQUESTS=5
MAX_BATCH_SIZE=50
EVALUATOR_BATCH_SIZE=50
MAX_UPLOAD_MB=10
LLM_TIMEOUT_SECONDS=60
LLM_MAX_ATTEMPTS=3
LLM_BACKOFF_BASE_SECONDS=1
LLM_STRUCTURED_OUTPUT_METHOD=json_schema
LLM_STRUCTURED_OUTPUT_STRICT=false
```

Model prices can also be configured with the `*_MODEL_INPUT_COST_PER_1M` and
`*_MODEL_OUTPUT_COST_PER_1M` variables shown in `.env.example`. Invalid numeric,
boolean, model, or limit values fail at startup with a clear configuration error.
Restart Streamlit after changing environment configuration. Never commit `.env`.

## Design

- `Settings` is the single owner of environment-driven runtime configuration.
- `reason_registry` owns source-reason aliases, structured mappings, and business labels.
- Classifiers and evaluators depend on an `LLMClient` protocol, so provider adapters can be replaced in tests or production.
- The OpenAI-compatible adapter caches structured LangChain runnables, applies bounded retry/backoff, and records usage.
- The pipeline preserves input order, sends bulk classifications and final evaluations in bounded batches, limits concurrent strong-model requests, and stops new AI work after fatal provider/configuration errors.
- Shared dashboard components keep overview and detailed analysis calculations consistent.

## Tests

```bash
pytest -q
```

## Deployment

The included Dockerfile runs Streamlit on port 7860. Store the API key in the hosting platform's secret manager.
