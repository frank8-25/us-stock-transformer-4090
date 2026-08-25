import os
from pathlib import Path

# Company switch settings. Change this block when switching to another stock.
TICKER = "NVDA"
COMPANY_NAME = "NVIDIA"
START_DATE = "2020-01-01"
END_DATE = "2026-6-28"
SEARCH_KEYWORDS = ["NVIDIA", "NVDA", "NVIDIA stock"]
GOOGLE_TRENDS_KEYWORDS = ["NVIDIA", "NVDA", "NVIDIA stock"]

# Earnings call source settings. Q4 Inc IR endpoints work for many large US companies,
# but the API URL is company-specific and should be changed with the ticker.
INVESTOR_RELATIONS_URL = "https://investor.nvidia.com/financial-info/quarterly-results/default.aspx"
EARNINGS_CALL_IR_API_URL = "https://investor.nvidia.com/feed/FinancialReport.svc/GetFinancialReportList"
EARNINGS_CALL_SOURCE_NAME = "NVIDIA Investor Relations"
EARNINGS_CALL_REPORT_TYPES = ["First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"]
EARNINGS_CALL_DOCUMENT_PRIORITY = ["transcript", "cfo", "news"]

DATA_DIR = "data"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
FINAL_DIR = "data/final"
MODEL_DIR = "models"

TEXT_SOURCES = ["news", "earnings_call", "ten_k"]
NON_TEXT_SOURCES = ["google_trends"]
DATA_SOURCES = [
    "news",
    "earnings_call",
    "ten_k",
    "google_trends",
]

PREDICTION_HORIZON_DAYS = 5

SOURCE_LANGUAGE = "english"


def _quote_query_term(term: str) -> str:
    return f'"{term}"' if " " in term else term


QUERY = f"({' OR '.join(_quote_query_term(term) for term in SEARCH_KEYWORDS)}) sourcelang:{SOURCE_LANGUAGE}"

# Source directories.
RAW_NEWS_DIR = f"{RAW_DIR}/news"
RAW_EARNINGS_CALL_DIR = f"{RAW_DIR}/earnings_call"
RAW_TEN_K_DIR = f"{RAW_DIR}/ten_k"
RAW_GOOGLE_TRENDS_DIR = f"{RAW_DIR}/google_trends"

PROCESSED_NEWS_DIR = f"{PROCESSED_DIR}/news"
PROCESSED_EARNINGS_CALL_DIR = f"{PROCESSED_DIR}/earnings_call"
PROCESSED_TEN_K_DIR = f"{PROCESSED_DIR}/ten_k"
PROCESSED_GOOGLE_TRENDS_DIR = f"{PROCESSED_DIR}/google_trends"
PROCESSED_FINGPT_DIR = f"{PROCESSED_DIR}/fingpt"
PROCESSED_FINGPT_NEWS_DIR = f"{PROCESSED_FINGPT_DIR}/news"
PROCESSED_FINGPT_EARNINGS_CALL_DIR = f"{PROCESSED_FINGPT_DIR}/earnings_call"
PROCESSED_FINGPT_TEN_K_DIR = f"{PROCESSED_FINGPT_DIR}/ten_k"

FEATURE_DIR = f"{FINAL_DIR}/features"
FINGPT_FEATURE_DIR = f"{FEATURE_DIR}/fingpt"
DATASET_DIR = f"{FINAL_DIR}/dataset"

# Canonical output files for the multi-source pipeline.
RAW_NEWS_FILE = f"{RAW_NEWS_DIR}/{TICKER}_news_raw.csv"
PROCESSED_NEWS_FILE = f"{PROCESSED_NEWS_DIR}/{TICKER}_news_with_sentiment.csv"
PROCESSED_FINGPT_NEWS_FILE = f"{PROCESSED_FINGPT_NEWS_DIR}/{TICKER}_news_with_fingpt_sentiment.csv"
WEEKLY_NEWS_FEATURE_FILE = f"{FEATURE_DIR}/{TICKER}_news_weekly_features.csv"
DAILY_NEWS_FEATURE_FILE = f"{FEATURE_DIR}/{TICKER}_news_daily_features.csv"
DAILY_FINGPT_NEWS_FEATURE_FILE = f"{FINGPT_FEATURE_DIR}/{TICKER}_news_daily_features.csv"

RAW_EARNINGS_CALL_FILE = f"{RAW_EARNINGS_CALL_DIR}/{TICKER}_earnings_call_raw.csv"
MANUAL_EARNINGS_CALL_FILE = f"{RAW_EARNINGS_CALL_DIR}/{TICKER}_earnings_call_manual.csv"
PROCESSED_EARNINGS_CALL_FILE = f"{PROCESSED_EARNINGS_CALL_DIR}/{TICKER}_earnings_call_with_sentiment.csv"
PROCESSED_FINGPT_EARNINGS_CALL_FILE = f"{PROCESSED_FINGPT_EARNINGS_CALL_DIR}/{TICKER}_earnings_call_with_fingpt_sentiment.csv"
WEEKLY_EARNINGS_CALL_FEATURE_FILE = f"{FEATURE_DIR}/{TICKER}_earnings_call_weekly_features.csv"
DAILY_EARNINGS_CALL_FEATURE_FILE = f"{FEATURE_DIR}/{TICKER}_earnings_call_daily_features.csv"
DAILY_FINGPT_EARNINGS_CALL_FEATURE_FILE = f"{FINGPT_FEATURE_DIR}/{TICKER}_earnings_call_daily_features.csv"

RAW_TEN_K_FILE = f"{RAW_TEN_K_DIR}/{TICKER}_10k_raw.csv"
PROCESSED_TEN_K_FILE = f"{PROCESSED_TEN_K_DIR}/{TICKER}_10k_with_sentiment.csv"
PROCESSED_FINGPT_TEN_K_FILE = f"{PROCESSED_FINGPT_TEN_K_DIR}/{TICKER}_10k_with_fingpt_sentiment.csv"
WEEKLY_TEN_K_FEATURE_FILE = f"{FEATURE_DIR}/{TICKER}_10k_weekly_features.csv"
DAILY_TEN_K_FEATURE_FILE = f"{FEATURE_DIR}/{TICKER}_10k_daily_features.csv"
DAILY_FINGPT_TEN_K_FEATURE_FILE = f"{FINGPT_FEATURE_DIR}/{TICKER}_10k_daily_features.csv"

RAW_GOOGLE_TRENDS_FILE = f"{RAW_GOOGLE_TRENDS_DIR}/{TICKER}_google_trends_raw.csv"
WEEKLY_GOOGLE_TRENDS_FEATURE_FILE = f"{FEATURE_DIR}/{TICKER}_google_trends_weekly_features.csv"
DAILY_GOOGLE_TRENDS_FEATURE_FILE = f"{FEATURE_DIR}/{TICKER}_google_trends_daily_features.csv"

FUSION_WEEKLY_FEATURE_FILE = f"{DATASET_DIR}/{TICKER}_fusion_weekly_features.csv"
FUSION_WEEKLY_DATASET_FILE = f"{DATASET_DIR}/{TICKER}_fusion_weekly_dataset.csv"
FUSION_DAILY_FEATURE_FILE = f"{DATASET_DIR}/{TICKER}_fusion_daily_features.csv"
FUSION_DAILY_DATASET_FILE = f"{DATASET_DIR}/{TICKER}_fusion_daily_dataset.csv"
TRANSFORMER_MODEL_FILE = f"{MODEL_DIR}/{TICKER}_weekly_transformer.pt"
TRANSFORMER_PREDICTION_FILE = f"{DATASET_DIR}/{TICKER}_transformer_predictions.csv"
DAILY_TRANSFORMER_MODEL_FILE = f"{MODEL_DIR}/{TICKER}_daily_transformer.pt"
DAILY_TRANSFORMER_PREDICTION_FILE = f"{DATASET_DIR}/{TICKER}_daily_transformer_predictions.csv"

# GDELT API settings.
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS_PER_DAY = 80
SLEEP_SECONDS = 5
RETRY_LIMIT = 2
RETRY_SLEEP_SECONDS = 60

# SEC asks automated clients to identify themselves. Override this locally with:
# $env:SEC_USER_AGENT="your-name your-email@example.com"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "us-stock-transformer research contact@example.com")


def ensure_directories() -> None:
    """Create the directory layout used by every pipeline stage."""
    for path in [
        RAW_NEWS_DIR,
        RAW_EARNINGS_CALL_DIR,
        RAW_TEN_K_DIR,
        RAW_GOOGLE_TRENDS_DIR,
        PROCESSED_NEWS_DIR,
        PROCESSED_EARNINGS_CALL_DIR,
        PROCESSED_TEN_K_DIR,
        PROCESSED_GOOGLE_TRENDS_DIR,
        PROCESSED_FINGPT_NEWS_DIR,
        PROCESSED_FINGPT_EARNINGS_CALL_DIR,
        PROCESSED_FINGPT_TEN_K_DIR,
        FEATURE_DIR,
        FINGPT_FEATURE_DIR,
        DATASET_DIR,
        MODEL_DIR,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)
