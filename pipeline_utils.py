import sys
from pathlib import Path

import pandas as pd


SENTIMENT_COLUMNS = ["sentiment_label", "sentiment_score", "finbert_confidence"]


def safe_print(value) -> None:
    try:
        print(value)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(str(value).encode(encoding, errors="replace").decode(encoding))


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Output: {path}")
    print(f"Rows: {len(df)}")


def empty_dataframe(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def add_week_columns(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    df["week_start"] = df[date_column] - pd.to_timedelta(df[date_column].dt.weekday, unit="d")
    df["week_start"] = df["week_start"].dt.normalize()
    df["week_end"] = df["week_start"] + pd.Timedelta(days=6)
    return df


def round_float_columns(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    df = df.copy()
    float_columns = df.select_dtypes(include=["float"]).columns
    df[float_columns] = df[float_columns].round(decimals)
    return df


def load_finbert_pipeline():
    """Load FinBERT lazily so scripts can fail gracefully when dependencies are missing."""
    from transformers import pipeline

    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def score_label(label: str) -> int:
    label = str(label).lower()
    if label == "positive":
        return 1
    if label == "negative":
        return -1
    return 0


def analyze_texts_with_finbert(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        for column in SENTIMENT_COLUMNS:
            df[column] = []
        return df

    sentiment_model = load_finbert_pipeline()

    labels = []
    scores = []
    confidences = []
    for text in df[text_column].fillna("").astype(str):
        try:
            result = sentiment_model(text[:512])[0]
            label = result["label"].lower()
            confidence = float(result["score"])
            labels.append(label)
            scores.append(score_label(label))
            confidences.append(confidence)
        except Exception as exc:
            print(f"FinBERT error, using neutral fallback: {exc}")
            labels.append("neutral")
            scores.append(0)
            confidences.append(0.0)

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores
    df["finbert_confidence"] = confidences
    return df


def aggregate_text_weekly(
    df: pd.DataFrame,
    output_prefix: str,
    count_column: str,
) -> pd.DataFrame:
    weekly_columns = [
        "week_start",
        "week_end",
        f"{output_prefix}_count",
        f"{output_prefix}_positive_count",
        f"{output_prefix}_negative_count",
        f"{output_prefix}_neutral_count",
        f"{output_prefix}_sentiment_score",
        f"{output_prefix}_avg_confidence",
    ]
    if df.empty:
        return empty_dataframe(weekly_columns)

    df = add_week_columns(df, "date")
    weekly_df = df.groupby(["week_start", "week_end"]).agg(
        **{
            f"{output_prefix}_count": (count_column, "count"),
            f"{output_prefix}_positive_count": ("sentiment_score", lambda x: (x == 1).sum()),
            f"{output_prefix}_negative_count": ("sentiment_score", lambda x: (x == -1).sum()),
            f"{output_prefix}_neutral_count": ("sentiment_score", lambda x: (x == 0).sum()),
            f"{output_prefix}_sentiment_score": ("sentiment_score", "sum"),
            f"{output_prefix}_avg_confidence": ("finbert_confidence", "mean"),
        }
    ).reset_index()
    weekly_df["week_start"] = weekly_df["week_start"].dt.date.astype(str)
    weekly_df["week_end"] = weekly_df["week_end"].dt.date.astype(str)
    weekly_df = weekly_df.sort_values("week_start")
    return round_float_columns(weekly_df)


def aggregate_text_daily(
    df: pd.DataFrame,
    output_prefix: str,
    count_column: str,
) -> pd.DataFrame:
    daily_columns = [
        "date",
        f"{output_prefix}_count",
        f"{output_prefix}_positive_count",
        f"{output_prefix}_negative_count",
        f"{output_prefix}_neutral_count",
        f"{output_prefix}_sentiment_score",
        f"{output_prefix}_avg_confidence",
    ]
    if df.empty:
        return empty_dataframe(daily_columns)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.date.astype(str)
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0)
    df["finbert_confidence"] = pd.to_numeric(df["finbert_confidence"], errors="coerce").fillna(0)

    daily_df = df.groupby("date").agg(
        **{
            f"{output_prefix}_count": (count_column, "count"),
            f"{output_prefix}_positive_count": ("sentiment_score", lambda x: (x == 1).sum()),
            f"{output_prefix}_negative_count": ("sentiment_score", lambda x: (x == -1).sum()),
            f"{output_prefix}_neutral_count": ("sentiment_score", lambda x: (x == 0).sum()),
            f"{output_prefix}_sentiment_score": ("sentiment_score", "sum"),
            f"{output_prefix}_avg_confidence": ("finbert_confidence", "mean"),
        }
    ).reset_index()
    daily_df = daily_df.sort_values("date")
    return round_float_columns(daily_df[daily_columns])
