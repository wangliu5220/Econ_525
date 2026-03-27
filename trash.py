from typing import List, Dict, Optional
import json

# /c:/Users/jeremy/Econ_525/trash.py
# Small utility to build a ChatGPT-style prompt payload for a GPT API.


def build_chat_messages(
    task: str,
    context: Optional[str] = None,
    constraints: Optional[str] = None,
    examples: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Return a list of chat messages (system/user/assistant) suitable for a Chat API.
    examples: list of {"user": "...", "assistant": "..."} pairs.
    """
    system = {
        "role": "system",
        "content": (
            "You are a helpful, concise assistant. Follow user's constraints precisely. "
            "If you cannot answer, say so and explain briefly."
        ),
    }
    messages = [system]

    if context:
        messages.append({"role": "user", "content": f"Context:\n{context}"})

    user_block = f"Task:\n{task}"
    if constraints:
        user_block += f"\n\nConstraints:\n{constraints}"
    messages.append({"role": "user", "content": user_block})

    if examples:
        for ex in examples:
            if "user" in ex:
                messages.append({"role": "user", "content": ex["user"]})
            if "assistant" in ex:
                messages.append({"role": "assistant", "content": ex["assistant"]})

    return messages


def build_excel_sentiment_messages(
    excel_path: str,
    text_column: str = "news_text",
    score_column: str = "text_score",
    date_column: Optional[str] = "date",
    context: Optional[str] = None,
    constraints: Optional[str] = None,
    examples: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build chat messages to analyze economic sentiment from Excel news data."""
    system = {
        "role": "system",
        "content": (
            "You are an economic analyst. You receive an Excel file with news rows and "
            "a textual sentiment score. Explain sector-level and aggregate economic sentiment, "
            "identify trends, and suggest implications for macro outlook."
        ),
    }
    messages = [system]

    user_block = (
        f"Task:\nRead the Excel file at '{excel_path}'. "
        f"Use column '{text_column}' for news text and '{score_column}' for sentiment score."
    )
    if date_column:
        user_block += f" Use '{date_column}' as date/time context if available."
    if context:
        user_block += f"\n\nContext:\n{context}"
    if constraints:
        user_block += f"\n\nConstraints:\n{constraints}"

    user_block += (
        "\n\nOutput requirements:\n"
        "1) Classify overall economic sentiment (positive/neutral/negative),\n"
        "2) Highlight top 3 positive stories and top 3 negative stories by score,\n"
        "3) Show sentiment drift over time,\n"
        "4) Provide 2 brief policy/market implications."
    )

    messages.append({"role": "user", "content": user_block})

    if examples:
        for ex in examples:
            if "user" in ex:
                messages.append({"role": "user", "content": ex["user"]})
            if "assistant" in ex:
                messages.append({"role": "assistant", "content": ex["assistant"]})

    return messages


def score_sentiment(text: str) -> float:
    """Simple lexicon-based sentiment for a news headline/body text."""
    if not text or not isinstance(text, str):
        return 0.0

    text = text.lower()
    words = [w.strip(".,!?';:\"()[]") for w in text.split() if w.strip(".,!?';:\"()[]")] 

    positives = {
        "gain", "gains", "up", "rises", "rise", "strong", "beats", "outperforms",
        "positive", "improve", "improved", "optimistic", "profit", "growth", "win",
        "bull", "record", "advance", "strength", "support", "advances",
    }
    negatives = {
        "down", "falls", "fall", "drops", "drop", "weak", "misses", "underperforms",
        "negative", "decline", "declines", "loss", "weakness", "bear", "cuts", "lower",
        "risk", "worry", "truck", "slips", "slip", "vulnerable", "recession", "layoff",
    }

    pos_count = sum(1 for w in words if w in positives)
    neg_count = sum(1 for w in words if w in negatives)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return round((pos_count - neg_count) / total, 4)


def build_zero_shot_sentiment_prompt(text: str) -> str:
    """Return a zero-shot prompt for news sentiment scoring (no dictionary)."""
    sanitized = text.strip().replace('\n', ' ').replace('\r', ' ')
    return (
        "You are a financial sentiment analyst.\n"
        "Given the following news text, assign a sentiment score between -1.0 (very negative) and +1.0 (very positive).\n"
        "Also provide a one-sentence rationale. Respond in JSON as {\"sentiment\": 0.XX, \"rationale\": \"...\"}.\n\n"
        f"News text: '{sanitized}'"
    )


def score_csv_news_items(
    input_csv: str,
    output_csv: Optional[str] = None,
    text_column: str = "headline",
    extra_text_column: Optional[str] = "event_text",
    method: str = "zero_shot",
) -> str:
    """Read a news CSV, compute per-row sentiment, save to output CSV, return path."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for score_csv_news_items; install with 'pip install pandas'."
        )

    df = pd.read_csv(input_csv)
    if text_column not in df.columns and extra_text_column not in df.columns:
        raise ValueError(f"None of columns '{text_column}' or '{extra_text_column}' found in {input_csv}")

    def _join_text(row):
        primary = str(row[text_column]) if text_column in row and pd.notna(row[text_column]) else ""
        extra = str(row[extra_text_column]) if extra_text_column and extra_text_column in row and pd.notna(row[extra_text_column]) else ""
        return (primary + " "+extra).strip() or ""

    all_text = df.apply(_join_text, axis=1)

    if method == "lexicon":
        df["sentiment_score"] = all_text.apply(score_sentiment)
    elif method == "zero_shot":
        # Zero-shot requests should be sent to GPT; we include prompts here.
        df["sentiment_prompt"] = all_text.apply(build_zero_shot_sentiment_prompt)
        # sentiment_score remains empty; scoring requires external model call.
        df["sentiment_score"] = None
    else:
        raise ValueError("method must be 'lexicon' or 'zero_shot'")

    if not output_csv:
        output_csv = input_csv.replace(".csv", "_with_sentiment.csv")

    df.to_csv(output_csv, index=False)
    return output_csv


if __name__ == "__main__":
    # Example usage: build a prompt payload and print JSON for sending to a GPT chat API.
    prompt = build_excel_sentiment_messages(
        excel_path="/path/to/news_data.xlsx",
        text_column="headline",
        score_column="text_score",
        date_column="published_date",
        constraints="Keep response concise with bullet points and include a numeric sentiment index.",
        examples=[
            {
                "user": "Analyze sample news sentiment dataset, explain economic direction.",
                "assistant": "Overall positive with moderation, inflation worries remain."
            }
        ],
    )
    print(json.dumps(prompt, indent=2))

    # New behavior: compute per-row sentiment for a CSV file with headlines and optional event text.
    output_file = score_csv_news_items(
        input_csv="llm_input_prompts.csv",
        text_column="headline",
        extra_text_column="event_text",
    )
    print("Created sentiment-scored CSV:", output_file)
