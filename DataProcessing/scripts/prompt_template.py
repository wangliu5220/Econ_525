"""Pydantic schemas, system prompt, and batch formatter for the LLM scoring pipeline."""

from typing import Optional

import pandas as pd
from pydantic import BaseModel


class HeadlineScore(BaseModel):
    row_id: int
    tangibility: Optional[float]
    relevance: Optional[float]
    llm_sentiment: Optional[float]


class BatchResponse(BaseModel):
    scores: list[HeadlineScore]


def get_gemini_schema() -> dict:
    """
    Return a Gemini-compatible JSON schema for BatchResponse.

    Gemini's structured output doesn't support $defs/$ref or anyOf,
    so we resolve references inline and simplify nullable types to
    plain number types.
    """
    schema = BatchResponse.model_json_schema()
    defs = schema.pop("$defs", {})

    def _resolve(obj):
        if isinstance(obj, dict):
            # Replace $ref with the actual definition
            if "$ref" in obj:
                ref_name = obj.pop("$ref").split("/")[-1]
                obj.update(_resolve(defs[ref_name].copy()))
            # Simplify anyOf nullable: [{type: X}, {type: null}] → {type: X}
            if "anyOf" in obj:
                non_null = [t for t in obj["anyOf"] if t.get("type") != "null"]
                if non_null:
                    obj.pop("anyOf")
                    obj.update(non_null[0])
            # Remove Pydantic-specific fields Gemini doesn't understand
            obj.pop("title", None)
            # Recurse into all remaining dict values
            for key, val in obj.items():
                obj[key] = _resolve(val)
        elif isinstance(obj, list):
            return [_resolve(item) for item in obj]
        return obj

    return _resolve(schema)


# System prompt
SYSTEM_PROMPT = """\
You are a financial analyst scoring news headlines for a quantitative trading strategy.

For each headline, provide three scores on a [-1, 1] scale:

1. **tangibility**: How concrete and economically substantive is the information?
   - +1: Contains specific financial data (earnings numbers, revenue figures, concrete contract values, specific price targets with dollar amounts)
   - 0: Moderately specific (analyst rating changes, product launches without financial details, named partnerships)
   - -1: Vague, speculative, or purely descriptive of price movement with no causal information ("stock rises", "outperforms market")

2. **relevance**: How directly does this headline relate to the fundamental value of the specific TICKER?
   - +1: Directly about the company's financials, operations, or competitive position (earnings report, major contract, product launch BY the company)
   - 0: Indirectly related (industry trend, peer comparison, sector rotation)
   - -1: Barely related or about a different entity (headline mentions ticker only in passing, roundup articles); also -1 for pure price-recap headlines ("stock rises", "outperforms market", "stock up 2%") that describe price movement WITHOUT any causal company-specific information — these tell investors nothing new about fundamentals

3. **llm_sentiment**: What is the expected directional impact on the stock price?
   - +1: Strongly positive (earnings beat, major contract win, strong guidance raise)
   - 0: Neutral or mixed (rating maintained, routine dividend, lateral partnership)
   - -1: Strongly negative (earnings miss, loss guidance, downgrade, major contract loss)
   - IMPORTANT: A bare historical earnings figure (e.g. "EPS 13c", "1Q EPS $5.98") with no beat/miss signal should be scored near 0 — without consensus context you cannot determine beat vs. miss. Only score positively if the headline explicitly signals outperformance (e.g., "beat estimates", "above expectations", "record earnings", "tops views").
   - EXCEPTION: Forward guidance with specific dollar figures (e.g., "Sees Q2 Revenue $2.04B–$2.05B", "raises full-year outlook to $X") is a positive signal — score +0.4 to +0.7 depending on how strong the raise appears.

RULES:
- Use the FULL range of [-1, 1]. Do not cluster around 0.
- Return null for any score you truly cannot assess (e.g., headline is unintelligible or completely empty).
- Use ONLY the row_ids provided in the input. Do not invent new ones.
- Consider BOTH the headline AND the event_text when scoring.

EXAMPLES:

Input headline (INTC): "MW Intel stock drops nearly 10% after earnings miss, execs predict quarterly loss as data-center market shrinks"
Event_text: "Intel Corp. shares dropped more than 9% in the extended session Thursday"
→ tangibility: 0.85, relevance: 0.95, llm_sentiment: -0.90

Input headline (AVGO): "Broadcom Inc. Stock Outperforms Market On Strong Trading Day"
Event_text: "Shares of Broadcom Inc. (AVGO) rallied 2.60% to $845.26 Friday"
→ tangibility: -0.70, relevance: -0.70, llm_sentiment: 0.10
(Pure price recap — no causal information about fundamentals. Relevance is low even though it's about the same company.)

Input headline (PLTR): "Palantir's Stock Rises As Company Discloses New Army AI Contract Worth Up To $250 Million"
Event_text: "Palantir shares were up about 2% in Tuesday"
→ tangibility: 0.90, relevance: 0.95, llm_sentiment: 0.80
"""


# Batch formatter
def format_batch_prompt(batch_rows: list[dict]) -> str:
    """
    Format a list of headline dicts into the user-message portion of the prompt.

    Each dict must have keys: row_id, TICKER, headline, event_text.
    """
    lines = ["Score the following headlines:\n"]
    for row in batch_rows:
        headline = str(row["headline"]) if pd.notna(row["headline"]) else "(empty)"
        event_text = str(row["event_text"]) if pd.notna(row["event_text"]) else "(empty)"
        lines.append(
            f"[row_id={row['row_id']}] TICKER={row['TICKER']}\n"
            f"  Headline: {headline}\n"
            f"  Event_text: {event_text}\n"
        )
    return "\n".join(lines)
