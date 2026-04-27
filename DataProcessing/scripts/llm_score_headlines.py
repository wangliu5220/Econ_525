"""
Score financial news headlines using Gemini for tangibility, relevance, and sentiment.
Supports checkpoint/resume for long-running batches.

Usage:
    python scripts/llm_score_headlines.py             # full run
    python scripts/llm_score_headlines.py --validate  # validation set only
"""

import argparse
import json
import pathlib
import sys
import time

# Ensure scripts/ is on the import path (needed when running from project root)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompt_template import SYSTEM_PROMPT, get_gemini_schema, format_batch_prompt

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

INPUT_PATH = DATA_DIR / "llm_input_prompts.csv"
CHECKPOINT_PATH = DATA_DIR / "llm_scores_checkpoint.csv"
VALIDATION_PATH = DATA_DIR / "validation_set_scored.csv"

# Configuration
BATCH_SIZE = 30              # headlines per API request
SLEEP_BETWEEN_CALLS = 6.0   # seconds → ~10 requests/minute (under ~15 RPM limit)
CHECKPOINT_INTERVAL = 100    # batches between checkpoint writes

# Backoff for 429 rate-limit errors
BACKOFF_INITIAL = 30         # seconds
BACKOFF_FACTOR = 2
BACKOFF_MAX = 120            # seconds
MAX_RETRIES = 5

MODEL_NAME = "gemini-3.1-flash-lite-preview"

# API client
load_dotenv(PROJECT_DIR / ".env")
client = genai.Client()      # auto-detects GEMINI_API_KEY from environment


# Checkpoint helpers
def load_checkpoint() -> pd.DataFrame:
    """Load existing checkpoint if present, else return empty DataFrame."""
    if CHECKPOINT_PATH.exists():
        try:
            df = pd.read_csv(CHECKPOINT_PATH)
            # Deduplicate in case of corrupted write
            df = df.drop_duplicates(subset=["row_id"], keep="last")
            print(f"Loaded {len(df):,} scored rows from checkpoint.")
            return df
        except Exception as e:
            print(f"Warning: could not read checkpoint ({e}), starting fresh.")
    return pd.DataFrame(columns=["row_id", "tangibility", "relevance", "llm_sentiment"])


def save_checkpoint(results_df: pd.DataFrame) -> None:
    """Overwrite checkpoint file with all results accumulated so far."""
    results_df.to_csv(CHECKPOINT_PATH, index=False)
    print(f"Checkpoint: {len(results_df):,} rows saved.")


# API call with exponential backoff
def call_api(prompt: str) -> str | None:
    """
    Send a single prompt to Gemini.  Returns the raw response text.
    Handles 429 errors with exponential backoff.  Returns None after
    MAX_RETRIES failures.
    """
    backoff = BACKOFF_INITIAL
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_json_schema=get_gemini_schema(),
                    temperature=0.2,
                ),
            )
            # Guard against empty / blocked responses
            if response.text is None or response.text.strip() == "":
                print(f"  Empty response (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(backoff)
                backoff = min(backoff * BACKOFF_FACTOR, BACKOFF_MAX)
                continue
            return response.text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"  Rate limited, backing off {backoff}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(backoff)
                backoff = min(backoff * BACKOFF_FACTOR, BACKOFF_MAX)
            else:
                print(f"  API error: {err_str[:200]}")
                return None
    print(f"  Failed after {MAX_RETRIES} retries.")
    return None


# Response parsing
def parse_and_validate(raw_json: str, expected_ids: set[int]) -> list[dict]:
    """
    Parse API response, validate row_ids and score ranges.
    Returns a list of validated score dicts.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        print("  Malformed JSON response.")
        return []

    # Handle both {"scores": [...]} and bare [...]
    if isinstance(data, dict) and "scores" in data:
        items = data["scores"]
    elif isinstance(data, list):
        items = data
    else:
        print("  Unexpected JSON structure, skipping batch.")
        return []

    validated = []
    for item in items:
        rid = item.get("row_id")

        # Row ID validation
        if rid not in expected_ids:
            print(f"  Unexpected row_id={rid}, discarding.")
            continue

        scores = {"row_id": rid}

        # Score validation: clamp to [-1, 1], allow null → NaN
        for key in ("tangibility", "relevance", "llm_sentiment"):
            val = item.get(key)
            if val is None:
                scores[key] = np.nan
            else:
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    scores[key] = np.nan
                    print(f"  [Type Error] row_id={rid}, {key}={item.get(key)} "
                          f"not numeric. Set to NaN.")
                    continue
                if val < -1 or val > 1:
                    print(f"  [Range] row_id={rid}, {key}={val:.3f} "
                          f"out of [-1,1]. Clamping.")
                    val = max(-1.0, min(1.0, val))
                scores[key] = val

        validated.append(scores)

    return validated


def main():
    parser = argparse.ArgumentParser(
        description="Score headlines via Gemini LLM."
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run only on the validation set (small batches of 5).",
    )
    args = parser.parse_args()

    # --- Load input data ---
    if args.validate:
        if not VALIDATION_PATH.exists():
            sys.exit(f"Validation set not found at {VALIDATION_PATH}")
        df = pd.read_csv(VALIDATION_PATH)
        # validation_set.csv already has row_id column
        batch_size = 5
        print(f"Validation mode: {len(df)} headlines loaded.")
    else:
        df = pd.read_csv(INPUT_PATH)
        df["row_id"] = df.index
        batch_size = BATCH_SIZE
        print(f"Loaded {len(df):,} headlines for scoring.")

    # --- Resume from checkpoint ---
    checkpoint_df = load_checkpoint()
    completed_ids = (
        set(checkpoint_df["row_id"].tolist()) if len(checkpoint_df) > 0 else set()
    )

    if completed_ids:
        remaining = df[~df["row_id"].isin(completed_ids)].copy()
        print(f"{len(completed_ids):,} already scored, {len(remaining):,} remaining.")
    else:
        remaining = df.copy()

    if remaining.empty:
        print("All rows already scored. Nothing to do.")
        return

    # --- Chunk into batches ---
    batches = [
        remaining.iloc[i : i + batch_size]
        for i in range(0, len(remaining), batch_size)
    ]
    total_batches = len(batches)
    print(f"Total batches to process: {total_batches}")

    # --- Accumulator for new results ---
    new_results: list[dict] = []
    batches_since_checkpoint = 0
    errors_logged = 0
    run_start = time.time()

    try:
        for batch_idx, batch_df in enumerate(batches):
            # Format the prompt
            batch_rows = batch_df[
                ["row_id", "TICKER", "headline", "event_text"]
            ].to_dict("records")
            prompt = format_batch_prompt(batch_rows)
            expected_ids = set(batch_df["row_id"].tolist())

            # Call API
            raw = call_api(prompt)

            if raw is None:
                # Total failure after retries — retry the batch once more
                print(f"  Batch {batch_idx + 1}: API call failed. "
                      f"Retrying once...")
                time.sleep(BACKOFF_INITIAL)
                raw = call_api(prompt)

            if raw is not None:
                validated = parse_and_validate(raw, expected_ids)
                new_results.extend(validated)

                missing = expected_ids - {r["row_id"] for r in validated}
                if missing:
                    errors_logged += len(missing)
                    print(f"  Batch {batch_idx + 1}: "
                          f"Missing row_ids {missing} in response.")
            else:
                errors_logged += len(expected_ids)
                print(f"  Batch {batch_idx + 1}: "
                      f"Skipped entirely after retries.")

            batches_since_checkpoint += 1

            # Per-batch heartbeat — always printed so you can see the script is alive
            total_done = len(completed_ids) + len(new_results)
            elapsed = time.time() - run_start
            batches_done_this_run = batch_idx + 1
            avg_seconds = elapsed / batches_done_this_run
            remaining_batches = total_batches - batches_done_this_run
            eta_seconds = avg_seconds * remaining_batches
            eta_h, eta_m = divmod(int(eta_seconds) // 60, 60)
            print(f"  {batches_done_this_run}/{total_batches} batches | "
                  f"scored {total_done:,} | errors {errors_logged} | "
                  f"elapsed {int(elapsed)//60}m | ETA {eta_h}h{eta_m:02d}m")

            # Checkpoint
            if batches_since_checkpoint >= CHECKPOINT_INTERVAL:
                frames = [f for f in [checkpoint_df, pd.DataFrame(new_results)] if len(f) > 0]
                combined = pd.concat(frames, ignore_index=True) if frames else checkpoint_df
                save_checkpoint(combined)
                checkpoint_df = combined
                completed_ids = set(checkpoint_df["row_id"].tolist())
                new_results = []
                batches_since_checkpoint = 0

            # Sleep for rate limiting
            time.sleep(SLEEP_BETWEEN_CALLS)

    finally:
        # Save any unsaved results on exit (Ctrl+C, crash, or completion)
        if new_results:
            frames = [f for f in [checkpoint_df, pd.DataFrame(new_results)] if len(f) > 0]
            combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(new_results)
            save_checkpoint(combined)
        else:
            print("No new results to save.")

    final_count = len(checkpoint_df) + len(new_results)
    print(f"\nDone. Scored {final_count:,}/{len(df):,}.")
    print(f"Errors/skipped: {errors_logged}")
    if final_count < len(df):
        print(f"Missing {len(df) - final_count:,} rows. "
              f"Re-run to attempt those.")


if __name__ == "__main__":
    main()
