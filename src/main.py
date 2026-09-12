"""Main End-to-End Orchestrator for Buy or Wait Financial Agent."""

import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.data_loader import load_all_datasets
from src.decision_ranker import evaluate_request_recommendation
from src.llm_client import LLMTracker
from src.multimodal_processor import (
    process_message_amendments,
    process_missing_event_amounts,
)
from src.simulation_engine import FinancialSimulator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_explanation(
    tracker: LLMTracker,
    rec: Dict[str, Any],
    req_row: pd.Series,
    prof_row: pd.Series,
    model_name: str = "gemini-3.6-flash",
) -> str:
    """Generates a concise 1-2 sentence decision_explanation via Gemini API with deterministic fallback."""
    home_curr = str(prof_row.get("home_currency", ""))
    min_keep = float(prof_row.get("minimum_balance_to_keep", 0.0))
    req_amt = float(req_row["requested_amount"])
    req_date = str(req_row["request_date"])
    deadline = str(req_row["desired_completion_date"])

    method = rec["recommended_payment_method"]
    status = rec["affordability_status"]
    earliest_date = rec.get("earliest_date_for_full_payment", "")
    safe_amt = rec.get("amount_safe_to_pay", 0.0)

    prompt = (
        "You are an expert AI financial orchestrator.\n"
        "Generate a concise 1 to 2 sentence 'decision_explanation' for the user's financial request.\n"
        "RULES:\n"
        "1. Output ONLY 1-2 clear, grounded sentences. No extra text or markdown formatting.\n"
        "2. Include exact monetary amounts, currency codes, and dates from the provided data.\n"
        "3. Explicitly mention keeping the user's minimum balance protected.\n\n"
        f"Context:\n"
        f"- Request ID: {rec['request_id']}\n"
        f"- Home Currency: {home_curr}\n"
        f"- Requested Amount: {req_amt:g}\n"
        f"- Request Date: {req_date}\n"
        f"- Deadline: {deadline}\n"
        f"- Minimum Balance to Keep: {min_keep:g}\n"
        f"- Recommended Method: {method}\n"
        f"- Affordability Status: {status}\n"
        f"- Payment Plan: {rec['payment_plan']}\n"
        f"- Amount Safe to Pay Today: {safe_amt:g}\n"
        f"- Earliest Date for Full Payment: {earliest_date}\n"
        f"- Spending Changes: {rec['spending_changes_needed']}\n"
    )

    if (
        not tracker.client
        or tracker.api_key == "your-key-here"
        or getattr(tracker, "_api_key_invalid", False)
        or getattr(tracker, "_api_quota_exceeded", False)
    ):
        pass
    else:
        try:
            explanation = tracker.ask_gemini(prompt, model_name=model_name)
            if explanation and isinstance(explanation, str) and explanation.strip():
                cleaned = explanation.strip().replace("\n", " ")
                return cleaned
        except Exception as e:
            err_str = str(e)
            if "API_KEY_INVALID" in err_str or "400" in err_str:
                tracker._api_key_invalid = True
                logger.warning("Disabling further LLM explanation calls due to invalid/placeholder API key.")
            elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                tracker._api_quota_exceeded = True
                logger.warning("Quota/rate limit reached. Disabling further LLM explanation calls to use grounded deterministic templates.")
            else:
                logger.warning(f"LLM call for decision explanation failed: {e}")

def fmt_num(val: float) -> str:
    """Formats numeric floats as clean strings avoiding scientific notation."""
    val = float(val)
    if val.is_integer():
        return str(int(val))
    return f"{val:.2f}".rstrip("0").rstrip(".")


    # Grounded fallback templates matching benchmark decision style
    if method == "full_payment":
        return f"Pay {home_curr} {fmt_num(req_amt)} today. This leaves at least {home_curr} {fmt_num(min_keep)} available over the next 90 days."
    elif method == "installments":
        return f"Use the recommended installment plan starting {req_date}. This keeps your {home_curr} {fmt_num(min_keep)} minimum balance protected."
    elif method == "wait":
        return f"Pay {home_curr} {fmt_num(req_amt)} in full on {earliest_date}. Paying earlier would take the balance below the {home_curr} {fmt_num(min_keep)} minimum."
    elif method == "partial_payment":
        return f"Pay {home_curr} {fmt_num(safe_amt)} today and the remaining amount on {earliest_date}. This keeps your {home_curr} {fmt_num(min_keep)} minimum protected."
    else:
        return f"Do not make this payment by {deadline}. None of the available options keeps the {home_curr} {fmt_num(min_keep)} minimum protected."


def main():
    logger.info("Starting Buy or Wait End-to-End Orchestrator Pipeline...")

    # 1. Ingest Datasets
    datasets = load_all_datasets()
    df_requests = datasets.get("requests", pd.DataFrame())
    df_profiles = datasets.get("financial_profiles", pd.DataFrame())
    df_events = datasets.get("financial_events", pd.DataFrame())
    df_images = datasets.get("images", pd.DataFrame())
    df_messages = datasets.get("messages", pd.DataFrame())
    df_payment_options = datasets.get("request_payment_options", pd.DataFrame())

    if df_requests.empty:
        raise RuntimeError("requests.csv is empty or missing.")

    # 2. Instantiate LLMTracker
    tracker = LLMTracker()

    # 3. Multimodal Image Processing
    logger.info("Processing missing amounts from receipt images...")
    try:
        datasets["financial_events"] = process_missing_event_amounts(
            df_events, df_images, tracker, max_process_count=3
        )
    except Exception as e:
        logger.warning(f"Image amount extraction step skipped/failed: {e}")

    # 4. Message Parsing & Safety Guardrails
    logger.info("Processing message updates & safety guardrails...")
    try:
        datasets["messages"] = process_message_amendments(
            df_messages, tracker, max_process_count=3
        )
    except Exception as e:
        logger.warning(f"Message processing step skipped/failed: {e}")

    # 5. Instantiate Financial Simulator
    logger.info("Initializing Financial Simulator...")
    simulator = FinancialSimulator(datasets)

    # 6. Evaluate all requests
    results: List[Dict[str, Any]] = []
    logger.info(f"Processing {len(df_requests)} evaluation requests...")

    for idx, req_row in df_requests.iterrows():
        user_id = str(req_row["user_id"])
        prof_matches = df_profiles[df_profiles["user_id"] == user_id]
        if prof_matches.empty:
            logger.error(f"User {user_id} not found in financial_profiles.csv")
            continue
        prof_row = prof_matches.iloc[0]

        rec = evaluate_request_recommendation(
            simulator,
            req_row,
            df_profiles,
            datasets["financial_events"],
            df_payment_options,
        )

        explanation = generate_explanation(tracker, rec, req_row, prof_row)
        rec["decision_explanation"] = explanation

        results.append(rec)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(df_requests):
            logger.info(f"Processed {idx + 1}/{len(df_requests)} requests.")

    # 7. Construct output DataFrame
    df_output = pd.DataFrame(results)

    required_columns = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ]

    df_output = df_output[required_columns]

    # 8. Assertion Checks
    logger.info("Running assertion checks on output Data...")

    df_check = df_output.merge(
        df_requests[["request_id", "requested_amount"]], on="request_id"
    )

    for idx, row in df_check.iterrows():
        req_amt = float(row["requested_amount"])
        safe_amt = float(row["amount_safe_to_pay"])
        req_id = row["request_id"]

        assert 0.0 <= safe_amt <= req_amt + 1e-5, (
            f"Assertion Failed for {req_id}: amount_safe_to_pay ({safe_amt}) "
            f"out of bounds [0, {req_amt}]"
        )
        assert row["affordability_status"] in [
            "affordable_now",
            "affordable_with_plan",
            "affordable_later",
            "not_affordable",
        ], f"Invalid affordability_status for {req_id}: {row['affordability_status']}"

        assert row["recommended_payment_method"] in [
            "full_payment",
            "partial_payment",
            "installments",
            "wait",
            "not_recommended",
        ], f"Invalid recommended_payment_method for {req_id}: {row['recommended_payment_method']}"

    logger.info("All output assertion checks PASSED successfully!")

    # 9. Export output.csv (both to root and dataset/output.csv)
    output_path = REPO_ROOT / "output.csv"
    dataset_output_path = REPO_ROOT / "dataset" / "output.csv"

    df_output.to_csv(output_path, index=False)
    df_output.to_csv(dataset_output_path, index=False)

    logger.info(f"Exported output.csv to {output_path} and {dataset_output_path}")

    # 10. Export final usage report
    usage_report_path = tracker.export_usage_report("evaluation/usage_report.md")
    logger.info(f"Exported final usage report to {usage_report_path}")

    print(f"\nPipeline Execution Complete!")
    print(f"Total Requests Evaluated: {len(df_output)}")
    print(f"Output saved at: {output_path}")
    print(f"Usage report saved at: {usage_report_path}")


if __name__ == "__main__":
    main()
