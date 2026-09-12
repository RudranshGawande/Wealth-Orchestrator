"""Decision Ranker Module for evaluating, ranking, and selecting optimal financial plans."""

from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from src.simulation_engine import (
    FinancialSimulator,
    calculate_amount_safe_to_pay,
    calculate_earliest_date_for_full_payment,
)

logger = logging.getLogger(__name__)


def parse_user_preferences(profile_row: pd.Series) -> Dict[str, Any]:
    """Parses payment methods, categories, and limits from a financial profile row."""
    methods_str = str(profile_row.get("payment_methods_user_will_consider", ""))
    methods = [m.strip() for m in methods_str.split("|") if m.strip()]

    max_inst_months = profile_row.get("max_installment_months")
    try:
        max_inst_months = float(max_inst_months) if pd.notna(max_inst_months) else None
    except (ValueError, TypeError):
        max_inst_months = None

    protect_cats = [
        c.strip()
        for c in str(profile_row.get("expense_categories_to_protect", "")).split("|")
        if c.strip()
    ]
    reduce_cats = [
        c.strip()
        for c in str(
            profile_row.get("expense_categories_user_is_willing_to_reduce", "")
        ).split("|")
        if c.strip()
    ]
    stop_cats = [
        c.strip()
        for c in str(
            profile_row.get("expense_categories_user_is_willing_to_stop", "")
        ).split("|")
        if c.strip()
    ]

    return {
        "payment_methods": methods,
        "max_installment_months": max_inst_months,
        "protected_categories": protect_cats,
        "reducible_categories": reduce_cats,
        "stoppable_categories": stop_cats,
    }


def find_candidate_spending_changes(
    df_events: pd.DataFrame, user_id: str, prefs: Dict[str, Any]
) -> List[List[str]]:
    """Identifies permitted spending change options (single or pairs) for non-protected events."""
    user_events = df_events[
        (df_events["user_id"] == user_id)
        & (df_events["status"].isin(["settled", "pending", "scheduled"]))
        & (df_events["direction"] == "debit")
    ]

    single_changes = []
    for _, row in user_events.iterrows():
        event_id = str(row["event_id"])
        cat = str(row.get("category", ""))
        flex = str(row.get("flexibility", ""))

        if cat in prefs["protected_categories"]:
            continue

        if flex == "stoppable" and cat in prefs["stoppable_categories"]:
            single_changes.append([f"stop:{event_id}"])
        elif flex == "reducible" and cat in prefs["reducible_categories"]:
            min_amt = row.get("minimum_allowed_amount")
            if pd.notna(min_amt):
                single_changes.append([f"reduce_to:{event_id}:{float(min_amt):g}"])

    candidates = [[]] + single_changes  # [] means no spending changes
    return candidates


def fmt_num(val: float) -> str:
    """Formats numeric floats as clean strings avoiding scientific notation."""
    val = float(val)
    if val.is_integer():
        return str(int(val))
    return f"{val:.2f}".rstrip("0").rstrip(".")


def generate_candidate_plans(
    simulator: FinancialSimulator,
    req_row: pd.Series,
    prefs: Dict[str, Any],
    df_payment_options: pd.DataFrame,
    spending_changes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Generates all eligible safe candidate payment plans for a request given spending_changes."""
    if spending_changes is None:
        spending_changes = []

    user_id = str(req_row["user_id"])
    request_id = str(req_row["request_id"])
    req_date = str(req_row["request_date"])
    req_amt = float(req_row["requested_amount"])
    deadline = str(req_row["desired_completion_date"])
    allows_partial = str(req_row.get("allows_partial_payment", "")).lower() == "true"

    user_methods = prefs["payment_methods"]
    max_inst_months = prefs["max_installment_months"]

    amount_safe = calculate_amount_safe_to_pay(
        simulator, user_id, req_date, req_amt, spending_changes
    )
    earliest_full_date = calculate_earliest_date_for_full_payment(
        simulator, user_id, req_date, req_amt, spending_changes
    )

    candidate_plans = []

    # 1. full_payment today
    if "full_payment" in user_methods and amount_safe >= req_amt:
        candidate_plans.append(
            {
                "recommended_payment_method": "full_payment",
                "affordability_status": (
                    "affordable_now" if not spending_changes else "affordable_with_plan"
                ),
                "amount_safe_to_pay": req_amt,
                "payment_plan": f"{req_date}:{fmt_num(req_amt)}",
                "earliest_date_for_full_payment": req_date,
                "spending_changes_needed": (
                    "|".join(spending_changes) if spending_changes else "none"
                ),
                "total_payable_amount": req_amt,
                "first_payment_date": req_date,
                "completion_date": req_date,
                "payments_count": 1,
                "option_id": "",
                "has_spending_changes": len(spending_changes) > 0,
            }
        )

    # 2. partial_payment (2 payments: today amount_safe, remaining on earliest_full_date)
    if (
        allows_partial
        and "partial_payment" in user_methods
        and 0 < amount_safe < req_amt
        and earliest_full_date
        and earliest_full_date <= deadline
    ):
        rem_amt = round(req_amt - amount_safe, 2)
        candidate_plans.append(
            {
                "recommended_payment_method": "partial_payment",
                "affordability_status": "affordable_with_plan",
                "amount_safe_to_pay": amount_safe,
                "payment_plan": f"{req_date}:{fmt_num(amount_safe)}|{earliest_full_date}:{fmt_num(rem_amt)}",
                "earliest_date_for_full_payment": earliest_full_date,
                "spending_changes_needed": (
                    "|".join(spending_changes) if spending_changes else "none"
                ),
                "total_payable_amount": req_amt,
                "first_payment_date": req_date,
                "completion_date": earliest_full_date,
                "payments_count": 2,
                "option_id": "",
                "has_spending_changes": len(spending_changes) > 0,
            }
        )

    # 3. installments
    if "installments" in user_methods and not df_payment_options.empty:
        opts = df_payment_options[
            (df_payment_options["request_id"] == request_id)
            & (df_payment_options["payment_method"] == "installments")
        ]

        for _, opt_row in opts.iterrows():
            opt_id = str(opt_row["payment_option_id"])
            n_payments = int(opt_row["number_of_payments"])
            pmt_amt = float(opt_row["payment_amount"])
            first_date = str(opt_row["first_payment_date"])
            freq_days = (
                int(opt_row["payment_frequency_days"])
                if pd.notna(opt_row["payment_frequency_days"])
                else 30
            )
            tot_payable = float(opt_row["total_payable_amount"])

            if max_inst_months is not None and (n_payments > max_inst_months * 1.5):
                continue

            first_dt = datetime.strptime(first_date, "%Y-%m-%d")
            inst_payments = {}
            plan_parts = []
            last_date = first_date
            for k in range(n_payments):
                p_dt = first_dt + timedelta(days=k * freq_days)
                p_str = p_dt.strftime("%Y-%m-%d")
                inst_payments[p_str] = inst_payments.get(p_str, 0.0) + pmt_amt
                plan_parts.append(f"{p_str}:{fmt_num(pmt_amt)}")
                last_date = p_str

            _, _, is_safe = simulator.simulate_user_balance(
                user_id, req_date, 90, spending_changes, inst_payments
            )

            if is_safe:
                candidate_plans.append(
                    {
                        "recommended_payment_method": "installments",
                        "affordability_status": "affordable_with_plan",
                        "amount_safe_to_pay": amount_safe,
                        "payment_plan": "|".join(plan_parts),
                        "earliest_date_for_full_payment": earliest_full_date or first_date,
                        "spending_changes_needed": (
                            "|".join(spending_changes) if spending_changes else "none"
                        ),
                        "total_payable_amount": tot_payable,
                        "first_payment_date": first_date,
                        "completion_date": last_date,
                        "payments_count": n_payments,
                        "option_id": opt_id,
                        "has_spending_changes": len(spending_changes) > 0,
                    }
                )

    # 4. wait (pay full amount on earliest_date_for_full_payment)
    if "full_payment" in user_methods and earliest_full_date:
        status = "affordable_later" if earliest_full_date > deadline else "affordable_with_plan"
        candidate_plans.append(
            {
                "recommended_payment_method": "wait",
                "affordability_status": status,
                "amount_safe_to_pay": amount_safe,
                "payment_plan": f"{earliest_full_date}:{fmt_num(req_amt)}",
                "earliest_date_for_full_payment": earliest_full_date,
                "spending_changes_needed": (
                    "|".join(spending_changes) if spending_changes else "none"
                ),
                "total_payable_amount": req_amt,
                "first_payment_date": earliest_full_date,
                "completion_date": earliest_full_date,
                "payments_count": 1,
                "option_id": "",
                "has_spending_changes": len(spending_changes) > 0,
            }
        )

    return candidate_plans


def rank_and_select_best_plan(
    candidate_plans: List[Dict[str, Any]], deadline: str
) -> Dict[str, Any]:
    """
    Ranks candidate plans according to the strict 6-tier order:
    1. Completes request by desired_completion_date
    2. Requires no spending changes
    3. Minimizes total amount paid
    4. Starts payment earlier
    5. Uses fewer payments
    6. Lowest payment_option_id (tie-breaker)
    """
    if not candidate_plans:
        return {}

    def ranking_key(plan: Dict[str, Any]):
        completes_by_deadline = plan["completion_date"] <= deadline
        return (
            not completes_by_deadline,  # False (0) is better than True (1)
            plan["has_spending_changes"],  # False (0) is better than True (1)
            plan["total_payable_amount"],  # Min total cost
            plan["first_payment_date"],  # Earlier start date
            plan["payments_count"],  # Fewer payments
            plan["option_id"],  # Tie breaker
        )

    sorted_plans = sorted(candidate_plans, key=ranking_key)
    return sorted_plans[0]


def evaluate_request_recommendation(
    simulator: FinancialSimulator,
    req_row: pd.Series,
    df_profiles: pd.DataFrame,
    df_events: pd.DataFrame,
    df_payment_options: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Evaluates all possible plans and spending changes for a request and selects the optimal recommendation.
    """
    user_id = str(req_row["user_id"])
    req_date = str(req_row["request_date"])
    req_amt = float(req_row["requested_amount"])
    deadline = str(req_row["desired_completion_date"])

    prof_matches = df_profiles[df_profiles["user_id"] == user_id]
    if prof_matches.empty:
        raise ValueError(f"User {user_id} not found in profiles.")

    prefs = parse_user_preferences(prof_matches.iloc[0])
    spending_change_candidates = find_candidate_spending_changes(
        df_events, user_id, prefs
    )

    all_candidates = []
    for sc in spending_change_candidates:
        plans = generate_candidate_plans(
            simulator, req_row, prefs, df_payment_options, spending_changes=sc
        )
        all_candidates.extend(plans)

    best_plan = rank_and_select_best_plan(all_candidates, deadline)

    if not best_plan:
        amount_safe = calculate_amount_safe_to_pay(simulator, user_id, req_date, req_amt)
        return {
            "request_id": str(req_row["request_id"]),
            "amount_safe_to_pay": amount_safe,
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": "",
            "spending_changes_needed": "none",
        }

    return {
        "request_id": str(req_row["request_id"]),
        "amount_safe_to_pay": best_plan["amount_safe_to_pay"],
        "affordability_status": best_plan["affordability_status"],
        "recommended_payment_method": best_plan["recommended_payment_method"],
        "payment_plan": best_plan["payment_plan"],
        "earliest_date_for_full_payment": best_plan["earliest_date_for_full_payment"],
        "spending_changes_needed": best_plan["spending_changes_needed"],
    }
