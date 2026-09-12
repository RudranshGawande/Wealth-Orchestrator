"""90-Day Deterministic Cash Flow Simulation Engine for Buy or Wait financial agent."""

from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data_loader import load_all_datasets

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


class CurrencyConverter:
    """Helper to convert foreign currency values to user's home currency based on event date."""

    def __init__(self, df_rates: pd.DataFrame):
        self.df_rates = df_rates.copy() if df_rates is not None else pd.DataFrame()
        if not self.df_rates.empty:
            self.df_rates["rate_date"] = pd.to_datetime(self.df_rates["rate_date"])
            self.df_rates["rate"] = self.df_rates["rate"].astype(float)

    def convert(
        self, amount: float, from_curr: str, to_curr: str, event_date_str: str
    ) -> float:
        """Converts amount from from_curr to to_curr using rate on or before event_date_str."""
        if not amount or pd.isna(amount) or float(amount) == 0.0:
            return 0.0

        if from_curr == to_curr or pd.isna(from_curr) or pd.isna(to_curr):
            return float(amount)

        if self.df_rates.empty:
            return float(amount)

        target_date = pd.to_datetime(event_date_str)

        # Direct rate lookup (from_curr -> to_curr)
        direct = self.df_rates[
            (self.df_rates["from_currency"] == from_curr)
            & (self.df_rates["to_currency"] == to_curr)
        ]
        if not direct.empty:
            match = direct[direct["rate_date"] <= target_date]
            best_row = (
                match.sort_values("rate_date", ascending=False).iloc[0]
                if not match.empty
                else direct.sort_values("rate_date", ascending=True).iloc[0]
            )
            return float(amount) * float(best_row["rate"])

        # Inverse rate lookup (to_curr -> from_curr)
        inverse = self.df_rates[
            (self.df_rates["from_currency"] == to_curr)
            & (self.df_rates["to_currency"] == from_curr)
        ]
        if not inverse.empty:
            match = inverse[inverse["rate_date"] <= target_date]
            best_row = (
                match.sort_values("rate_date", ascending=False).iloc[0]
                if not match.empty
                else inverse.sort_values("rate_date", ascending=True).iloc[0]
            )
            rate = float(best_row["rate"])
            return float(amount) / rate if rate != 0 else float(amount)

        # Bridge via USD or EUR
        for bridge in ["USD", "EUR"]:
            if from_curr != bridge and to_curr != bridge:
                rate1 = self.convert(1.0, from_curr, bridge, event_date_str)
                if rate1 != 1.0:
                    rate2 = self.convert(1.0, bridge, to_curr, event_date_str)
                    return float(amount) * rate1 * rate2

        return float(amount)


class FinancialSimulator:
    """90-day deterministic cash flow simulator."""

    def __init__(self, datasets: Optional[Dict[str, pd.DataFrame]] = None):
        if datasets is None:
            datasets = load_all_datasets()

        self.datasets = datasets
        self.df_profiles = datasets.get("financial_profiles", pd.DataFrame())
        self.df_events = datasets.get("financial_events", pd.DataFrame()).copy()
        self.df_rates = datasets.get("exchange_rates", pd.DataFrame())
        self.df_messages = datasets.get("messages", pd.DataFrame())
        self.df_images = datasets.get("images", pd.DataFrame())

        self.currency_converter = CurrencyConverter(self.df_rates)
        self._preprocess_events_with_messages()

    def _preprocess_events_with_messages(self):
        """Applies extracted message actions/amendments to self.df_events."""
        if self.df_messages.empty or "extracted_action" not in self.df_messages.columns:
            return

        for _, row in self.df_messages.iterrows():
            event_id = row.get("related_event_id")
            if not event_id or pd.isna(event_id):
                continue

            action = row.get("extracted_action")
            new_amount = row.get("extracted_new_amount")
            new_date = row.get("extracted_new_date")

            event_matches = self.df_events[self.df_events["event_id"] == event_id]
            if event_matches.empty:
                continue

            e_idx = event_matches.index[0]
            if action == "cancelled":
                self.df_events.at[e_idx, "status"] = "cancelled"
            elif action == "settled":
                self.df_events.at[e_idx, "status"] = "settled"
            elif action == "amended_amount" and new_amount is not None:
                self.df_events.at[e_idx, "amount"] = float(new_amount)
            elif action == "postponed" and new_date:
                self.df_events.at[e_idx, "settlement_date"] = str(new_date)
                self.df_events.at[e_idx, "event_date"] = str(new_date)

    def _get_projected_recurring_events(
        self, user_id: str, start_dt: datetime, days: int = 90
    ) -> List[Dict[str, Any]]:
        """Detects recurring historical patterns for user_id and projects events across 90-day window."""
        if not hasattr(self, "_user_projected_cache"):
            self._user_projected_cache = {}

        cache_key = (user_id, start_dt.strftime("%Y-%m-%d"), days)
        if cache_key in self._user_projected_cache:
            return self._user_projected_cache[cache_key]

        end_dt = start_dt + timedelta(days=days)

        user_events = self.df_events[
            (self.df_events["user_id"] == user_id)
            & (self.df_events["status"].isin(["settled", "pending", "scheduled"]))
            & (self.df_events["direction"] != "non_cash")
        ].copy()

        if user_events.empty:
            return []

        user_events["dt"] = pd.to_datetime(user_events["event_date"])
        hist_events = user_events[user_events["dt"] <= start_dt]

        projected = []

        for (cat, desc, dirn), group in hist_events.groupby(
            ["category", "description", "direction"]
        ):
            sorted_g = group.sort_values("dt")
            dates = sorted_g["dt"].tolist()
            if len(dates) < 2:
                continue

            amounts = sorted_g["amount"].dropna().tolist()
            last_amt = float(amounts[-1]) if amounts else 0.0
            curr = str(sorted_g.iloc[-1]["currency"])
            flex = str(sorted_g.iloc[-1].get("flexibility", "fixed"))
            event_type = str(sorted_g.iloc[-1].get("event_type", "expense"))

            doms = [d.day for d in dates]
            avg_dom = int(round(float(np.mean(doms))))
            diffs = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            avg_diff = float(np.mean(diffs))

            projected_dates = []
            if 25 <= avg_diff <= 35 or len(set(doms)) <= 2:
                # Monthly recurring event
                curr_y = start_dt.year
                curr_m = start_dt.month
                for _ in range(4):
                    try:
                        target_dom = min(avg_dom, 28)
                        p_dt = datetime(curr_y, curr_m, target_dom)
                        if start_dt <= p_dt <= end_dt:
                            projected_dates.append(p_dt)
                    except ValueError:
                        pass
                    curr_m += 1
                    if curr_m > 12:
                        curr_m = 1
                        curr_y += 1
            elif 5 <= avg_diff <= 25:
                # Interval event
                last_dt = dates[-1]
                next_dt = last_dt + timedelta(days=int(round(avg_diff)))
                while next_dt <= end_dt:
                    if next_dt >= start_dt:
                        projected_dates.append(next_dt)
                    next_dt += timedelta(days=int(round(avg_diff)))

            for p_dt in projected_dates:
                p_date_str = p_dt.strftime("%Y-%m-%d")
                projected.append(
                    {
                        "event_id": f"proj_{cat}_{p_date_str}",
                        "user_id": user_id,
                        "event_type": event_type,
                        "description": desc,
                        "category": cat,
                        "direction": dirn,
                        "amount": last_amt,
                        "currency": curr,
                        "event_date": p_date_str,
                        "settlement_date": p_date_str,
                        "status": "scheduled",
                        "flexibility": flex,
                    }
                )

        self._user_projected_cache[cache_key] = projected
        return projected

    def simulate_user_balance(
        self,
        user_id: str,
        start_date: str,
        days: int = 90,
        spending_changes: Optional[List[str]] = None,
        additional_payments: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[float], Dict[str, float], bool]:
        """
        Simulates user's day-by-day cash balance over `days` days starting on `start_date`.

        Args:
            user_id: User identifier.
            start_date: YYYY-MM-DD start date.
            days: Number of days to simulate (default 90).
            spending_changes: List of change directives e.g. ['stop:event_06', 'reduce_to:event_08:1000'].
            additional_payments: Dict of {'YYYY-MM-DD': amount} for extra payments in home currency.

        Returns:
            Tuple: (daily_balances_list, date_balance_dict, is_safe_bool)
        """
        if not hasattr(self, "_daily_flow_cache"):
            self._daily_flow_cache = {}

        sc_key = tuple(sorted(spending_changes)) if spending_changes else ()
        flow_key = (user_id, start_date, days, sc_key)

        if flow_key in self._daily_flow_cache:
            start_bal, min_keep, daily_cash_flow = self._daily_flow_cache[flow_key]
        else:
            user_prof = self.df_profiles[self.df_profiles["user_id"] == user_id]
            if user_prof.empty:
                raise ValueError(f"User '{user_id}' not found in financial_profiles.csv")

            prof_row = user_prof.iloc[0]
            home_curr = str(prof_row["home_currency"])
            start_bal = float(prof_row["current_available_balance"])
            min_keep = float(prof_row["minimum_balance_to_keep"])

            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = start_dt + timedelta(days=days - 1)

            # Parse spending changes
            stopped_events = set()
            reduced_events = {}
            if spending_changes:
                for sc in spending_changes:
                    parts = sc.split(":")
                    if parts[0] == "stop" and len(parts) >= 2:
                        stopped_events.add(parts[1])
                    elif parts[0] == "reduce_to" and len(parts) >= 3:
                        try:
                            reduced_events[parts[1]] = float(parts[2])
                        except ValueError:
                            pass

            # Gather events
            user_events = self.df_events[self.df_events["user_id"] == user_id].copy()
            proj_events = self._get_projected_recurring_events(user_id, start_dt, days)
            df_proj = pd.DataFrame(proj_events)

            if not df_proj.empty:
                all_events = pd.concat([user_events, df_proj], ignore_index=True)
            else:
                all_events = user_events

            # Build daily cash flow map
            daily_cash_flow: Dict[str, float] = {}
            for i in range(days):
                d_str = (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
                daily_cash_flow[d_str] = 0.0

            for _, row in all_events.iterrows():
                st = str(row.get("status", "settled"))
                if st in ["cancelled", "failed"]:
                    continue

                dirn = str(row.get("direction"))
                if dirn == "non_cash":
                    continue

                e_date = str(row.get("settlement_date") or row.get("event_date") or "")
                if not e_date or e_date < start_date or e_date > end_dt.strftime("%Y-%m-%d"):
                    continue

                event_id = str(row.get("event_id"))
                if event_id in stopped_events:
                    continue

                raw_amt = float(row.get("amount", 0.0)) if pd.notna(row.get("amount")) else 0.0
                if event_id in reduced_events:
                    raw_amt = min(raw_amt, reduced_events[event_id])

                curr = str(row.get("currency", home_curr))
                converted_amt = self.currency_converter.convert(raw_amt, curr, home_curr, e_date)

                if dirn == "debit" and st in ["settled", "pending", "scheduled"]:
                    daily_cash_flow[e_date] -= converted_amt
                elif dirn == "credit":
                    e_type = str(row.get("event_type", ""))
                    if st == "settled" or e_type == "income":
                        daily_cash_flow[e_date] += converted_amt

            self._daily_flow_cache[flow_key] = (start_bal, min_keep, daily_cash_flow)

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        current_bal = start_bal
        daily_balances = []
        date_balance_map = {}

        for i in range(days):
            d_str = (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
            current_bal += daily_cash_flow.get(d_str, 0.0)

            if additional_payments and d_str in additional_payments:
                current_bal -= float(additional_payments[d_str])

            daily_balances.append(current_bal)
            date_balance_map[d_str] = current_bal

        is_safe = all(bal >= min_keep for bal in daily_balances)
        return daily_balances, date_balance_map, is_safe


def calculate_amount_safe_to_pay(
    simulator: FinancialSimulator,
    user_id: str,
    request_date: str,
    requested_amount: float,
    spending_changes: Optional[List[str]] = None,
) -> float:
    """
    Finds the largest safe payment amount today (0 to requested_amount) that keeps all 90 days >= minimum_balance_to_keep.
    """
    requested_amount = float(requested_amount)
    if requested_amount <= 0:
        return 0.0

    # 1. Test full requested amount
    _, _, is_safe_full = simulator.simulate_user_balance(
        user_id, request_date, 90, spending_changes, {request_date: requested_amount}
    )
    if is_safe_full:
        return requested_amount

    # 2. Test 0 amount
    _, _, is_safe_zero = simulator.simulate_user_balance(
        user_id, request_date, 90, spending_changes, {request_date: 0.0}
    )
    if not is_safe_zero:
        return 0.0

    # 3. Binary search for maximum safe payment
    low = 0.0
    high = requested_amount
    best_safe = 0.0

    for _ in range(25):
        mid = (low + high) / 2.0
        _, _, safe = simulator.simulate_user_balance(
            user_id, request_date, 90, spending_changes, {request_date: mid}
        )
        if safe:
            best_safe = mid
            low = mid
        else:
            high = mid

    safe_amount = round(best_safe, 2)

    # Final safety check on rounded amount
    _, _, safe_check = simulator.simulate_user_balance(
        user_id, request_date, 90, spending_changes, {request_date: safe_amount}
    )
    if not safe_check and safe_amount > 0:
        safe_amount = max(0.0, safe_amount - 0.01)

    return safe_amount


def calculate_earliest_date_for_full_payment(
    simulator: FinancialSimulator,
    user_id: str,
    request_date: str,
    requested_amount: float,
    spending_changes: Optional[List[str]] = None,
) -> str:
    """
    Finds the earliest date (within 0 to 90 days from request_date) where paying full requested_amount is safe.
    """
    start_dt = datetime.strptime(request_date, "%Y-%m-%d")

    for offset in range(91):
        test_date = (start_dt + timedelta(days=offset)).strftime("%Y-%m-%d")
        _, _, is_safe = simulator.simulate_user_balance(
            user_id, request_date, 90, spending_changes, {test_date: float(requested_amount)}
        )
        if is_safe:
            return test_date

    return ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Initializing Financial Simulator...")
    datasets = load_all_datasets()
    simulator = FinancialSimulator(datasets)

    df_requests = datasets.get("requests", pd.DataFrame())
    print("\n--- Running 90-Day Cash Flow Simulation on Sample Requests ---")
    
    sample_rows = df_requests.head(5)
    for idx, row in sample_rows.iterrows():
        req_id = row["request_id"]
        u_id = row["user_id"]
        req_date = row["request_date"]
        req_amt = float(row["requested_amount"])

        safe_amt = calculate_amount_safe_to_pay(simulator, u_id, req_date, req_amt)
        earliest_date = calculate_earliest_date_for_full_payment(
            simulator, u_id, req_date, req_amt
        )

        print(f"\nRequest ID: {req_id} (User: {u_id})")
        print(f"  Request Date: {req_date}, Requested Amount: {req_amt:,.2f}")
        print(f"  Amount Safe to Pay Today: {safe_amt:,.2f}")
        print(f"  Earliest Date for Full Payment: '{earliest_date}'")
