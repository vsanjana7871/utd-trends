"""
EZBreZ — Step 3: score a user + money-move and explain why.

Returns:
  history_score  (0-100)  + per-factor breakdown (payment/util/age/inquiries/mix)
  ezbreze_score  (0-100)  + verdict + the drivers behind the number

The breakdown dict is what you pass to Grok to generate the friendly
plain-English message ("Your utilization is what's holding your score back...").
The number itself comes from the ML model, not the LLM.
"""
import joblib
import pandas as pd
from generate_data import q_payment, q_util, q_age, q_inq, q_mix

_history = joblib.load("model_history.pkl")
_ezbreze = joblib.load("model_ezbreze.pkl")


def verdict(score):
    if score >= 75:  return "Go for it"
    if score >= 55:  return "OK, but watch the buffer"
    if score >= 35:  return "Risky — maybe wait"
    return "Don't do it right now"


def score_user(profile: dict):
    row = pd.DataFrame([profile])

    hist = float(min(100, max(0, _history["model"].predict(row[_history["features"]])[0])))
    ez = float(min(100, max(0, _ezbreze["model"].predict(row[_ezbreze["features"]])[0])))

    # per-factor health of the credit profile (drives the history explanation)
    factors = {
        "payment history (35%)":     round(float(q_payment(profile["payment_on_time_pct"])), 1),
        "amounts owed (30%)":        round(float(q_util(profile["credit_utilization"])), 1),
        "length of history (15%)":   round(float(q_age(profile["credit_age_months"])), 1),
        "new credit (10%)":          round(float(q_inq(profile["recent_inquiries"])), 1),
        "credit mix (10%)":          round(float(q_mix(profile["credit_account_types"])), 1),
    }
    weakest = min(factors, key=factors.get)

    disposable = profile["monthly_income"] - profile["monthly_expenses"]
    return {
        "history_score": round(hist),
        "history_factors": factors,
        "weakest_factor": weakest,
        "ezbreze_score": round(ez),
        "verdict": verdict(ez),
        "context": {
            "monthly_disposable": round(disposable),
            "purchase_amount": profile["purchase_amount"],
            "months_to_afford": round(profile["purchase_amount"] / max(disposable, 1), 1),
            "essential": bool(profile["is_essential"]),
        },
    }


if __name__ == "__main__":
    # Example: a first-time credit user eyeing a $900 laptop, paying on credit
    demo = {
        "payment_on_time_pct": 0.92,
        "credit_utilization": 0.41,
        "credit_age_months": 8,
        "recent_inquiries": 2,
        "credit_account_types": 1,
        "credit_limit": 1000,
        "monthly_income": 2600,
        "monthly_expenses": 2050,
        "current_savings": 1200,
        "purchase_amount": 900,
        "pay_upfront": 0,
        "is_essential": 0,
    }
    from pprint import pprint
    pprint(score_user(demo))