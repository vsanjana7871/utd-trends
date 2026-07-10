"""EZBreZ — Step 1: synthetic-but-grounded dataset."""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def q_payment(on_time_pct):
    return np.clip(100 * (on_time_pct ** 1.5), 0, 100)

def q_util(util):
    return np.clip(100 - (np.maximum(util - 0.10, 0) * 220), 0, 100)

def q_age(age_months):
    return np.clip(100 * (1 - np.exp(-age_months / 40)), 0, 100)

def q_inq(inquiries):
    return np.clip(100 - inquiries * 18, 0, 100)

def q_mix(n_types):
    return np.clip(100 * (1 - np.exp(-n_types / 1.8)), 0, 100)


def history_score(row):
    return (0.35 * q_payment(row["payment_on_time_pct"]) +
            0.30 * q_util(row["credit_utilization"]) +
            0.15 * q_age(row["credit_age_months"]) +
            0.10 * q_inq(row["recent_inquiries"]) +
            0.10 * q_mix(row["credit_account_types"]))


def ezbreze_score(row):
    disposable = row["monthly_income"] - row["monthly_expenses"]
    amt = row["purchase_amount"]
    afford = disposable / (amt + 1e-6)
    afford_q = np.clip(np.tanh(afford) * 100, 0, 100)
    target_buffer = 3 * row["monthly_expenses"]
    buffer_after = row["current_savings"] - (amt if row["pay_upfront"] else 0)
    buffer_q = np.clip((buffer_after / (target_buffer + 1e-6)) * 100, 0, 100)
    new_bal = row["credit_utilization"] + (0 if row["pay_upfront"]
                                           else amt / (row["credit_limit"] + 1e-6))
    util_q = q_util(new_bal)
    risk_q = q_payment(row["payment_on_time_pct"])
    base = 0.30 * afford_q + 0.28 * buffer_q + 0.24 * util_q + 0.18 * risk_q
    base += 8 if row["is_essential"] else -6
    return np.clip(base, 0, 100)


def sample_population(n=12000):
    df = pd.DataFrame({
        "payment_on_time_pct": np.clip(RNG.beta(6, 1.4, n), 0, 1),
        "credit_utilization":  np.clip(RNG.gamma(2.0, 0.16, n), 0, 1.5),
        "credit_age_months":   RNG.integers(1, 130, n),
        "recent_inquiries":    RNG.poisson(1.4, n),
        "credit_account_types": RNG.integers(1, 5, n),
        "credit_limit":        RNG.choice([500, 1000, 2000, 3500, 6000], n),
        "monthly_income":      RNG.normal(3200, 900, n).clip(1200, 9000),
        "monthly_expenses":    RNG.normal(2300, 700, n).clip(800, 7000),
        "current_savings":     RNG.gamma(2.0, 1500, n).clip(0, 30000),
        "purchase_amount":     RNG.gamma(1.8, 260, n).clip(20, 6000),
        "pay_upfront":         RNG.integers(0, 2, n),
        "is_essential":        RNG.integers(0, 2, n),
    })
    df["monthly_expenses"] = np.minimum(df["monthly_expenses"], df["monthly_income"] * 0.95)
    df["history_score"] = df.apply(history_score, axis=1)
    df["ezbreze_score"] = df.apply(ezbreze_score, axis=1)
    df["history_score"] = (df["history_score"] + RNG.normal(0, 3.5, n)).clip(0, 100)
    df["ezbreze_score"] = (df["ezbreze_score"] + RNG.normal(0, 4.0, n)).clip(0, 100)
    return df


if __name__ == "__main__":
    data = sample_population()
    data.to_csv("ezbrez_training_data.csv", index=False)
    print("rows:", len(data))