"""EZBreZ — Step 2: train the two models."""
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from generate_data import sample_population

HISTORY_FEATURES = [
    "payment_on_time_pct", "credit_utilization", "credit_age_months",
    "recent_inquiries", "credit_account_types",
]

EZBREZE_FEATURES = [
    "payment_on_time_pct", "credit_utilization", "credit_age_months",
    "recent_inquiries", "credit_account_types", "credit_limit",
    "monthly_income", "monthly_expenses", "current_savings",
    "purchase_amount", "pay_upfront", "is_essential",
]


def train_one(df, features, target, name):
    X, y = df[features], df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
        random_state=7,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(f"\n=== {name} ===")
    print(f"  MAE : {mean_absolute_error(yte, pred):.2f}")
    print(f"  R^2 : {r2_score(yte, pred):.3f}")
    joblib.dump({"model": model, "features": features}, f"model_{name}.pkl")
    return model


if __name__ == "__main__":
    df = sample_population(14000)
    train_one(df, HISTORY_FEATURES, "history_score", "history")
    train_one(df, EZBREZE_FEATURES, "ezbreze_score", "ezbreze")
    print("\nsaved: model_history.pkl, model_ezbreze.pkl")