"""EZBreZ model service — Step 1. Serves score_user() over HTTP."""
from flask import Flask, request, jsonify
from flask_cors import CORS
from score import score_user

app = Flask(__name__)
CORS(app)

FIELDS = {
    "payment_on_time_pct": (float, 0.95),
    "credit_utilization":  (float, 0.30),
    "credit_age_months":   (int,   12),
    "recent_inquiries":    (int,   1),
    "credit_account_types":(int,   1),
    "credit_limit":        (float, 1000),
    "monthly_income":      (float, 2500),
    "monthly_expenses":    (float, 2000),
    "current_savings":     (float, 1000),
    "purchase_amount":     (float, 0),
    "pay_upfront":         (int,   0),
    "is_essential":        (int,   0),
}


def clean(payload):
    out = {}
    for name, (caster, default) in FIELDS.items():
        raw = payload.get(name, default)
        try:
            out[name] = caster(raw)
        except (TypeError, ValueError):
            out[name] = default
    return out


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/score")
def score():
    payload = request.get_json(silent=True) or {}
    profile = clean(payload)
    result = score_user(profile)
    result["echo"] = profile
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)