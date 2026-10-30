import os
import sys
import time

import numpy as np
import redis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

MODEL_LATENCY_MS = float(os.getenv("MODEL_LATENCY_MS", "25"))

N_USERS = int(os.getenv("N_USERS", "1000"))
FIRST_USER_ID = 1000          
FEATURE_TTL = 86400           # 1 day
PRED_TTL = 300                # 5 minutes

FEATURE_NAMES = ["orders_30d", "avg_basket", "account_age_days", "night_txn_ratio"]

KEY_PATTERNS = ["user:*", "pred:*", "stats:*", "events:*"]


def header(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def get_redis():
    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError:
        print(f"Could not connect to Redis at {REDIS_URL}.")
        print("Start one with:  docker compose up -d")
        sys.exit(1)
    return r


def reset(r, patterns=None):
    """Delete the demo keys so every run starts clean."""
    for pattern in patterns or KEY_PATTERNS:
        keys = list(r.scan_iter(match=pattern, count=1000))
        if keys:
            r.delete(*keys)


def make_dataset(n=N_USERS, seed=42):
    """Synthetic 'training data': one row per user plus a fraud label."""
    rng = np.random.default_rng(seed)
    orders = rng.poisson(6, n)
    basket = rng.gamma(4, 10, n).round(2)
    age = rng.integers(1, 2000, n)
    night = rng.beta(2, 8, n).round(3)
    X = np.column_stack([orders, basket, age, night])
    score = 6 * night + 0.01 * basket - 0.0008 * age - 0.05 * orders
    score += rng.normal(0, 0.2, n)
    y = (score > np.percentile(score, 90)).astype(int)
    return X, y


def row_to_features(row):
    """Turn a numpy row into a clean dict (orders as int, the rest as float)."""
    return {
        "orders_30d": int(row[0]),
        "avg_basket": float(row[1]),
        "account_age_days": int(row[2]),
        "night_txn_ratio": float(row[3]),
    }


def get_model():
    """Train a small logistic-regression fraud model on the synthetic data."""
    X, y = make_dataset()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    model.fit(X, y)
    return model


def predict(model, features):
    """Run the model on a feature dict (values come back from Redis as strings)."""
    time.sleep(MODEL_LATENCY_MS / 1000)   # simulated inference cost
    x = [[float(features[name]) for name in FEATURE_NAMES]]
    return "fraud" if model.predict(x)[0] == 1 else "ok"


def require_features(r):
    if not r.exists(f"user:{FIRST_USER_ID + 42}"):
        print("No features found in Redis. Run step 1 first:  python demo/step1_load_features.py")
        sys.exit(1)
