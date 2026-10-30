"""Step 2 - Serve and cache: look up features, predict, cache the result with a TTL."""
import hashlib
import json
import time

from common import (FIRST_USER_ID, PRED_TTL, get_model, get_redis, header, predict,
                    require_features)


def cache_key(features):
    """Same feature values -> same key, so repeated inputs share one cached prediction."""
    blob = json.dumps(features, sort_keys=True)
    return "pred:" + hashlib.sha1(blob.encode()).hexdigest()[:8]


def serve(r, model, user_id):
    """Return (prediction, was_cached, latency_ms) for one request."""
    t0 = time.perf_counter()

    features = r.hgetall(f"user:{user_id}")          # online feature lookup (1 round trip)
    if not features:
        raise KeyError(f"no features for user {user_id}")

    key = cache_key(features)
    pred = r.get(key)                                 # GET: is there a cached prediction?
    cached = pred is not None
    if not cached:
        pred = predict(model, features)               # cache miss: pay for the model
        r.set(key, pred, ex=PRED_TTL)                 # SET ... EX 300

    latency_ms = (time.perf_counter() - t0) * 1000
    return pred, cached, latency_ms


def run(r, model):
    header("STEP 2 - Serve and cache (GET / SET EX)")
    user_id = FIRST_USER_ID + 42

    for attempt in ("1st request", "2nd request (same input)"):
        pred, cached, ms = serve(r, model, user_id)
        print(f"{attempt:<26} prediction={pred:<6} cached={str(cached):<5} latency={ms:7.2f} ms")

    key = cache_key(r.hgetall(f"user:{user_id}"))
    print(f"\nCached under {key}, expires in {r.ttl(key)} seconds")


if __name__ == "__main__":
    _r = get_redis()
    require_features(_r)
    run(_r, get_model())
