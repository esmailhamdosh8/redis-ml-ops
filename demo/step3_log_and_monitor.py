"""Step 3 - Log and monitor: every request goes to a stream, counters track cache hits."""
import numpy as np

from common import FIRST_USER_ID, N_USERS, get_model, get_redis, header, require_features
from step2_serve_and_cache import serve

STREAM = "events:predictions"


def log_request(r, user_id, pred, cached, latency_ms):
    """One round trip: append the event (XADD) and bump the counters (INCR)."""
    pipe = r.pipeline()
    pipe.xadd(
        STREAM,
        {"user_id": user_id, "prediction": pred, "cached": int(cached),
         "latency_ms": round(latency_ms, 3)},
        maxlen=10_000, approximate=True,      # keep the log bounded
    )
    pipe.incr("stats:requests")
    pipe.incr("stats:cache_hits" if cached else "stats:cache_misses")
    pipe.execute()


def handle_request(r, model, user_id):
    """The full request path: serve (features + cache + model), then log."""
    pred, cached, latency_ms = serve(r, model, user_id)
    log_request(r, user_id, pred, cached, latency_ms)
    return pred, cached, latency_ms


def get_stats(r):
    requests = int(r.get("stats:requests") or 0)
    hits = int(r.get("stats:cache_hits") or 0)
    misses = int(r.get("stats:cache_misses") or 0)
    return {"requests": requests, "hits": hits, "misses": misses,
            "hit_rate": hits / requests if requests else 0.0}


def run(r, model, n_requests=300, seed=7):
    header("STEP 3 - Log and monitor (XADD / INCR)")

    # Realistic traffic: a few popular users show up again and again.
    rng = np.random.default_rng(seed)
    user_ids = np.arange(FIRST_USER_ID, FIRST_USER_ID + N_USERS)
    weights = 1 / np.arange(1, N_USERS + 1)
    weights /= weights.sum()
    traffic = rng.choice(user_ids, size=n_requests, p=weights)

    for uid in traffic:
        handle_request(r, model, int(uid))

    s = get_stats(r)
    print(f"Handled {s['requests']} requests with skewed traffic")
    print(f"  stats:cache_hits   = {s['hits']}")
    print(f"  stats:cache_misses = {s['misses']}")
    print(f"  cache hit rate     = {s['hit_rate']:.0%}")
    print(f"\nXLEN {STREAM} = {r.xlen(STREAM)}")
    print("Last 3 events (XREVRANGE):")
    for event_id, fields in r.xrevrange(STREAM, count=3):
        print(f"  {event_id}  {fields}")


if __name__ == "__main__":
    _r = get_redis()
    require_features(_r)
    run(_r, get_model())
