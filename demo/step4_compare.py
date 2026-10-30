"""Step 4 - Compare: cold vs. cached latency and the cache hit rate."""
import numpy as np

from common import FIRST_USER_ID, MODEL_LATENCY_MS, get_model, get_redis, header, reset, require_features
from step3_log_and_monitor import get_stats, handle_request


def _summary(latencies):
    a = np.array(latencies)
    return {"p50": np.percentile(a, 50), "p95": np.percentile(a, 95), "mean": a.mean()}


def run(r, model, n_users=200):
    header("STEP 4 - Compare cold vs. cached latency")

    # Start from an empty cache and empty counters so the numbers are clean.
    reset(r, ["pred:*", "stats:*", "events:*"])
    users = range(FIRST_USER_ID, FIRST_USER_ID + n_users)

    cold = [handle_request(r, model, u)[2] for u in users]    # first time: all misses
    warm = [handle_request(r, model, u)[2] for u in users]    # same inputs again: all hits

    c, w = _summary(cold), _summary(warm)
    print(f"{n_users} distinct users, each requested twice "
          f"(simulated model latency: {MODEL_LATENCY_MS:g} ms)\n")
    print(f"{'':<16}{'p50 (ms)':>10}{'p95 (ms)':>10}{'mean (ms)':>11}")
    print(f"{'cold (miss)':<16}{c['p50']:>10.2f}{c['p95']:>10.2f}{c['mean']:>11.2f}")
    print(f"{'cached (hit)':<16}{w['p50']:>10.2f}{w['p95']:>10.2f}{w['mean']:>11.2f}")
    print(f"\nSpeed-up at p50: {c['p50'] / w['p50']:.1f}x")

    s = get_stats(r)
    print(f"Cache hit rate over the run: {s['hit_rate']:.0%} "
          f"({s['hits']} hits / {s['requests']} requests)")


if __name__ == "__main__":
    _r = get_redis()
    require_features(_r)
    run(_r, get_model())
