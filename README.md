# redis-mlops-demo

Live demo for the **YZV 448E MLOps tool presentation: Redis for ML**.

A tiny fraud-prediction service that uses **one Redis instance** as a feature store,
a prediction cache and a monitoring log.

| Step | What happens | Redis commands |
|------|--------------|----------------|
| 1. Load features | data becomes one hash per user | `HSET`, `EXPIRE` |
| 2. Serve and cache | Look up features, predict, cache the result with a TTL | `HGETALL`, `GET`, `SET ... EX` |
| 3. Log and monitor | Every request goes to a stream; counters track cache hits | `XADD`, `INCR` |
| 4. Compare | Cold vs. cached latency and the cache hit rate | (reads the counters) |

## Run it

```bash
# 1. start Redis (any Redis works; Docker is the easiest)
docker compose up -d

# 2. install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. run all four steps
python demo/run_all.py
```

Each step also runs on its own (steps 2-4 need step 1 first):

```bash
python demo/step1_load_features.py
python demo/step2_serve_and_cache.py
python demo/step3_log_and_monitor.py
python demo/step4_compare.py
```

## Configuration

| Environment variable | Default | Meaning |
|----------------------|---------|---------|
| `REDIS_URL` | `redis://localhost:6379/0` | Where Redis is |
| `MODEL_LATENCY_MS` | `25` | Simulated inference time added to every model call |
| `N_USERS` | `1000` | Number of users loaded in step 1 |

**About `MODEL_LATENCY_MS`:** the sklearn model here is tiny and answers in well under a
millisecond, so caching it would show almost no benefit. The 25 ms delay stands in for a
real model (a deep network, a remote API). Set `MODEL_LATENCY_MS=0` to see the honest
numbers for this toy model. Redis lookups are real and measured either way.

## Look inside Redis

```bash
docker compose exec redis redis-cli
> HGETALL user:1042
> TTL user:1042
> KEYS pred:*
> GET stats:cache_hits
> XLEN events:predictions
> XREVRANGE events:predictions + - COUNT 3
```

## Notes

- Prediction cache keys are a hash of the feature values, so identical inputs share a
  cached result (`pred:9f3a...`, as on the slide).
