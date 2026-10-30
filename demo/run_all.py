"""Run the whole demo: python demo/run_all.py"""
import step1_load_features
import step2_serve_and_cache
import step3_log_and_monitor
import step4_compare
from common import REDIS_URL, get_model, get_redis, header, reset


def main():
    r = get_redis()
    header(f"Redis for ML demo - connected to {REDIS_URL}")
    reset(r)                      # start from a clean slate
    model = get_model()

    step1_load_features.run(r)
    step2_serve_and_cache.run(r, model)
    step3_log_and_monitor.run(r, model)
    step4_compare.run(r, model)

    print("\nDone. Explore the data yourself with:  redis-cli  (try HGETALL user:1042)")


if __name__ == "__main__":
    main()
