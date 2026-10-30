"""Step 1 - Load features: training data becomes one Redis hash per entity (HSET)."""
from common import (FEATURE_NAMES, FEATURE_TTL, FIRST_USER_ID, N_USERS, get_redis,
                    header, make_dataset, row_to_features)


def run(r):
    header("STEP 1 - Load features (HSET)")
    X, _ = make_dataset(N_USERS)

    # One pipeline = one round trip for all 2 * N_USERS commands.
    pipe = r.pipeline(transaction=False)
    for i, row in enumerate(X):
        key = f"user:{FIRST_USER_ID + i}"
        pipe.hset(key, mapping=row_to_features(row))
        pipe.expire(key, FEATURE_TTL)
    pipe.execute()

    sample = f"user:{FIRST_USER_ID + 42}"
    print(f"Loaded {N_USERS} users as hashes with fields {FEATURE_NAMES}")
    print(f"\nHGETALL {sample}")
    for field, value in r.hgetall(sample).items():
        print(f"  {field:<18} {value}")
    print(f"TTL {sample} = {r.ttl(sample)} seconds")


if __name__ == "__main__":
    run(get_redis())
