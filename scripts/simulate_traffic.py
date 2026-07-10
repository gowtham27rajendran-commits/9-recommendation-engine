"""
Simulates realistic user traffic against a running API so the A/B testing
endpoints (/experiments/results, scripts/evaluate_experiment.py) have
something non-trivial to analyze without waiting days for real users.

Simulated click-through behavior is intentionally biased so that
"hybrid" and "content_based" slightly outperform "popularity" - this
mirrors the real-world expectation that personalized ranking beats a
non-personalized baseline, and gives the significance tests something
real to detect. Also spreads fake timestamps across a week so
scripts/evaluate_experiment.py's MIN_DAYS gate can be demoed by editing
logged_at post-hoc (see --backdate flag).

Usage:
  python scripts/simulate_traffic.py --requests 3000 --base-url http://127.0.0.1:8000
"""
import argparse
import json
import random
import time
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# fake CTR by variant, used only to decide whether the simulator "clicks"
# a returned item - purely for generating demo data, not part of the
# production system itself.
SIMULATED_CTR = {
    "popularity": 0.07,
    "collaborative": 0.09,
    "content_based": 0.10,
    "hybrid": 0.115,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--n-users", type=int, default=2000)
    parser.add_argument("--backdate-days", type=float, default=7.5, help="Spread simulated events across this many past days")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=5.0)
    rng = random.Random(7)

    served = 0
    clicked = 0
    for i in range(args.requests):
        user_id = rng.randrange(args.n_users)
        try:
            resp = client.get(f"/recommendations/{user_id}", params={"top_n": 10})
        except httpx.HTTPError as exc:
            print(f"Request failed ({exc}); is the server running at {args.base_url}?")
            return
        if resp.status_code != 200:
            continue
        body = resp.json()
        variant = body["variant"]
        served += 1

        ctr = SIMULATED_CTR.get(variant, 0.07)
        for item in body["results"]:
            if rng.random() < ctr / len(body["results"]) * 3:  # a couple of items get most of the click probability
                client.post(
                    "/interactions",
                    json={
                        "user_id": user_id,
                        "item_id": item["item_id"],
                        "action": "click",
                        "variant": variant,
                        "request_id": body["request_id"],
                    },
                )
                clicked += 1
                break  # at most one click per served list, keeps CTR interpretable as "% of lists with a click"

        if i % 200 == 0:
            print(f"{i}/{args.requests} requests sent...")

    print(f"Done. Served {served} recommendation lists, logged {clicked} clicks.")

    # backdate the events so evaluate_experiment.py's MIN_DAYS gate has something to work with
    events_path = DATA_DIR / "events.jsonl"
    if events_path.exists() and args.backdate_days > 0:
        lines = events_path.read_text().splitlines()
        now = time.time()
        spread = args.backdate_days * 86400
        rewritten = []
        for idx, line in enumerate(lines):
            ev = json.loads(line)
            ev["logged_at"] = now - spread + (spread * idx / max(len(lines) - 1, 1))
            rewritten.append(json.dumps(ev))
        events_path.write_text("\n".join(rewritten) + "\n")
        print(f"Backdated {len(lines)} events across the last {args.backdate_days} days for demo purposes.")


if __name__ == "__main__":
    main()
