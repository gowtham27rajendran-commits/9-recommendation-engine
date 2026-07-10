"""
Auto-promotion: implements the README's "Auto-promote winning variant
after N days or alpha=0.05" rule as a runnable script (intended to run on
a schedule, e.g. a daily cron/Airflow job in a real deployment).

Logic:
  1. Load logged events (data/events.jsonl).
  2. For each non-control variant, run a two-proportion z-test on CTR
     against the control ("popularity").
  3. If a variant is significant (p < alpha), positive lift, AND the
     experiment has run for >= MIN_DAYS - promote it: give it 100% of
     traffic in a and write out an updated A/B config.
  4. Otherwise, if MAX_DAYS have elapsed with no significant winner,
     keep the best-performing variant by point estimate but flag it as
     "inconclusive" rather than silently picking a possibly-noisy winner.

This intentionally does NOT peek continuously and promote at the first
moment p < 0.05 (classic "peeking" bug that inflates false-positive
rate) - it only evaluates once MIN_DAYS of data has accumulated.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ab_testing.router import DEFAULT_CONFIG, Variant, ABTestConfig
from app.ab_testing.stats import two_proportion_z_test

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONTROL_VARIANT = "popularity"
ALPHA = 0.05
MIN_DAYS = 7
MAX_DAYS = 21


def load_events() -> list[dict]:
    path = DATA_DIR / "events.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def aggregate_ctr(events: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    served, clicks = {}, {}
    for ev in events:
        variant = ev.get("variant")
        if ev.get("event_type") == "recommendation_served":
            served[variant] = served.get(variant, 0) + len(ev.get("item_ids", []))
        elif ev.get("event_type") == "interaction" and ev.get("action") == "click":
            clicks[variant] = clicks.get(variant, 0) + 1
    return served, clicks


def days_running(events: list[dict]) -> float:
    timestamps = [ev.get("logged_at") for ev in events if ev.get("logged_at")]
    if len(timestamps) < 2:
        return 0.0
    return (max(timestamps) - min(timestamps)) / 86400


def main():
    events = load_events()
    if not events:
        print("No events logged yet (data/events.jsonl is empty or missing). Nothing to evaluate.")
        print("Hit /recommendations/{user_id} and POST /interactions a few times first, or run the demo traffic simulator.")
        return

    served, clicks = aggregate_ctr(events)
    elapsed_days = days_running(events)
    print(f"Experiment running for ~{elapsed_days:.2f} days (based on logged event timestamps)")

    if CONTROL_VARIANT not in served:
        print(f"No served events for control variant '{CONTROL_VARIANT}' yet - can't compare.")
        return

    results = {}
    for variant, trials in served.items():
        if variant == CONTROL_VARIANT:
            continue
        result = two_proportion_z_test(
            metric_name="ctr",
            variant_a_name=CONTROL_VARIANT,
            successes_a=clicks.get(CONTROL_VARIANT, 0),
            trials_a=served[CONTROL_VARIANT],
            variant_b_name=variant,
            successes_b=clicks.get(variant, 0),
            trials_b=trials,
            alpha=ALPHA,
        )
        results[variant] = result
        print(
            f"  {variant:<15} ctr={result.rate_b:.4f} vs control ctr={result.rate_a:.4f} "
            f"lift={result.lift:+.1%}  p={result.p_value:.4f}  significant={result.significant}"
        )

    winners = [v for v, r in results.items() if r.significant and r.lift > 0]

    if elapsed_days < MIN_DAYS:
        print(f"\nNot promoting yet: experiment has only run {elapsed_days:.1f} of the required {MIN_DAYS} minimum days.")
        return

    if winners:
        # if multiple are significant, take the one with the highest lift
        best = max(winners, key=lambda v: results[v].lift)
        print(f"\nPromoting '{best}' to 100% traffic (p={results[best].p_value:.4f}, lift={results[best].lift:+.1%}).")
        new_config = ABTestConfig(
            experiment_name=DEFAULT_CONFIG.experiment_name + "_promoted",
            variants=[Variant(name=best, traffic_start=0, traffic_end=100)],
        )
        (DATA_DIR / "promoted_config.json").write_text(
            json.dumps({"experiment_name": new_config.experiment_name, "winner": best, "traffic_start": 0, "traffic_end": 100}, indent=2)
        )
        print(f"Wrote {DATA_DIR / 'promoted_config.json'} - wire this into ab_testing/router.py's active config to apply it.")
    elif elapsed_days >= MAX_DAYS:
        best_by_point_estimate = max(results, key=lambda v: results[v].rate_b) if results else CONTROL_VARIANT
        print(
            f"\nNo statistically significant winner after {elapsed_days:.1f} days (max {MAX_DAYS}). "
            f"Best point estimate is '{best_by_point_estimate}' but flagging as INCONCLUSIVE rather than auto-promoting - "
            "recommend extending the test, checking for a too-small minimum-detectable-effect, or redesigning the experiment."
        )
    else:
        print(f"\nNo significant winner yet, still within the {MAX_DAYS}-day evaluation window. Continue running.")


if __name__ == "__main__":
    main()
