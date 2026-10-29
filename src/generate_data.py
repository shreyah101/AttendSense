"""Independent synthetic trajectories; no real semester records are read."""
import numpy as np
import pandas as pd
from src import ROOT

SEED = 42
FEATURES = ["conducted", "attended", "current_percentage", "remaining", "recent_rate",
            "historical_rate", "consecutive_absences", "semester_progress", "weekly_frequency", "threshold"]


def generate(n: int = 1500, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    behaviours = ["consistent", "irregular", "improving", "declining", "streaks", "borderline"]
    for identity in range(n):
        kind = behaviours[identity % len(behaviours)]
        total = int(rng.integers(35, 101))
        c = int(rng.integers(8, total))
        threshold = int(rng.choice([65, 70, 75, 80, 85], p=[.08, .12, .6, .12, .08]))
        base = rng.uniform(.84, .98) if kind == "consistent" else rng.uniform(.55, .89)
        if kind == "borderline":
            base = threshold / 100 + rng.uniform(-.06, .06)
        slope = rng.uniform(.12, .35) * (1 if kind == "improving" else -1 if kind == "declining" else 0)
        # All events are generated in order, including the snapshot's future.
        events = []
        shock = 0
        for i in range(total):
            probability = base + slope * (i / total - .5)
            if kind == "irregular":
                probability += rng.normal(0, .17)
            if kind == "streaks" and rng.random() < .07:
                shock = int(rng.integers(2, 5))
            if shock:
                probability -= .5
                shock -= 1
            events.append(int(rng.random() < np.clip(probability, .05, .99)))
        past, future = events[:c], events[c:]
        a = sum(past)
        streak = 0
        for event in reversed(past):
            if event:
                break
            streak += 1
        recent = past[-min(8, c):]
        historical = past[:-len(recent)] or past
        final = a + sum(future)
        rows.append(dict(trajectory_id=identity, behaviour=kind, conducted=c, attended=a,
                         current_percentage=100*a/c, remaining=total-c, recent_rate=np.mean(recent),
                         historical_rate=np.mean(historical), consecutive_absences=streak,
                         semester_progress=c/total, weekly_frequency=int(rng.integers(1, 6)),
                         threshold=threshold, final_attended=final, final_percentage=100*final/total,
                         shortage=int(100*final < threshold*total),
                         past_events="".join(map(str, past)), future_events="".join(map(str, future))))
    return pd.DataFrame(rows)


def main() -> None:
    path = ROOT / "data" / "synthetic_attendance.csv"
    path.parent.mkdir(exist_ok=True)
    data = generate()
    data.to_csv(path, index=False)
    print(f"Saved {len(data)} synthetic trajectories to {path}; shortage rate {data.shortage.mean():.1%}")


if __name__ == "__main__":
    main()
