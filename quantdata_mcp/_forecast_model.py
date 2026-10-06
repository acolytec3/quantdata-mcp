"""Reproduction of QuantData's client-side "Exposure Forecast" statistics.

NOT part of the upstream `quantdata-mcp` package. The Exposure Forecast chart
(`qd_get_exposure_forecast`'s underlying endpoint, `options/exposure/forecast/
{tool_id}`) returns raw per-session state/price observations, but the "Holds
X%", "Weak/Moderate/Strong", "Typical move", and forward-range cone shown in
QuantData's own UI are computed entirely CLIENT-SIDE from that same payload
-- there is no separate backend field or endpoint for them.

This module is a best-effort Python port of that client-side algorithm,
reverse engineered from QuantData's own (lightly-minified, unminified
identifier names) Next.js JS bundle on 2026-10-06. Validated against a live
session: typical-move output matched the live chart to within 0.01pp per
horizon; hold-probability output was within ~1-2 percentage points (likely
due to the live chart's exact session cutoff / anchor timing differing by a
few minutes from this reproduction's capture moment).

Algorithm (from bundle constants/functions, see AGENTS.md for the full trace):

1. Hold probability + confidence ("Holds 79% ... Strong"):
   - Pool = the last ``TRANSITION_WINDOW_BUSINESS_DAYS`` (20) business days
     (calendar weekdays, NOT 20 actual trading sessions -- a holiday in the
     window just yields fewer matched sessions).
   - For every observation in that pool, for each horizon in ``HORIZONS``
     (30/60/90/120 min), find the observation ~that far ahead (nearest
     match within ``OBSERVATION_TOLERANCE_MS``) and tally a per-session
     state -> state transition count matrix.
   - Aggregate counts across sessions, then Laplace-smooth
     (``SMOOTHING_ALPHA`` = 1) over the 4 states (PIN/GRIND/HOSTILE/
     TRANSITION): ``P(state') = (count[state'] + alpha) / (total + alpha*4)``.
   - "Holds" = the diagonal entry, ``P(same state | same state, horizon)``.
   - Confidence: a session-block bootstrap (1000 draws, seed 7, resampling
     whole sessions with replacement) gives a 90% CI (5th/95th percentile of
     the resampled hold-rate). Flagged low-confidence ("Weak") when the row
     has < 25 total samples or the CI is wider than 30 percentage points.

2. Typical move + forward-range cone:
   - Separate pool = the last ``OUTCOME_POOL_SESSIONS`` (70) sessions with
     any observations (NOT business-day filtered).
   - Each session contributes up to one matched sample per fixed daily
     checkpoint (``READ_BUCKET_MINUTES_OF_DAY``: 9:45/10:30/11:30/12:30/
     13:30/14:30 ET) -- the first observation at/after that checkpoint.
   - For each sample whose starting state matches the current state (at
     the checkpoint nearest-but-not-after "now"), the price move to the
     observation ~30/60/90/120 min later is recorded as a signed fraction.
   - "Typical move" = the **median absolute** move fraction across the
     matched sample pool, per horizon.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

STATES: tuple[str, ...] = ("PIN", "GRIND", "HOSTILE", "TRANSITION")
HORIZON_MINUTES: tuple[int, ...] = (30, 60, 90, 120)
OBSERVATION_TOLERANCE_MS = 600_000  # 10 minutes
READ_BUCKET_MINUTES_OF_DAY: tuple[int, ...] = (585, 630, 690, 750, 810, 870)
SMOOTHING_ALPHA = 1
MIN_TRANSITION_ROW_SAMPLE_COUNT = 25
MAX_HOLD_INTERVAL_WIDTH = 0.3
OUTCOME_POOL_SESSIONS = 70
TRANSITION_WINDOW_BUSINESS_DAYS = 20
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 7

# UI display names / "Hostile" is shown as "Volatile" in QuantData's own UI.
DISPLAY_NAMES: dict[str, str] = {
    "PIN": "Pin",
    "GRIND": "Grind",
    "HOSTILE": "Volatile",
    "TRANSITION": "Transition",
}


def _business_days_before(date_str: str, n: int) -> set[str]:
    """Last ``n`` calendar weekdays (Mon-Fri) strictly before ``date_str``."""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    out: set[str] = set()
    while len(out) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            out.add(d.isoformat())
    return out


def _nearest_observation(
    obs: list[dict[str, Any]], target_ms: int, tol_ms: int = OBSERVATION_TOLERANCE_MS
) -> dict[str, Any] | None:
    best, best_diff = None, float("inf")
    for o in obs:
        ts = o.get("targetTimeEpochMillisTimestamp")
        if ts is None:
            continue
        diff = abs(ts - target_ms)
        if diff <= tol_ms and diff < best_diff:
            best, best_diff = o, diff
    return best


def _first_at_or_after(
    obs: list[dict[str, Any]], target_ms: int, tol_ms: int = OBSERVATION_TOLERANCE_MS
) -> dict[str, Any] | None:
    best = None
    for o in obs:
        ts = o.get("targetTimeEpochMillisTimestamp")
        if ts is None:
            continue
        d = ts - target_ms
        if 0 <= d <= tol_ms and (
            best is None or ts < best["targetTimeEpochMillisTimestamp"]
        ):
            best = o
    return best


def _mulberry32(seed: int) -> Any:
    """Port of the bundle's `createExposureForecastBootstrapRandom` (mulberry32 PRNG)."""
    state = {"v": seed & 0xFFFFFFFF}

    def next_int(bound: int) -> int:
        state["v"] = (state["v"] + 0x6D2B79F5) & 0xFFFFFFFF
        i = state["v"]
        i = ((i ^ (i >> 15)) * (i | 1)) & 0xFFFFFFFF
        inner = (i + (((i ^ (i >> 7)) * (i | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        i = (i ^ inner) & 0xFFFFFFFF
        i = (i ^ (i >> 14)) & 0xFFFFFFFF
        return math.floor(i / 0x100000000 * bound)

    return next_int


def _bootstrap_hold_ci(
    session_rows: list[dict[str, int]],
) -> tuple[float, float] | None:
    """Session-block bootstrap 90% CI for the aggregate hold rate."""
    if not any(r["sampleCount"] > 0 for r in session_rows):
        return None
    rng = _mulberry32(BOOTSTRAP_SEED)
    n = len(session_rows)
    draws: list[float] = []
    while len(draws) < BOOTSTRAP_DRAWS:
        hold = total = 0
        for _ in range(n):
            r = session_rows[rng(n)]
            hold += r["holdCount"]
            total += r["sampleCount"]
        if total > 0:
            draws.append(hold / total)
    draws.sort()
    lo = math.floor(0.05 * BOOTSTRAP_DRAWS)
    hi = math.floor(0.95 * BOOTSTRAP_DRAWS) - 1
    return draws[lo], draws[hi]


def _median_abs(values: list[float]) -> float | None:
    if not values:
        return None
    v = sorted(abs(x) for x in values)
    n = len(v)
    mid = n // 2
    return (v[mid - 1] + v[mid]) / 2 if n % 2 == 0 else v[mid]


def nearest_bucket_at_or_before(minute_of_day: int) -> int:
    chosen = READ_BUCKET_MINUTES_OF_DAY[0]
    for b in READ_BUCKET_MINUTES_OF_DAY:
        if b <= minute_of_day:
            chosen = b
    return chosen


def _transition_basis(sessions: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    """Per (horizon, starting-state): smoothed hold probability + bootstrap CI."""
    agg = {h: {s: {s2: 0 for s2 in STATES} for s in STATES} for h in HORIZON_MINUTES}
    per_session: list[dict[int, dict[str, dict[str, int]]]] = []

    for sess in sessions:
        obs = [o for o in (sess.get("observations") or []) if o.get("state") in STATES]
        if not obs:
            continue
        mat = {h: {s: {s2: 0 for s2 in STATES} for s in STATES} for h in HORIZON_MINUTES}
        matched = False
        for o in obs:
            st = o["state"]
            for h in HORIZON_MINUTES:
                fut = _nearest_observation(obs, o["targetTimeEpochMillisTimestamp"] + h * 60_000)
                if fut is not None and fut.get("state") in STATES:
                    mat[h][st][fut["state"]] += 1
                    matched = True
        if not matched:
            continue
        for h in HORIZON_MINUTES:
            for s in STATES:
                for s2 in STATES:
                    agg[h][s][s2] += mat[h][s][s2]
        per_session.append(mat)

    rows: dict[int, dict[str, dict[str, Any]]] = {}
    for h in HORIZON_MINUTES:
        rows[h] = {}
        for s in STATES:
            dest_map = dict(agg[h][s])
            total = sum(dest_map.values())
            ci = _bootstrap_hold_ci(
                [
                    {"holdCount": m[h][s][s], "sampleCount": sum(m[h][s].values())}
                    for m in per_session
                ]
            )
            is_low = (
                total < MIN_TRANSITION_ROW_SAMPLE_COUNT
                or ci is None
                or (ci[1] - ci[0]) > MAX_HOLD_INTERVAL_WIDTH
            )
            prob_total = total + SMOOTHING_ALPHA * len(STATES)
            prob = {
                s2: (0.0 if prob_total == 0 else (dest_map[s2] + SMOOTHING_ALPHA) / prob_total)
                for s2 in STATES
            }
            rows[h][s] = {
                "destCounts": dest_map,
                "total": total,
                "ci": ci,
                "isLowSample": is_low,
                "prob": prob,
            }
    return rows


def _session_moment_ms(session_date: str, minute_of_day: int) -> int:
    d = datetime.strptime(session_date, "%Y-%m-%d")
    return int(d.timestamp() * 1000) + minute_of_day * 60_000


def _outcome_pool(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One matched sample per (session, fixed daily checkpoint)."""
    samples: list[dict[str, Any]] = []
    for sess in sessions:
        obs = sess.get("observations") or []
        if not obs:
            continue
        for bucket in READ_BUCKET_MINUTES_OF_DAY:
            bucket_ms = _session_moment_ms(sess["sessionDate"], bucket)
            s = _first_at_or_after(obs, bucket_ms)
            if s is None or s.get("state") not in STATES:
                continue
            outcomes: dict[int, dict[str, Any]] = {}
            for h in HORIZON_MINUTES:
                fut = _nearest_observation(obs, s["targetTimeEpochMillisTimestamp"] + h * 60_000)
                if fut is not None and fut.get("stockPriceInCents") and s.get("stockPriceInCents"):
                    mf = (fut["stockPriceInCents"] - s["stockPriceInCents"]) / s["stockPriceInCents"]
                    outcomes[h] = {"endingState": fut["state"], "moveFraction": mf}
            samples.append(
                {
                    "sessionDate": sess["sessionDate"],
                    "bucketMinute": bucket,
                    "startingState": s["state"],
                    "outcomes": outcomes,
                }
            )
    return samples


def confidence_label(ci: tuple[float, float] | None, is_low_sample: bool) -> str:
    """Weak / Moderate / Strong, approximating the UI's own bucketing from CI width."""
    if is_low_sample or ci is None:
        return "Weak"
    width = ci[1] - ci[0]
    if width <= 0.12:
        return "Strong"
    if width <= 0.2:
        return "Moderate"
    return "Weak"


def compute_forecast_projection(
    resp: dict[str, Any],
) -> dict[str, Any] | None:
    """Reproduce the Exposure Forecast chart's Holds%/Typical-move numbers.

    Args:
        resp: The ``response`` object from ``options/exposure/forecast/{tool_id}``
            (i.e. ``fetch_exposure_forecast(...)["response"]``).

    Returns:
        None if there's no live observation to anchor on. Otherwise:
            {
                "current_state": "PIN",
                "matched_bucket_minute": 630,
                "horizons": {
                    30: {"hold_pct": 77.1, "confidence": "Strong", "n": 214,
                         "typical_move_pct": 0.092, "move_n": 14},
                    ...
                },
            }
    """
    live = resp.get("liveSession") or {}
    live_obs = [o for o in (live.get("observations") or []) if o.get("state") in STATES]
    if not live_obs:
        return None

    today = live.get("sessionDate")
    hist = [s for s in (resp.get("historicalSessions") or []) if s.get("observations")]
    all_sorted = sorted(hist, key=lambda s: s["sessionDate"])

    cur = live_obs[-1]
    cur_state = cur["state"]
    cur_dt = datetime.fromtimestamp(cur["targetTimeEpochMillisTimestamp"] / 1000)
    cur_minute = cur_dt.hour * 60 + cur_dt.minute
    bucket = nearest_bucket_at_or_before(cur_minute)

    transition_window = _business_days_before(today, TRANSITION_WINDOW_BUSINESS_DAYS)
    transition_pool = [s for s in all_sorted if s["sessionDate"] in transition_window]
    outcome_sessions = all_sorted[-OUTCOME_POOL_SESSIONS:]

    rows = _transition_basis(transition_pool)
    samples = _outcome_pool(outcome_sessions)
    matched = [s for s in samples if s["startingState"] == cur_state and s["bucketMinute"] == bucket]

    horizons: dict[int, dict[str, Any]] = {}
    for h in HORIZON_MINUTES:
        row = rows[h][cur_state]
        moves = [s["outcomes"][h]["moveFraction"] for s in matched if h in s["outcomes"]]
        typical = _median_abs(moves)
        horizons[h] = {
            "hold_pct": row["prob"][cur_state] * 100,
            "confidence": confidence_label(row["ci"], row["isLowSample"]),
            "n": row["total"],
            "ci": row["ci"],
            "typical_move_pct": None if typical is None else typical * 100,
            "move_n": len(moves),
        }

    return {
        "current_state": cur_state,
        "current_state_display": DISPLAY_NAMES.get(cur_state, cur_state),
        "matched_bucket_minute": bucket,
        "transition_pool_sessions": len(transition_pool),
        "outcome_pool_sessions": len(outcome_sessions),
        "horizons": horizons,
    }
