"""Test historical trip-flow corrections against two ordinary Bluebikes weeks."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

from collect_status import DEFAULT_TRIP_FILES, station_map
from dock_regression import MAX_AGE, MAX_GAP, TARGET_TOLERANCE, metrics

TIME_ZONE = "America/New_York"
FIRST_WEEK = pd.Timestamp("2026-09-14", tz=TIME_ZONE)
SECOND_WEEK = pd.Timestamp("2026-09-21", tz=TIME_ZONE)
END = pd.Timestamp("2026-09-28", tz=TIME_ZONE)
WEEKS = ("2026-09-14_to_20", "2026-09-21_to_27")
WEEK_MINUTES = 7 * 24 * 60
HORIZON = 15


def historical_profile(paths):
    first, end = date(2026, 7, 1), date(2026, 9, 1)
    days = [first + timedelta(days=i) for i in range((end - first).days)]
    weekdays = {day.isoformat(): day.weekday() for day in days if day != date(2026, 7, 4)}
    denominators = Counter(weekdays.values())
    events = defaultdict(lambda: np.zeros(WEEK_MINUTES, dtype=np.int32))
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                for time_key, station_key, sign in (
                    ("started_at", "start_station_id", -1),
                    ("ended_at", "end_station_id", 1),
                ):
                    timestamp = row[time_key]
                    weekday = weekdays.get(timestamp[:10])
                    station = row[station_key].strip()
                    if weekday is not None and station:
                        minute = int(timestamp[11:13]) * 60 + int(timestamp[14:16])
                        events[station][weekday * 1440 + minute] += sign
    days_per_minute = np.repeat([denominators[day] for day in range(7)], 1440)
    profiles = {}
    for station, totals in events.items():
        means = totals / days_per_minute
        # Wrap Sunday into Monday for windows crossing the week boundary.
        cumulative = np.r_[0, np.cumsum(np.r_[means, means[:HORIZON - 1]])]
        profiles[station] = cumulative[HORIZON:] - cumulative[:-HORIZON]
    return profiles, dict(sorted(denominators.items()))


def load_week_records(path):
    frame = pd.read_parquet(path, columns=["id", "nuid", "timestamp", "free", "extra"])
    # CityBikes exports UTC timestamps without a timezone annotation.
    frame["local"] = frame.timestamp.dt.tz_localize("UTC").dt.tz_convert(TIME_ZONE)
    frame = frame[(frame.local >= FIRST_WEEK) & (frame.local < END)].copy()
    if frame.duplicated(["id", "timestamp"]).any():
        raise ValueError("Archive contains ambiguous duplicate station timestamps")
    extra = pd.json_normalize(frame.pop("extra").map(json.loads)).set_axis(frame.index)
    frame["capacity"] = pd.to_numeric(extra["slots"], errors="coerce")
    age = (frame.timestamp - pd.to_datetime(extra["last_updated"], errors="coerce")).dt.total_seconds()
    frame["valid"] = (extra["returning"].eq(1) & extra["virtual"].eq(False)
                      & age.between(0, MAX_AGE) & frame.free.ge(0)
                      & frame.free.le(frame.capacity))
    frame["week"] = np.where(frame.local < SECOND_WEEK, WEEKS[0], WEEKS[1])
    # Start at the next whole minute, shifting by less than a minute.
    rounded = frame.timestamp.dt.ceil("min").dt.tz_localize("UTC").dt.tz_convert(TIME_ZONE)
    frame["profile_index"] = rounded.dt.dayofweek * 1440 + rounded.dt.hour * 60 + rounded.dt.minute
    return frame.sort_values(["id", "timestamp"])


def match_targets(station):
    times = station.timestamp.to_numpy(dtype="datetime64[us]").astype(np.int64)
    requested = times + HORIZON * 60 * 1_000_000
    right = np.searchsorted(times, requested)
    left = np.maximum(right - 1, 0)
    right = np.minimum(right, len(times) - 1)
    target = np.where(requested - times[left] <= np.abs(times[right] - requested), left, right)
    # Compare gaps in UTC and keep each observation and its target in one report week.
    gaps = np.r_[0, np.cumsum(np.diff(times) > MAX_GAP * 1_000_000)]
    week = station.week.to_numpy()
    week_end = np.where(week == WEEKS[0], SECOND_WEEK.value // 1000, END.value // 1000)
    valid = station.valid.to_numpy()
    ok = ((np.abs(times[target] - requested) <= TARGET_TOLERANCE * 1_000_000)
          & (gaps == gaps[target]) & (week == week[target])
          & (requested < week_end) & valid & valid[target])
    return np.flatnonzero(ok), target[ok]


def score(samples):
    actual = samples["actual"].to_numpy()
    result = {"samples": len(samples), "stations": int(samples["station"].nunique())}
    for name in ("persistence", "net_flow_adjusted"):
        prediction = samples[name].to_numpy()
        errors = prediction - actual
        full = actual == 0
        # A continuous estimate of at least one dock triggers this decision.
        result[name] = {
            "all": metrics(errors),
            "actual_0_to_2": metrics(errors[actual <= 2]),
            "actual_full": int(full.sum()),
            "actual_full_predicted_at_least_one": int((full & (prediction >= 1)).sum()),
            "actual_full_predicted_above_zero": int((full & (prediction > 0)).sum()),
            "actual_at_least_one_predicted_below_one": int(((actual >= 1) & (prediction < 1)).sum()),
        }
    return result


def validate(archive, station_information, trips):
    profiles, days_per_weekday = historical_profile(trips)
    mapping = station_map(station_information)
    frame = load_week_records(archive)
    samples = []
    matched_counts = Counter()
    matched_station_counts = defaultdict(set)
    for archive_id, station in frame.groupby("id", sort=False):
        base_indices, target_indices = match_targets(station)
        if not len(base_indices):
            continue
        base = station.iloc[base_indices]
        for week, count in base.week.value_counts().items():
            matched_counts[week] += int(count)
            matched_station_counts[week].add(archive_id)
        ids = station.nuid.dropna().unique()
        if len(ids) != 1 or station.nuid.isna().any():
            continue
        trip_id = mapping.get(ids[0])
        if trip_id not in profiles:
            continue
        current = base.free.to_numpy(dtype=float)
        expected = profiles[trip_id][base.profile_index.to_numpy()]
        adjusted = np.clip(current - expected, 0, base.capacity.to_numpy())
        samples.append(pd.DataFrame({
            "week": base.week.to_numpy(),
            "station": archive_id,
            "actual": station.iloc[target_indices].free.to_numpy(dtype=float),
            "persistence": current,
            "net_flow_adjusted": adjusted,
        }))
    if not samples:
        raise ValueError("No matched samples with a historical net-flow profile")
    paired = pd.concat(samples, ignore_index=True)
    report = {
        "archive": archive.name,
        "reference_dates": ["2026-07-01", "2026-08-31"],
        "reference_excluded_dates": ["2026-07-04"],
        "reference_days_per_weekday_monday_first": list(days_per_weekday.values()),
        "horizon_minutes": HORIZON,
        "target_tolerance_seconds": TARGET_TOLERANCE,
        "maximum_gap_seconds": MAX_GAP,
        "station_status_max_age_seconds": MAX_AGE,
        "weeks": {},
    }
    for week in WEEKS:
        observed = frame[frame.week == week]
        scores = score(paired[paired.week == week])
        report["weeks"][week] = {
            "archive_rows": len(observed),
            "archive_stations": int(observed.id.nunique()),
            "candidate_pairs_before_mapping": matched_counts[week],
            "candidate_stations_before_mapping": len(matched_station_counts[week]),
            **scores,
        }
    report["combined"] = score(paired)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--station-information", required=True, type=Path)
    parser.add_argument("--trips", nargs=2, default=DEFAULT_TRIP_FILES, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.archive, args.station_information, args.trips),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
