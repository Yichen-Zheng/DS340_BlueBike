"""Explore Bluebikes station groups defined by historical trip activity."""

import argparse
from collections import defaultdict
from contextlib import closing
import csv
from datetime import date
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler

from dock_regression import (
    bounded,
    metrics,
    parse_time,
    return_decisions,
    split_station_histories,
    station_histories,
    store_identity,
    store_uri,
)
from collect_status import DEFAULT_TRIP_FILES, station_map


def traffic_groups(paths):
    counts = defaultdict(lambda: np.zeros((2, 4, 2)))
    days = defaultdict(set)
    day_cache = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                for time_key, station_key, direction in (
                    ("started_at", "start_station_id", 0),
                    ("ended_at", "end_station_id", 1),
                ):
                    t = row[time_key]
                    day = t[:10]
                    station = row[station_key].strip()
                    if (
                        not station
                        or not "2026-07-01" <= day < "2026-09-01"
                        or day in ("2026-07-03", "2026-07-04")
                    ):
                        continue
                    if day not in day_cache:
                        day_cache[day] = int(date.fromisoformat(day).weekday() >= 5)
                    weekend = day_cache[day]
                    hour = int(t[11:13])
                    period = (
                        0
                        if hour < 7 or hour >= 20
                        else 1 if hour < 10 else 2 if hour < 16 else 3
                    )
                    counts[station][weekend, period, direction] += 1
                    days[station].add(day)
    station_ids = sorted(
        s for s in counts if len(days[s]) >= 20 and counts[s].sum() >= 100
    )
    weekday_count = sum(day_cache[d] == 0 for d in day_cache)
    weekend_count = len(day_cache) - weekday_count
    raw = np.stack(
        [
            counts[s] / np.array([weekday_count, weekend_count])[:, None, None]
            for s in station_ids
        ]
    )
    total = raw.sum(axis=(1, 2, 3))
    normalized = raw / total[:, None, None, None]
    features = np.column_stack([np.log1p(total), normalized.reshape(-1, 16)])
    scaler = StandardScaler().fit(features)
    standard = scaler.transform(features)
    centers, _ = kmeans(standard, 4, iter=20, seed=42)
    labels, _ = vq(standard, centers)
    return (
        dict(zip(station_ids, labels.tolist())),
        {
            str(i): {
                "stations": int((labels == i).sum()),
                "median_weekday_events": float(
                    np.median(raw[labels == i, 0].sum(axis=(1, 2)))
                ),
                "median_weekend_events": float(
                    np.median(raw[labels == i, 1].sum(axis=(1, 2)))
                ),
                "weekday_events_by_time_and_direction": raw[labels == i, 0]
                .mean(axis=0)
                .round(2)
                .tolist(),
            }
            for i in range(4)
        },
        {"scaler": scaler, "centers": centers},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trips", nargs=2, default=DEFAULT_TRIP_FILES, type=Path)
    parser.add_argument("--station-information", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--classifier", required=True, type=Path)
    args = parser.parse_args()
    labels, description, classifier = traffic_groups(args.trips)
    mapping = station_map(args.station_information)
    artifact = joblib.load(args.model)
    global_model = artifact["pipeline"]
    md = artifact["metadata"]
    bounds = tuple(parse_time(md[k]) for k in ("validation_starts", "test_starts"))
    with closing(sqlite3.connect(store_uri(args.db), uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        _, source = store_identity(conn)
        # The splitter needs a later nonempty group. Build through the prior test window,
        # then discard it without scoring either method on it.
        end = parse_time(md["test_ends_exclusive"])
        histories = (rows for _, rows in station_histories(conn, source, end=end))
        (train, val, _), _ = split_station_histories(histories, 15, bounds, end, True)

    def group(samples):
        g = defaultdict(list)
        for row, y, *_ in samples:
            g[labels.get(mapping.get(row["station_id"]), -1)].append((row, y))
        return g

    train_groups, val_groups = group(train), group(val)

    def score(actual, pred):
        return {
            "all": metrics(pred - actual),
            "near_full": metrics((pred - actual)[actual <= 2]),
            "return_decisions_at_one": return_decisions(actual, pred, 1.0),
        }

    total = defaultdict(list)
    report = {
        "clusters": description,
        "reference_period": ["2026-07-01", "2026-08-31"],
        "validation_period": ["2026-10-05", "2026-10-18"],
        "validation_by_cluster": {},
    }
    for k, rows in sorted(val_groups.items()):
        x, y = zip(*rows)
        truth = np.array(y)
        pool = bounded(global_model.predict(x), x)
        refs = train_groups[k]
        stations = len({features["station_id"] for features, _ in refs})
        local = k >= 0 and len(refs) >= 500 and stations >= 5
        pred = pool
        if local:
            x_train, y_train = zip(*refs)
            model = clone(global_model).fit(x_train, y_train)
            pred = bounded(model.predict(x), x)
        values = {
            "persistence": np.array([row["available_docks"] for row in x]),
            "global": pool,
            "traffic_group_or_global_fallback": pred,
        }
        report["validation_by_cluster"][str(k)] = {
            "training_samples": len(refs),
            "training_stations": stations,
            "split_model": local,
            **{name: score(truth, p) for name, p in values.items()},
        }
        total["truth"].append(truth)
        for name, p in values.items():
            total[name].append(p)
    truth = np.concatenate(total.pop("truth"))
    report["validation_total"] = {
        k: score(truth, np.concatenate(v)) for k, v in total.items()
    }
    report["station_groups_by_trip_id"] = labels
    args.classifier.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**classifier, "station_groups_by_trip_id": labels}, args.classifier)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
