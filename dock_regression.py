"""Train and serve time-based Bluebikes dock-availability regressions."""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from itertools import groupby
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOCAL_TZ = ZoneInfo("America/New_York")
TIME_BANDS = (
    ("00:00–06:00", 0, 6),
    ("06:00–10:00", 6, 10),
    ("10:00–16:00", 10, 16),
    ("16:00–20:00", 16, 20),
    ("20:00–24:00", 20, 24),
)
LAGS = (5, 15, 30)
TARGET_TOLERANCE = 90
LAG_TOLERANCE = 120
MAX_GAP = 180
MAX_AGE = 300
FEATURE_VERSION = 1
OPTIONAL_LAG_FEATURE_VERSION = 2
FLOW_GROUP_FEATURE_VERSION = 3
CHANGE_FEATURE_VERSION = 4
DOCK_CLASSIFIER_KIND = "dock_availability_logistic_v1"
FEATURES = (
    "station_id", "available_docks", "lag_5", "lag_15", "lag_30", "change_5",
    "hour_sin", "hour_cos", "weekday", "weekend", "capacity", "latitude", "longitude",
)


def parse_time(value):
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.utcoffset() is None:
        raise ValueError("Time must include an explicit UTC offset")
    return timestamp.timestamp()


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def store_uri(path):
    return Path(path).resolve().as_uri() + "?mode=ro"


def store_identity(db):
    metadata = dict(db.execute("SELECT key, value FROM metadata"))
    kind = metadata["data_kind"]
    source = metadata.get("data_source", "official_gbfs" if kind == "real" else kind)
    return kind, source


def station_histories(db, source, station_id=None, at=None, end=math.inf):
    query = "SELECT *, ? AS data_source FROM station_status"
    args = (source,)
    if station_id is not None:
        query += " WHERE station_id=? AND observed_at BETWEEN ? AND ?"
        args += (station_id, at - 60 * max(LAGS) - LAG_TOLERANCE, at)
    elif math.isfinite(end):
        query += " WHERE observed_at < ?"
        args += (end,)
    rows = db.execute(query + " ORDER BY station_id, observed_at", args)
    for key, group in groupby(rows, key=lambda row: row["station_id"]):
        yield key, list(group)


def load_history(path, station_id=None, at=None):
    with closing(sqlite3.connect(store_uri(path), uri=True)) as db:
        db.row_factory = sqlite3.Row
        kind, source = store_identity(db)
        history = dict(station_histories(db, source, station_id, at))
    return kind, history


def dock_count(row):
    value = row["available_docks"]
    return type(value) is int and value >= 0


def returnable(row):
    archive = "data_source" in row.keys() and row["data_source"] == "citybikes"
    installed = row["is_installed"] == 1 or (archive and row["is_installed"] is None)
    return installed and row["is_returning"] == 1


def fresh(row, at):
    reported = row["last_reported"]
    return (0 <= at - row["feed_updated_at"] <= MAX_AGE
            and reported is not None and 0 <= at - reported <= MAX_AGE)


def features_at(rows, times, at, allow_missing_lags=False, flow_groups=None, predict_change=False,
                include_time_band=False):
    index = bisect_right(times, at) - 1
    if index < 0:
        return "missing_history", None, None
    current = rows[index]
    if at - times[index] > MAX_AGE or not 0 <= at - current["feed_updated_at"] <= MAX_AGE:
        return "stale_data", current, None
    if not returnable(current):
        return "returns_unavailable", current, None
    if not fresh(current, at):
        return "stale_data", current, None
    if not dock_count(current):
        return "missing_dock_count", current, None

    earliest_lag_time = at - max(LAGS) * 60 - LAG_TOLERANCE
    earliest = bisect_left(times, earliest_lag_time, hi=index + 1)
    for j in range(index, earliest, -1):
        if times[j] - times[j - 1] > MAX_GAP:
            earliest = j
            break

    lag_values = {}
    for lag in LAGS:
        requested = at - lag * 60
        lag_index = bisect_right(times, requested, earliest, index + 1) - 1
        valid = (lag_index >= earliest and requested - times[lag_index] <= LAG_TOLERANCE
                 and dock_count(rows[lag_index]) and fresh(rows[lag_index], times[lag_index]))
        if not valid and not allow_missing_lags:
            return "missing_history", current, None
        lag_values[f"lag_{lag}"] = rows[lag_index]["available_docks"] if valid else np.nan
        if allow_missing_lags:
            lag_values[f"lag_{lag}_missing"] = int(not valid)
            lag_values[f"lag_{lag}_age_seconds"] = at - times[lag_index] if valid else np.nan

    local = datetime.fromtimestamp(at, LOCAL_TZ)
    angle = 2 * math.pi * (local.hour + local.minute / 60) / 24
    values = {
        "station_id": current["station_id"],
        "available_docks": current["available_docks"],
        **lag_values,
        "change_5": current["available_docks"] - lag_values["lag_5"],
        "hour_sin": math.sin(angle),
        "hour_cos": math.cos(angle),
        "weekday": str(local.weekday()),
        "weekend": int(local.weekday() >= 5),
        **{key: current[key] if current[key] is not None else np.nan
           for key in ("capacity", "latitude", "longitude")},
    }
    if include_time_band:
        values["time_band"] = next(label for label, start, end in TIME_BANDS if start <= local.hour < end)
    if flow_groups is not None:
        values["flow_group"] = flow_groups.get(current["station_id"], "unknown")
    if predict_change:
        change, age = recent_observed_change(rows, times, at, earliest, index)
        values.update(recent_change=change, recent_change_age_seconds=age,
                      recent_change_missing=int(math.isnan(change)))
        if flow_groups is not None and not math.isnan(change):
            values[f"recent_change_flow_group_{values['flow_group']}"] = change
    return "ok", current, values


def recent_observed_change(rows, times, at, earliest, index):
    current = rows[index]["available_docks"]
    first_seen = times[index]
    for previous in range(index - 1, earliest - 1, -1):
        row = rows[previous]
        if not returnable(row) or not dock_count(row) or not fresh(row, times[previous]):
            break
        if row["available_docks"] != current:
            return current - row["available_docks"], at - first_seen
        first_seen = times[previous]
    return np.nan, np.nan


def target_index(rows, times, base, horizon):
    requested = times[base] + horizon * 60
    right = bisect_left(times, requested, lo=base + 1)
    candidates = [i for i in (right - 1, right) if base < i < len(times)]
    if not candidates:
        return None
    index = min(candidates, key=lambda i: (abs(times[i] - requested), times[i]))
    if abs(times[index] - requested) > TARGET_TOLERANCE:
        return None
    if any(times[j] - times[j - 1] > MAX_GAP for j in range(base + 1, index + 1)):
        return None
    row = rows[index]
    if dock_count(row) and returnable(row) and fresh(row, times[index]):
        return index
    return None


def time_splits(history, horizon, boundaries=None, test_end=math.inf, allow_missing_lags=False):
    if boundaries is None:
        timestamps = sorted({row["observed_at"] for rows in history.values() for row in rows})
        if len(timestamps) < 3:
            raise ValueError("Insufficient station-status observations")
        boundaries = chronological_boundaries(timestamps)
    return split_station_histories(history.values(), horizon, boundaries, test_end, allow_missing_lags)


def chronological_boundaries(timestamps):
    return timestamps[int(0.6 * len(timestamps))], timestamps[int(0.8 * len(timestamps))]


def split_station_histories(histories, horizon, boundaries, test_end, allow_missing_lags,
                             flow_groups=None, predict_change=False, include_time_band=False):
    validation, test = boundaries
    if not validation < test < test_end:
        raise ValueError("Expected chronological validation start, test start and test end")
    groups = ([], [], [])
    for rows in histories:
        times = [row["observed_at"] for row in rows]
        for i, at in enumerate(times):
            if at >= test_end:
                break
            split = bisect_right((validation, test), at)
            target = target_index(rows, times, i, horizon)
            if target is None:
                continue
            end = (validation, test, test_end)[split]
            if max(times[target], at + horizon * 60) >= end:
                continue
            status, _, values = features_at(
                rows, times, at, allow_missing_lags, flow_groups, predict_change, include_time_band,
            )
            if status == "ok":
                groups[split].append((values, rows[target]["available_docks"], at, times[target]))
    if not all(groups):
        raise ValueError("Insufficient eligible observations for chronological train/validation/test")
    known = {row[0]["station_id"] for row in groups[0]}
    return tuple([row for row in group if row[0]["station_id"] in known] for group in groups), (
        validation, test,
    )


def bounded(prediction, rows):
    capacities = np.array([row["capacity"] for row in rows], dtype=float)
    upper = np.where(np.isfinite(capacities) & (capacities >= 0), capacities, np.inf)
    return np.clip(prediction, 0, upper)


def metrics(errors):
    if not len(errors):
        return {"n": 0, "mae": None, "rmse": None}
    return {
        "n": len(errors),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
    }


def forecast_docks(model, rows, predict_change=False):
    prediction = model.predict(rows)
    if predict_change:
        prediction = prediction + np.array([row["available_docks"] for row in rows])
    return bounded(prediction, rows)


def evaluate(model, group, predict_change=False):
    rows, targets, _, _ = zip(*group)
    truth = np.array(targets)
    predictions = {
        "persistence": bounded([row["available_docks"] for row in rows], rows),
        "regression": forecast_docks(model, rows, predict_change),
    }
    result = {}
    for name, prediction in predictions.items():
        errors = prediction - truth
        result[name] = {
            "all": metrics(errors),
            "actual_0_to_2": metrics(errors[truth <= 2]),
            "actual_full": int((truth == 0).sum()),
            "actual_full_predicted_at_least_one": int(((truth == 0) & (prediction >= 1)).sum()),
            "actual_at_least_one_predicted_below_one": int(((truth >= 1) & (prediction < 1)).sum()),
        }
    return result


def train(db, horizon, output, allow_simulated=False, boundaries=None, test_end=math.inf,
          allow_missing_lags=False, flow_groups=None, predict_change=False):
    if predict_change and not allow_missing_lags:
        raise ValueError("Change prediction requires --allow-missing-lags")
    if flow_groups is not None and (not flow_groups or not allow_missing_lags):
        raise ValueError("Flow groups require matched station IDs and --allow-missing-lags")
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    with closing(sqlite3.connect(store_uri(db), uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        kind, source = store_identity(conn)
        if kind != "real" and (kind != "simulated" or not allow_simulated):
            raise ValueError("Simulation requires explicit --allow-simulated")
        if boundaries is None:
            timestamps = [row[0] for row in conn.execute(
                "SELECT DISTINCT observed_at FROM station_status ORDER BY observed_at"
            )]
            if len(timestamps) < 3:
                raise ValueError("Insufficient station-status observations")
            boundaries = chronological_boundaries(timestamps)
        histories = (rows for _, rows in station_histories(conn, source, end=test_end))
        (training, validation, test), boundaries = split_station_histories(
            histories, horizon, boundaries, test_end, allow_missing_lags, flow_groups, predict_change
        )
    if not validation or not test:
        raise ValueError("No eligible validation/test observations at trained stations")
    rows, targets, base_times, target_times = zip(*training)
    fit_targets = np.asarray(targets)
    if predict_change:
        fit_targets = fit_targets - np.array([row["available_docks"] for row in rows])
    candidates = []
    for alpha in (1.0, 100.0, 1000.0):
        model = make_pipeline(
            DictVectorizer(),
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(with_mean=False),
            Ridge(alpha=alpha, solver="lsqr"),
        )
        model.fit(rows, fit_targets)
        report = evaluate(model, validation, predict_change)
        candidates.append((report["regression"]["all"]["mae"], alpha, model, report))
    _, alpha, model, validation_report = min(candidates, key=lambda candidate: candidate[0])
    feature_version = OPTIONAL_LAG_FEATURE_VERSION if allow_missing_lags else FEATURE_VERSION
    feature_names = list(FEATURES)
    if allow_missing_lags:
        feature_names.extend(f"lag_{lag}_{suffix}" for lag in LAGS for suffix in ("missing", "age_seconds"))
    if flow_groups is not None:
        feature_version = FLOW_GROUP_FEATURE_VERSION
        feature_names.append("flow_group")
        if all(row["flow_group"] == "unknown" for row in rows):
            raise ValueError("No training stations match the supplied flow groups")
    if predict_change:
        feature_version = CHANGE_FEATURE_VERSION
        feature_names.extend(("recent_change", "recent_change_age_seconds", "recent_change_missing"))
        if flow_groups is not None:
            feature_names.extend(
                f"recent_change_flow_group_{group}"
                for group in sorted(set(flow_groups.values()) | {"unknown"})
            )
    metadata = {
        "data_kind": kind,
        "regression_target": "dock_change" if predict_change else "available_docks",
        "training_data_source": source,
        "horizon_minutes": horizon,
        "feature_version": feature_version,
        "features": feature_names,
        "allow_missing_lags": allow_missing_lags,
        "timing_seconds": {
            "target_tolerance": TARGET_TOLERANCE,
            "lag_tolerance": LAG_TOLERANCE,
            "max_gap": MAX_GAP,
            "max_age": MAX_AGE,
        },
        "sklearn_version": sklearn.__version__,
        "training_base_range": [iso(min(base_times)), iso(max(base_times))],
        "training_label_range": [iso(min(target_times)), iso(max(target_times))],
        "validation_starts": iso(boundaries[0]),
        "test_starts": iso(boundaries[1]),
        "test_ends_exclusive": iso(test_end) if math.isfinite(test_end) else None,
        "test_stations": len({values["station_id"] for values, _, _, _ in test}),
        "training_rows": len(training),
        "ridge_alpha": alpha,
        "validation": validation_report,
        "validation_candidates": [
            {"alpha": alpha_value, "mae": mae} for mae, alpha_value, _, _ in candidates
        ],
        "test": evaluate(model, test, predict_change),
    }
    if allow_missing_lags:
        fully_observed = [
            sample for sample in test if all(sample[0][f"lag_{lag}_missing"] == 0 for lag in LAGS)
        ]
        metadata["test_history_coverage"] = {
            "complete_lag_samples": len(fully_observed),
            "complete_lag_stations": len({sample[0]["station_id"] for sample in fully_observed}),
            "missing_lag_samples": {
                str(lag): sum(sample[0][f"lag_{lag}_missing"] for sample in test) for lag in LAGS
            },
        }
    if flow_groups is not None:
        metadata["flow_group_test_coverage"] = {
            "mapped_samples": sum(sample[0]["flow_group"] != "unknown" for sample in test),
            "mapped_stations": len({
                sample[0]["station_id"] for sample in test if sample[0]["flow_group"] != "unknown"
            }),
        }
    if math.isfinite(test_end):
        by_week = defaultdict(list)
        for sample in test:
            local_date = datetime.fromtimestamp(sample[2], LOCAL_TZ).date()
            monday = local_date - timedelta(days=local_date.weekday())
            by_week[monday.isoformat()].append(sample)
        metadata["test_by_week"] = {
            week: evaluate(model, samples, predict_change) for week, samples in sorted(by_week.items())
        }
    artifact = {"metadata": metadata, "pipeline": model,
                "stations": sorted({row["station_id"] for row in rows})}
    if flow_groups is not None:
        artifact["flow_groups"] = flow_groups
    version = joblib.hash(artifact, hash_name="sha1")
    metadata["model_version"] = f"{kind}-docks-v{feature_version}-{version[:12]}"
    save_artifact(artifact, output)
    return metadata


def save_artifact(artifact, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(artifact, temporary_path)
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def recommendation_threshold(value):
    if not math.isfinite(value) or value < 1:
        raise ValueError("Minimum recommendation docks must be finite and at least 1")
    return value


def return_decisions(actual, predicted, threshold):
    full = actual == 0
    meets_threshold = predicted >= threshold
    false_open = int((full & meets_threshold).sum())
    full_count = int(full.sum())
    return {
        "minimum_recommendation_docks": threshold,
        "samples": len(actual),
        "actual_full": full_count,
        "actual_full_meets_threshold": false_open,
        "full_station_false_open_rate": false_open / full_count if full_count else None,
        "actual_open": int((~full).sum()),
        "actual_open_below_threshold": int((~full & ~meets_threshold).sum()),
        "meets_threshold": int(meets_threshold.sum()),
    }


def select_return_threshold(actual, predicted, max_false_open_rate):
    if not 0 <= max_false_open_rate <= 1:
        raise ValueError("Maximum full-station false-open rate must be within [0, 1]")
    full_predictions = predicted[actual == 0]
    if not len(full_predictions):
        raise ValueError("Threshold calibration requires actual full stations in validation")
    # Select the smallest half-dock step satisfying the validation constraint.
    upper_step = max(2, math.floor(float(full_predictions.max()) * 2) + 1)
    return next(
        step / 2 for step in range(2, upper_step + 1)
        if np.mean(full_predictions >= step / 2) <= max_false_open_rate
    )


def sustained_decline(rows, times, base, minutes):
    """Return true for two observed decreases without a rebound, or None without enough history."""
    requested_start = times[base] - minutes * 60
    first = bisect_right(times, requested_start, hi=base + 1) - 1
    if first < 0 or requested_start - times[first] > LAG_TOLERANCE or base - first < 2:
        return None
    for index in range(first, base + 1):
        row = rows[index]
        if not returnable(row) or not dock_count(row) or not fresh(row, times[index]):
            return None
        if index > first and times[index] - times[index - 1] > MAX_GAP:
            return None
    counts = [rows[index]["available_docks"] for index in range(first, base + 1)]
    return all(after <= before for before, after in zip(counts, counts[1:])) and sum(
        after < before for before, after in zip(counts, counts[1:])
    ) >= 2


def paired_predictions(artifacts, db, ranges, allow_live=False, decline_windows=()):
    metadata = [artifact["metadata"] for artifact in artifacts]
    horizons = {item["horizon_minutes"] for item in metadata}
    if len(horizons) != 1:
        raise ValueError("Paired forecasts must have the same horizon")
    horizon = horizons.pop()
    known = set.intersection(*(set(artifact["stations"]) for artifact in artifacts))
    groups = [[] for _ in ranges]
    with closing(sqlite3.connect(store_uri(db), uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        kind, source = store_identity(conn)
        for item in metadata:
            trained_source = item.get("training_data_source")
            live_transfer = allow_live and source == "official_gbfs" and trained_source == "citybikes"
            if kind != item["data_kind"] or (trained_source and source != trained_source and not live_transfer):
                raise ValueError("Evaluation data must match model provenance or use explicit live GBFS evaluation")
        for station, rows in station_histories(conn, source, end=max(end for _, end in ranges)):
            if station not in known:
                continue
            times = [row["observed_at"] for row in rows]
            for (start, end), group in zip(ranges, groups):
                for base in range(bisect_left(times, start), bisect_left(times, end - horizon * 60)):
                    target = target_index(rows, times, base, horizon)
                    if target is None or times[target] >= end:
                        continue
                    built = [
                        features_at(
                            rows, times, times[base],
                            item["feature_version"] >= OPTIONAL_LAG_FEATURE_VERSION,
                            artifact.get("flow_groups"),
                            item["feature_version"] == CHANGE_FEATURE_VERSION,
                            item.get("include_time_band", False),
                        )
                        for artifact, item in zip(artifacts, metadata)
                    ]
                    if all(status == "ok" for status, _, _ in built):
                        trends = [sustained_decline(rows, times, base, minutes) for minutes in decline_windows]
                        group.append((station, times[base], rows[target]["available_docks"],
                                      [values for _, _, values in built], trends))
    if not all(groups):
        raise ValueError("Insufficient matched observations for paired evaluation")
    result = []
    for group in groups:
        stations, base_times, truth, per_sample, trends = zip(*group)
        matrix = list(zip(*per_sample))
        result.append((
            np.array(truth),
            [artifact["pipeline"].decision_function(features)
             if item.get("model_type") == DOCK_CLASSIFIER_KIND
             else forecast_docks(artifact["pipeline"], features, item["feature_version"] == CHANGE_FEATURE_VERSION)
             for artifact, item, features in zip(artifacts, metadata, matrix)],
            bounded([row["available_docks"] for row in matrix[0]], matrix[0]),
            np.array(stations),
            np.array(base_times),
            np.array(trends, dtype=object),
        ))
    return result


def calibration_predictions(artifact, db, validation_range, test_range):
    if artifact["metadata"].get("model_type") == DOCK_CLASSIFIER_KIND:
        raise ValueError("Dock-count threshold calibration requires a regression model")
    predictions = paired_predictions([artifact], db, (validation_range, test_range))
    return [(truth, models[0], baseline) for truth, models, baseline, _, _, _ in predictions]


def calibrate_return_threshold(artifact, db, test_start, test_end, max_false_open_rate):
    metadata = artifact["metadata"]
    validation = (parse_time(metadata["validation_starts"]), parse_time(metadata["test_starts"]))
    previous_test_end = metadata.get("test_ends_exclusive")
    if previous_test_end is None:
        raise ValueError("Use a model with an explicit previous test end to verify the new holdout")
    if not validation[1] <= parse_time(previous_test_end) <= test_start < test_end:
        raise ValueError("New test period must follow the model's previous test period")
    (actual_val, model_val, baseline_val), (actual_test, model_test, baseline_test) = (
        calibration_predictions(artifact, db, validation, (test_start, test_end))
    )
    report = {
        "validation_range": [iso(time) for time in validation],
        "test_range": [iso(test_start), iso(test_end)],
        "validation_max_full_station_false_open_rate": max_false_open_rate,
    }
    for name, validation_prediction, test_prediction in (
        ("regression", model_val, model_test), ("persistence", baseline_val, baseline_test)
    ):
        threshold = select_return_threshold(actual_val, validation_prediction, max_false_open_rate)
        report[name] = {
            "validation_adjusted": return_decisions(actual_val, validation_prediction, threshold),
            "test_default": return_decisions(actual_test, test_prediction, 1.0),
            "test_adjusted": return_decisions(actual_test, test_prediction, threshold),
        }
    metadata["minimum_recommendation_docks"] = report["regression"]["validation_adjusted"][
        "minimum_recommendation_docks"
    ]
    metadata["return_threshold_calibration"] = report
    return report


def predict(artifact, db, station_id, at, horizon=15, minimum_recommendation_docks=None):
    metadata = artifact["metadata"]
    if metadata["feature_version"] not in (
        FEATURE_VERSION, OPTIONAL_LAG_FEATURE_VERSION, FLOW_GROUP_FEATURE_VERSION, CHANGE_FEATURE_VERSION,
    ):
        raise ValueError("Model feature version is incompatible")
    threshold = recommendation_threshold(
        metadata.get("minimum_recommendation_docks", 1.0)
        if minimum_recommendation_docks is None else minimum_recommendation_docks
    )
    result = {
        "station_id": station_id,
        "minimum_recommendation_docks": threshold,
        "meets_return_threshold": None,
        "data_updated_at": None,
        "station_last_reported_at": None,
        "target_time": iso(at + horizon * 60),
        "predicted_available_docks": None,
        "model_version": metadata["model_version"],
        "training_data_source": metadata.get("training_data_source"),
        "data_kind": metadata["data_kind"],
    }
    if metadata["horizon_minutes"] != horizon:
        return {**result, "status": "unsupported_horizon"}
    if station_id not in artifact["stations"]:
        return {**result, "status": "unknown_station"}
    kind, history = load_history(db, station_id, at)
    if kind != metadata["data_kind"]:
        return {**result, "status": "data_kind_mismatch"}
    rows = history.get(station_id, [])
    times = [row["observed_at"] for row in rows]
    status, current, values = features_at(
        rows, times, at, metadata["feature_version"] >= OPTIONAL_LAG_FEATURE_VERSION,
        artifact.get("flow_groups"), metadata["feature_version"] == CHANGE_FEATURE_VERSION,
    )
    if current is None:
        # Distinguish a long outage from a request before the station's first observation.
        with closing(sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            latest = conn.execute(
                "SELECT feed_updated_at, last_reported FROM station_status WHERE station_id=? AND observed_at<=?"
                " ORDER BY observed_at DESC LIMIT 1", (station_id, at),
            ).fetchone()
        if latest is not None:
            result["data_updated_at"] = iso(latest[0])
            if latest[1] is not None:
                result["station_last_reported_at"] = iso(latest[1])
            status = "stale_data"
    else:
        result["data_updated_at"] = iso(current["feed_updated_at"])
        if current["last_reported"] is not None:
            result["station_last_reported_at"] = iso(current["last_reported"])
    if status == "ok":
        if "flow_group" in values:
            result["flow_group"] = values["flow_group"]
        result["predicted_available_docks"] = float(
            forecast_docks(
                artifact["pipeline"], [values], metadata["feature_version"] == CHANGE_FEATURE_VERSION
            )[0]
        )
        result["meets_return_threshold"] = result["predicted_available_docks"] >= threshold
    return {**result, "status": status}


def load_flow_groups(classifier_path, station_information):
    from collect_status import station_map

    groups_by_short_name = joblib.load(classifier_path)["station_groups_by_trip_id"]
    return {
        station_id: str(groups_by_short_name[short_name])
        for station_id, short_name in station_map(Path(station_information)).items()
        if short_name in groups_by_short_name
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--db", required=True)
    fit.add_argument("--model", required=True)
    fit.add_argument("--horizon", type=int, default=15)
    fit.add_argument("--allow-simulated", action="store_true")
    fit.add_argument("--allow-missing-lags", action="store_true")
    fit.add_argument("--predict-change", action="store_true")
    fit.add_argument("--flow-classifier")
    fit.add_argument("--station-information")
    fit.add_argument("--validation-start")
    fit.add_argument("--test-start")
    fit.add_argument("--test-end")
    report = commands.add_parser("evaluate")
    report.add_argument("--model", required=True)
    calibration = commands.add_parser("calibrate-return-threshold")
    calibration.add_argument("--db", required=True)
    calibration.add_argument("--model", required=True)
    calibration.add_argument("--output-model", required=True)
    calibration.add_argument("--test-start", required=True)
    calibration.add_argument("--test-end", required=True)
    calibration.add_argument("--max-full-station-false-open-rate", type=float, required=True)
    forecast = commands.add_parser("predict")
    forecast.add_argument("--model", required=True)
    forecast.add_argument("--db", required=True)
    forecast.add_argument("--station-id", required=True)
    forecast.add_argument("--at", required=True)
    forecast.add_argument("--horizon", type=int, default=15)
    forecast.add_argument("--minimum-recommendation-docks", type=float)
    args = parser.parse_args()
    if getattr(args, "horizon", 15) < 1:
        parser.error("--horizon must be a positive number of minutes")
    try:
        if args.command == "train":
            if bool(args.validation_start) != bool(args.test_start):
                parser.error("--validation-start and --test-start must be supplied together")
            boundaries = (parse_time(args.validation_start), parse_time(args.test_start)) if args.test_start else None
            test_end = parse_time(args.test_end) if args.test_end else math.inf
            if bool(args.flow_classifier) != bool(args.station_information):
                parser.error("--flow-classifier and --station-information must be supplied together")
            flow_groups = (load_flow_groups(args.flow_classifier, args.station_information)
                           if args.flow_classifier else None)
            result = train(
                args.db, args.horizon, args.model, args.allow_simulated, boundaries, test_end,
                args.allow_missing_lags, flow_groups, args.predict_change,
            )
        else:
            artifact = joblib.load(args.model)
            if args.command == "evaluate":
                result = artifact["metadata"]
            elif args.command == "calibrate-return-threshold":
                if Path(args.model).resolve() == Path(args.output_model).resolve():
                    raise ValueError("Use a separate output model for the calibrated threshold")
                result = calibrate_return_threshold(
                    artifact, args.db, parse_time(args.test_start), parse_time(args.test_end),
                    args.max_full_station_false_open_rate,
                )
                save_artifact(artifact, args.output_model)
            else:
                result = predict(artifact, args.db, args.station_id, parse_time(args.at), args.horizon,
                                 args.minimum_recommendation_docks)
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("status", "ok") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
