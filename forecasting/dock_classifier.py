"""Directly classify 15-minute return availability from Bluebikes status history."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_return_coverage import coverage_results
from dock_regression import (
    CHANGE_FEATURE_VERSION,
    DOCK_CLASSIFIER_KIND,
    bounded,
    features_at,
    forecast_docks,
    iso,
    load_history,
    parse_time,
    save_artifact,
    split_station_histories,
    station_histories,
    store_identity,
    store_uri,
)


def separate_rows(samples, known):
    matched = [item for item in samples if item[0]["station_id"] in known]
    if not matched:
        raise ValueError("No samples at stations supported by the reference regression")
    features, counts, _, _ = zip(*matched)
    return features, np.asarray(counts)


def availability_classifier(c):
    return make_pipeline(
        DictVectorizer(),
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=c, solver="liblinear", class_weight="balanced", max_iter=500
        ),
    )


def evaluation(actual, recommended):
    full = actual == 0
    return {
        "samples": len(actual),
        "recommended": int(recommended.sum()),
        "actual_full": int(full.sum()),
        "actual_full_recommended": int((recommended & full).sum()),
        "actual_open": int((~full).sum()),
        "actual_open_not_recommended": int((~recommended & ~full).sum()),
    }


def train(db, reference, output, validation_coverage=0.8, include_time_band=False):
    if not 0 < validation_coverage <= 1:
        raise ValueError("Validation coverage must be greater than 0 and at most 1")
    ref = reference["metadata"]
    if ref["feature_version"] != CHANGE_FEATURE_VERSION or not ref.get(
        "test_ends_exclusive"
    ):
        raise ValueError("Use the change-regression artifact with explicit split dates")
    boundaries = tuple(
        parse_time(ref[key]) for key in ("validation_starts", "test_starts")
    )
    end = parse_time(ref["test_ends_exclusive"])
    with closing(sqlite3.connect(store_uri(db), uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        kind, source = store_identity(conn)
        if kind != ref["data_kind"] or source != ref["training_data_source"]:
            raise ValueError(
                "Training database must match the reference regression data source"
            )
        histories = (rows for _, rows in station_histories(conn, source, end=end))
        (training, validation, test), _ = split_station_histories(
            histories,
            ref["horizon_minutes"],
            boundaries,
            end,
            True,
            reference.get("flow_groups"),
            True,
            include_time_band,
        )
    known = set(reference["stations"])
    x_train, y_train = separate_rows(training, known)
    x_validation, y_validation = separate_rows(validation, known)
    x_test, y_test = separate_rows(test, known)
    has_dock = y_train >= 1
    if len(np.unique(has_dock)) != 2 or len(np.unique(y_validation >= 1)) != 2:
        raise ValueError(
            "Training and validation each require both full and available examples"
        )

    candidates = []
    for c in (0.01, 0.1, 1.0):
        pipeline = availability_classifier(c)
        pipeline.fit(x_train, has_dock)
        scores = pipeline.decision_function(x_validation)
        full_ap = float(average_precision_score(y_validation == 0, -scores))
        candidates.append((full_ap, c, pipeline, scores))
    _, c, pipeline, validation_scores = max(
        candidates, key=lambda candidate: candidate[0]
    )
    # Fit the runtime operating point on validation only.
    count = max(1, int(len(validation_scores) * validation_coverage))
    decision_cutoff = float(np.sort(validation_scores)[-count])
    test_scores = pipeline.decision_function(x_test)
    occupancy_scores = {
        "persistence": bounded([row["available_docks"] for row in x_test], x_test),
        "change_regression": forecast_docks(reference["pipeline"], x_test, True),
        "logistic": test_scores,
    }
    fractions = (0.8, 0.9)
    selections = [max(1, int(len(y_test) * fraction)) for fraction in fractions]
    tie_order = np.random.default_rng(42).permutation(len(y_test))
    report = {
        "model_type": DOCK_CLASSIFIER_KIND,
        "feature_version": CHANGE_FEATURE_VERSION,
        "data_kind": kind,
        "training_data_source": source,
        "horizon_minutes": ref["horizon_minutes"],
        "features": (
            [*ref["features"], "time_band"] if include_time_band else ref["features"]
        ),
        "include_time_band": include_time_band,
        "training_base_range": ref["training_base_range"],
        "validation_starts": ref["validation_starts"],
        "test_starts": ref["test_starts"],
        "test_ends_exclusive": ref["test_ends_exclusive"],
        "reference_regression_version": ref["model_version"],
        "training_rows": len(x_train),
        "training_actual_full": int((y_train == 0).sum()),
        "test_stations": len({row["station_id"] for row in x_test}),
        "c": c,
        "validation_full_average_precision_by_c": [
            {"c": candidate[1], "full_average_precision": candidate[0]}
            for candidate in candidates
        ],
        "validation_requested_coverage": validation_coverage,
        "decision_cutoff": decision_cutoff,
        "validation_at_cutoff": evaluation(
            y_validation, validation_scores >= decision_cutoff
        ),
        "test_at_cutoff": evaluation(y_test, test_scores >= decision_cutoff),
        "equal_coverage_fractions": fractions,
        "test_equal_coverage": {
            method: coverage_results(y_test, scores, selections, tie_order)
            for method, scores in occupancy_scores.items()
        },
    }
    artifact = {
        "metadata": report,
        "pipeline": pipeline,
        "stations": sorted({row["station_id"] for row in x_train}),
        "flow_groups": reference.get("flow_groups"),
    }
    report["model_version"] = (
        "logistic-docks-v1-" + joblib.hash(artifact, hash_name="sha1")[:12]
    )
    save_artifact(artifact, output)
    return report


def predict(artifact, db, station, at, horizon=15):
    meta = artifact["metadata"]
    if meta.get("model_type") != DOCK_CLASSIFIER_KIND:
        raise ValueError("Expected a dock availability classifier")
    result = {
        "station_id": station,
        "data_updated_at": None,
        "target_time": iso(at + horizon * 60),
        "predicted_has_available_dock": None,
        "model_version": meta["model_version"],
    }
    if horizon != meta["horizon_minutes"]:
        return {**result, "status": "unsupported_horizon"}
    if station not in artifact["stations"]:
        return {**result, "status": "unknown_station"}
    kind, history = load_history(db, station, at)
    if kind != meta["data_kind"]:
        return {**result, "status": "data_kind_mismatch"}
    rows = history.get(station, [])
    status, current, features = features_at(
        rows,
        [row["observed_at"] for row in rows],
        at,
        True,
        artifact["flow_groups"],
        True,
        meta.get("include_time_band", False),
    )
    if current is None:
        with closing(sqlite3.connect(store_uri(db), uri=True)) as conn:
            last = conn.execute(
                "SELECT feed_updated_at FROM station_status WHERE station_id=? AND observed_at<=?"
                " ORDER BY observed_at DESC LIMIT 1",
                (station, at),
            ).fetchone()
        if last is not None:
            status = "stale_data"
            result["data_updated_at"] = iso(last[0])
    else:
        result["data_updated_at"] = iso(current["feed_updated_at"])
    if status == "ok":
        if "time_band" in features:
            result["time_band"] = features["time_band"]
        score = artifact["pipeline"].decision_function([features])[0]
        result["predicted_has_available_dock"] = bool(score >= meta["decision_cutoff"])
    return {**result, "status": status}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--db", required=True)
    fit.add_argument("--reference-regression", required=True)
    fit.add_argument("--model", required=True)
    fit.add_argument("--validation-coverage", type=float, default=0.8)
    fit.add_argument("--time-band", action="store_true")
    show = commands.add_parser("evaluate")
    show.add_argument("--model", required=True)
    inference = commands.add_parser("predict")
    inference.add_argument("--db", required=True)
    inference.add_argument("--model", required=True)
    inference.add_argument("--station-id", required=True)
    inference.add_argument("--at", required=True)
    inference.add_argument("--horizon", type=int, default=15)
    args = parser.parse_args()
    try:
        if args.command == "train":
            if Path(args.reference_regression).resolve() == Path(args.model).resolve():
                raise ValueError("Use a separate output path for the classifier")
            result = train(
                args.db,
                joblib.load(args.reference_regression),
                args.model,
                args.validation_coverage,
                args.time_band,
            )
        else:
            artifact = joblib.load(args.model)
            if artifact["metadata"].get("model_type") != DOCK_CLASSIFIER_KIND:
                raise ValueError("Expected a dock availability classifier")
            result = (
                artifact["metadata"]
                if args.command == "evaluate"
                else predict(
                    artifact,
                    args.db,
                    args.station_id,
                    parse_time(args.at),
                    args.horizon,
                )
            )
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("status", "ok") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
