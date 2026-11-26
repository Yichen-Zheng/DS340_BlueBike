"""Offline calibration and evaluation for 15-minute dock availability."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from dock_regression import (
    DOCK_CLASSIFIER_KIND,
    LOCAL_TZ,
    iso,
    paired_predictions,
    parse_time,
    save_artifact,
)

KIND = "dock_probability_calibration_v1"
BIN_EDGES = (0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
CURRENT_DOCK_EDGES = (1, 3, 6)


def observed_rate(actual):
    return float((actual.sum() + 0.5) / (len(actual) + 1))


def fit_probability_models(actual, scores, current):
    if len(np.unique(actual)) != 2:
        raise ValueError("Calibration requires both full and available examples")
    sigmoid = LogisticRegression(C=1e6, solver="lbfgs").fit(
        scores.reshape(-1, 1), actual
    )
    dock_groups = np.searchsorted(CURRENT_DOCK_EDGES, current, side="right")
    global_rate = observed_rate(actual)
    rates = []
    for index in range(len(CURRENT_DOCK_EDGES) + 1):
        group = actual[dock_groups == index]
        rates.append(observed_rate(group) if len(group) else global_rate)
    return sigmoid, global_rate, rates


def probability_metrics(actual, probability):
    n = len(actual)
    if not n:
        return {
            "samples": 0,
            "actual_full": 0,
            "brier": None,
            "log_loss": None,
            "calibration_gap": None,
            "ece": None,
            "reliability": [],
        }
    safe = np.clip(probability, 1e-15, 1 - 1e-15)
    groups = np.searchsorted(BIN_EDGES, probability, side="right")
    boundaries = (0.0, *BIN_EDGES, 1.0)
    reliability = []
    for index in np.unique(groups):
        mask = groups == index
        reliability.append(
            {
                "range": [boundaries[index], boundaries[index + 1]],
                "samples": int(mask.sum()),
                "mean_prediction": float(probability[mask].mean()),
                "observed_availability": float(actual[mask].mean()),
            }
        )
    return {
        "samples": n,
        "actual_full": int((~actual).sum()),
        "mean_prediction": float(probability.mean()),
        "observed_availability": float(actual.mean()),
        "brier": float(np.mean((probability - actual) ** 2)),
        "log_loss": float(-np.mean(np.where(actual, np.log(safe), np.log1p(-safe)))),
        "calibration_gap": float(probability.mean() - actual.mean()),
        "ece": sum(
            row["samples"] * abs(row["mean_prediction"] - row["observed_availability"])
            for row in reliability
        )
        / n,
        "reliability": reliability,
    }


def report(artifact, test, test_start, test_end):
    count, (scores,), current, stations, times, _ = test
    actual = count >= 1
    calibrated = artifact["sigmoid"].predict_proba(scores.reshape(-1, 1))[:, 1]
    bin_index = np.searchsorted(CURRENT_DOCK_EDGES, current, side="right")
    probabilities = {
        "calibrated_logistic": calibrated,
        "calibration_base_rate": np.full(len(actual), artifact["global_rate"]),
        "current_dock_bin": np.asarray(artifact["current_dock_rates"])[bin_index],
    }
    hours = np.array([datetime.fromtimestamp(int(at), LOCAL_TZ).hour for at in times])
    groups = {
        "all": np.ones(len(actual), dtype=bool),
        "late_peak_16_to_20": (hours >= 16) & (hours < 20),
        "current_0_to_2_docks": current <= 2,
    }
    return {
        "range": [iso(test_start), iso(test_end)],
        "stations": len(set(stations)),
        "groups": {
            name: {
                model: probability_metrics(actual[mask], predicted[mask])
                for model, predicted in probabilities.items()
            }
            for name, mask in groups.items()
        },
    }


def calibrate(classifier, db, calibration_range, test_range):
    metadata = classifier["metadata"]
    if metadata.get("model_type") != DOCK_CLASSIFIER_KIND:
        raise ValueError("Expected an availability classifier")
    previous_end = metadata.get("test_ends_exclusive")
    if previous_end is None or not (
        parse_time(previous_end)
        <= calibration_range[0]
        < calibration_range[1]
        <= test_range[0]
        < test_range[1]
    ):
        raise ValueError(
            "Calibration and test must be disjoint and follow the classifier's previous test"
        )
    calibration, test = paired_predictions(
        [classifier], db, (calibration_range, test_range)
    )
    count, (scores,), current, stations, _, _ = calibration
    actual = count >= 1
    sigmoid, global_rate, rates = fit_probability_models(actual, scores, current)
    artifact = {
        "model_type": KIND,
        "classifier": classifier,
        "sigmoid": sigmoid,
        "global_rate": global_rate,
        "current_dock_rates": rates,
        "calibration_range": [iso(at) for at in calibration_range],
        "calibration_samples": len(count),
        "calibration_actual_full": int((~actual).sum()),
        "calibration_stations": len(set(stations)),
        "classifier_version": metadata["model_version"],
    }
    result = describe(artifact)
    result["test"] = report(artifact, test, *test_range)
    return artifact, result


def describe(artifact):
    return {
        "model_type": artifact["model_type"],
        "classifier_version": artifact["classifier_version"],
        "calibration_range": artifact["calibration_range"],
        "calibration_samples": artifact["calibration_samples"],
        "calibration_stations": artifact["calibration_stations"],
        "calibration_actual_full": artifact["calibration_actual_full"],
        "sigmoid_slope": float(artifact["sigmoid"].coef_[0, 0]),
        "sigmoid_intercept": float(artifact["sigmoid"].intercept_[0]),
        "base_rate": artifact["global_rate"],
        "current_dock_bin_edges": CURRENT_DOCK_EDGES,
        "current_dock_bin_rates": artifact["current_dock_rates"],
    }


def evaluate(artifact, db, test_range, allow_live=False):
    if artifact.get("model_type") != KIND:
        raise ValueError("Expected an offline probability calibration artifact")
    if (
        not parse_time(artifact["calibration_range"][1])
        <= test_range[0]
        < test_range[1]
    ):
        raise ValueError("Test must follow the calibration period")
    (test,) = paired_predictions(
        [artifact["classifier"]], db, (test_range,), allow_live=allow_live
    )
    return {**describe(artifact), "test": report(artifact, test, *test_range)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("calibrate")
    fit.add_argument("--classifier", type=Path, required=True)
    fit.add_argument("--output-model", type=Path, required=True)
    fit.add_argument("--calibration-start", required=True)
    fit.add_argument("--calibration-end", required=True)
    check = commands.add_parser("evaluate")
    check.add_argument("--model", type=Path, required=True)
    check.add_argument("--allow-live-gbfs", action="store_true")
    for command in (fit, check):
        command.add_argument("--db", required=True)
        command.add_argument("--test-start", required=True)
        command.add_argument("--test-end", required=True)
    args = parser.parse_args()
    try:
        test_range = parse_time(args.test_start), parse_time(args.test_end)
        if args.command == "calibrate":
            if args.classifier.resolve() == args.output_model.resolve():
                raise ValueError(
                    "Use a separate output path for the calibration artifact"
                )
            artifact, result = calibrate(
                joblib.load(args.classifier),
                args.db,
                (parse_time(args.calibration_start), parse_time(args.calibration_end)),
                test_range,
            )
            save_artifact(artifact, args.output_model)
        else:
            result = evaluate(
                joblib.load(args.model), args.db, test_range, args.allow_live_gbfs
            )
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
