"""Compare full-station errors at the same number of recommended observations."""
import argparse
from datetime import datetime
import json

import joblib
import numpy as np

from dock_regression import (
    DOCK_CLASSIFIER_KIND, LOCAL_TZ, TIME_BANDS, iso, metrics, paired_predictions, parse_time,
)

def coverage_results(actual, predicted, selections, tie_order):
    results = []
    actual_full = actual == 0
    ranked = np.lexsort((tie_order, -predicted))
    for count in selections:
        selected = ranked[:count]
        false_open = int(actual_full[selected].sum())
        results.append({
            "selected": count,
            "actual_full_selected": false_open,
            "actual_open_not_selected": int((~actual_full).sum()) - (count - false_open),
        })
    return results


def summarize_period(actual, station_ids, scores, coverage, tie_order):
    selections = [max(1, int(len(actual) * fraction)) if len(actual) else 0 for fraction in coverage]
    methods = {}
    for name, predicted in scores.items():
        method = {"equal_coverage": coverage_results(actual, predicted, selections, tie_order)}
        if name != "logistic":
            errors = predicted - actual
            method["all"] = metrics(errors)
            method["actual_0_to_2"] = metrics(errors[actual <= 2])
        methods[name] = method
    return {
        "samples": len(actual),
        "stations": len(set(station_ids)),
        "actual_full": int((actual == 0).sum()),
        "actual_close_to_full": int((actual <= 2).sum()),
        "methods": methods,
    }


def compare(db, direct, change, start, end, coverage, allow_live=False, classifier=None,
            by_time_of_day=False):
    if not start < end:
        raise ValueError("Test start must precede test end")
    if not coverage or any(not 0 < fraction <= 1 for fraction in coverage):
        raise ValueError("Coverage must be greater than 0 and at most 1")
    artifacts = [direct, change]
    if classifier is not None:
        if classifier["metadata"].get("model_type") != DOCK_CLASSIFIER_KIND:
            raise ValueError("Expected the availability classifier with --classifier")
        artifacts.append(classifier)
    for artifact in artifacts:
        if start < parse_time(artifact["metadata"]["test_starts"]):
            raise ValueError("Comparison cannot use model training or validation time")
    (actual, forecasts, baseline, station_ids, base_times, _), = paired_predictions(
        artifacts, db, [(start, end)], allow_live=allow_live,
    )
    scores = {
        "persistence": baseline,
        "direct_regression": forecasts[0],
        "change_regression": forecasts[1],
    }
    if classifier is not None:
        scores["logistic"] = forecasts[2]
    # Break equal scores identically across methods without depending on observed outcomes.
    tie_order = np.random.default_rng(42).permutation(len(actual))
    result = {
        "start": iso(start),
        "end_exclusive": iso(end),
        "requested_coverage": coverage,
        **summarize_period(actual, station_ids, scores, coverage, tie_order),
    }
    if by_time_of_day:
        local_hours = np.array([datetime.fromtimestamp(int(at), LOCAL_TZ).hour for at in base_times])
        result["time_zone"] = LOCAL_TZ.key
        result["hourly_samples"] = np.bincount(local_hours, minlength=24).tolist()
        result["by_time_of_day"] = {}
        for label, first_hour, end_hour in TIME_BANDS:
            mask = (local_hours >= first_hour) & (local_hours < end_hour)
            result["by_time_of_day"][label] = summarize_period(
                actual[mask], station_ids[mask],
                {name: values[mask] for name, values in scores.items()}, coverage, tie_order[mask],
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--direct-model", required=True)
    parser.add_argument("--change-model", required=True)
    parser.add_argument("--classifier")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--coverage", type=float, nargs="+", default=[0.8, 0.9])
    parser.add_argument("--allow-live-gbfs", action="store_true")
    parser.add_argument("--by-time-of-day", action="store_true")
    args = parser.parse_args()
    try:
        result = compare(
            args.db, joblib.load(args.direct_model), joblib.load(args.change_model),
            parse_time(args.start), parse_time(args.end), args.coverage, args.allow_live_gbfs,
            joblib.load(args.classifier) if args.classifier else None, args.by_time_of_day,
        )
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
