"""Tune and compare low-dock and downward-trend ranking rules."""

import argparse
from datetime import datetime
import json

import joblib
import numpy as np

from dock_regression import (
    DOCK_CLASSIFIER_KIND,
    LOCAL_TZ,
    iso,
    paired_predictions,
    parse_time,
)

COVERAGE = 0.8
WINDOWS = (5, 10)
LOW_CUTOFFS = (2, 3, 5)
TREND_PENALTIES = (1, 2, 3)


def rank(scores, ties, demoted=None):
    keys = (ties, -scores) if demoted is None else (ties, -scores, demoted)
    return np.lexsort(keys)


def cases(actual, current, order):
    count = int(len(actual) * COVERAGE)
    selected = np.zeros(len(actual), dtype=bool)
    selected[order[:count]] = True
    full = actual == 0
    groups = {
        "0_to_2_current_docks": current <= 2,
        "3_to_5_current_docks": (current >= 3) & (current <= 5),
        "6_or_more_current_docks": current >= 6,
    }
    return {
        "selected": count,
        "actual_full_selected": int((full & selected).sum()),
        "actual_open_not_selected": int((~full & ~selected).sum()),
        "by_current_docks": {
            key: {
                "actual_full": int((full & subset).sum()),
                "actual_full_selected": int((full & selected & subset).sum()),
            }
            for key, subset in groups.items()
        },
    }


def evaluate(window, low_cutoffs, trend_parameters):
    actual, (logistic,), current, stations, times, trends = window
    tie = np.random.default_rng(42).permutation(len(actual))
    methods = {
        "current_docks": cases(actual, current, rank(current, tie)),
        "logistic": cases(actual, current, rank(logistic, tie)),
    }
    for cutoff in low_cutoffs:
        methods[f"logistic_demote_{cutoff}_or_fewer"] = cases(
            actual, current, rank(logistic, tie, current <= cutoff)
        )
    for minutes, penalty in trend_parameters:
        declining = np.fromiter(
            (value is True for value in trends[:, WINDOWS.index(minutes)]), dtype=bool
        )
        methods[f"current_docks_minus_{penalty}_on_{minutes}min_decline"] = cases(
            actual, current, rank(current - penalty * declining, tie)
        )

    hours = np.array([datetime.fromtimestamp(int(at), LOCAL_TZ).hour for at in times])
    late = (hours >= 16) & (hours < 20)
    signal_counts = {}
    for column, minutes in enumerate(WINDOWS):
        observed = np.fromiter(
            (value is not None for value in trends[:, column]), dtype=bool
        )
        declining = np.fromiter(
            (value is True for value in trends[:, column]), dtype=bool
        )
        signal_counts[str(minutes)] = {}
        for name, subset in (
            ("all", np.ones(len(actual), dtype=bool)),
            ("16:00–20:00", late),
        ):
            signal_counts[str(minutes)][name] = {
                "samples_with_usable_history": int((subset & observed).sum()),
                "declining_samples": int((subset & declining).sum()),
                "full_after_decline": int((subset & declining & (actual == 0)).sum()),
                "stable_or_rising_samples": int((subset & observed & ~declining).sum()),
                "full_after_stable_or_rising": int(
                    (subset & observed & ~declining & (actual == 0)).sum()
                ),
            }
    return {
        "samples": len(actual),
        "stations": len(set(stations)),
        "actual_full": int((actual == 0).sum()),
        "methods": methods,
        "decline_signal": signal_counts,
    }


def compare(classifier, validation_db, test_db, start, end, allow_live=False):
    metadata = classifier["metadata"]
    if metadata.get("model_type") != DOCK_CLASSIFIER_KIND:
        raise ValueError("Expected an availability classifier")
    val_start = parse_time(metadata["validation_starts"])
    val_end = parse_time(metadata["test_starts"])
    if not val_end <= start < end:
        raise ValueError("Test period must start after validation")
    (validation,) = paired_predictions(
        [classifier], validation_db, [(val_start, val_end)], decline_windows=WINDOWS
    )
    (test,) = paired_predictions(
        [classifier],
        test_db,
        [(start, end)],
        allow_live=allow_live,
        decline_windows=WINDOWS,
    )
    trend_parameters = [
        (minutes, penalty) for minutes in WINDOWS for penalty in TREND_PENALTIES
    ]
    val_report = evaluate(validation, LOW_CUTOFFS, trend_parameters)
    val_methods = val_report["methods"]
    if not val_report["actual_full"]:
        raise ValueError("Validation needs actual full observations")
    low = min(
        LOW_CUTOFFS,
        key=lambda cutoff: val_methods[f"logistic_demote_{cutoff}_or_fewer"][
            "actual_full_selected"
        ],
    )
    trend = min(
        trend_parameters,
        key=lambda pair: val_methods[
            f"current_docks_minus_{pair[1]}_on_{pair[0]}min_decline"
        ]["actual_full_selected"],
    )
    return {
        "coverage": COVERAGE,
        "validation_range": [iso(val_start), iso(val_end)],
        "test_range": [iso(start), iso(end)],
        "selected_low_dock_cutoff": low,
        "selected_trend_window_minutes": trend[0],
        "selected_trend_penalty_docks": trend[1],
        "validation": val_report,
        "test": evaluate(test, [low], [trend]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--validation-db", required=True)
    parser.add_argument("--test-db", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--allow-live-gbfs", action="store_true")
    args = parser.parse_args()
    try:
        report = compare(
            joblib.load(args.classifier),
            args.validation_db,
            args.test_db,
            parse_time(args.start),
            parse_time(args.end),
            args.allow_live_gbfs,
        )
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
