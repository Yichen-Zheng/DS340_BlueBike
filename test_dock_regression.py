"""Repeatable smoke run and targeted timing, availability and artifact checks."""
from contextlib import closing
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from calibrate_dock_probability import fit_probability_models, probability_metrics
from collect_status import open_store, snapshot
from dock_classifier import predict as predict_availability, train as train_availability
from dock_regression import (
    bounded, features_at, load_history, parse_time, predict, target_index, time_splits, train,
    return_decisions, select_return_threshold, sustained_decline,
)

START = int(parse_time("2026-07-01T12:00:00Z"))
DEMO_TIME = START + 10 * 3600


def create_demo(path):
    with closing(open_store(path, "simulated")) as db, db:
        db.execute("DELETE FROM station_status")
        for offset, station in enumerate(("SIM-A", "SIM-B", "SIM-C")):
            for minute in range(720):
                at = START + minute * 60
                docks = round(7 + 6 * math.sin(minute / 22 + offset))
                db.execute("INSERT INTO station_status VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                    station, at, at, at, docks, 17 - docks, 1, 1,
                    int(minute != 610), 20, 42.35 + offset * 0.01, -71.08,
                ))


class RegressionTests(unittest.TestCase):
    def test_target_uses_seconds_and_skips_outages(self):
        times = [0, 100, 240, 380, 520, 660, 800, 940]
        rows = [dict(available_docks=4, is_installed=1, is_returning=1, feed_updated_at=t, last_reported=t)
                for t in times]
        self.assertEqual(target_index(rows, times, 0, 15), 7)
        self.assertIsNone(target_index(rows[::2] + rows[-1:], times[::2] + times[-1:], 0, 15))
        self.assertIsNone(target_index(rows, times, 0, 18))
        self.assertEqual(parse_time("2026-07-01T08:00:00-04:00"), START)
        with self.assertRaises(ValueError):
            parse_time("2026-07-01T08:00:00")

    def test_collection_keeps_missing_distinct_from_zero_and_deduplicates(self):
        discovery = {"data": {"en": {"feeds": [
            {"name": "station_status", "url": "status"},
            {"name": "station_information", "url": "info"},
        ]}}}
        status = {"last_updated": START, "data": {"stations": [
            {"station_id": "A", "num_docks_available": 0, "is_returning": 1},
            {"station_id": "B", "num_bikes_available": 5, "is_returning": 0},
        ]}}
        info = {"data": {"stations": [{"station_id": "A", "capacity": 20}]}}
        with closing(open_store(":memory:", "simulated")) as db:
            with patch("collect_status.fetch", side_effect=[discovery, info, status] * 2):
                with patch("collect_status.time.time", return_value=START):
                    snapshot(db)
                    snapshot(db)
            self.assertEqual(db.execute(
                "SELECT available_docks, available_bikes, capacity FROM station_status ORDER BY station_id"
            ).fetchall(), [(0, None, 20), (None, 5, None)])

    def test_optional_lags_preserve_zero_and_ignore_history_before_a_gap(self):
        at = DEMO_TIME
        times = [at + 60 * minute for minute in (-30, -15, -6, -5, -4, -3, -2, -1, 0, 1)]
        rows = [
            {
                "station_id": "SIM-A", "available_docks": 0 if timestamp == at - 300 else 4,
                "feed_updated_at": timestamp, "last_reported": timestamp,
                "is_installed": 1, "is_returning": 1, "capacity": 20,
                "latitude": 42.35, "longitude": -71.08,
            }
            for timestamp in times
        ]
        self.assertEqual(features_at(rows, times, at)[0], "missing_history")
        status, current, features = features_at(rows, times, at, allow_missing_lags=True)
        self.assertEqual(status, "ok")
        self.assertEqual(current["feed_updated_at"], at)
        self.assertEqual(features["lag_5"], 0)
        self.assertEqual(features["lag_5_age_seconds"], 300)
        self.assertEqual(features["lag_5_missing"], 0)
        self.assertEqual(features["change_5"], 4)
        for lag in (15, 30):
            self.assertEqual(features[f"lag_{lag}_missing"], 1)
            self.assertTrue(math.isnan(features[f"lag_{lag}"]))
            self.assertTrue(math.isnan(features[f"lag_{lag}_age_seconds"]))
        self.assertEqual(features_at(rows[:-1], times[:-1], at, True)[2], features)

    def test_sustained_decline_needs_two_drops_and_no_rebound(self):
        at = DEMO_TIME
        times = [at + offset for offset in (-300, -180, -60, 0, 60)]
        rows = [
            {"available_docks": value, "is_installed": 1, "is_returning": 1,
             "feed_updated_at": timestamp, "last_reported": timestamp}
            for value, timestamp in zip((8, 7, 7, 5, 9), times)
        ]
        self.assertIs(sustained_decline(rows, times, 3, 5), True)
        self.assertIs(sustained_decline(rows[:-1], times[:-1], 3, 5), True)
        self.assertIs(sustained_decline(rows, times, 3, 10), None)
        rows[2]["available_docks"] = 9
        self.assertIs(sustained_decline(rows, times, 3, 5), False)

    def test_recent_change_uses_first_observation_of_new_count(self):
        at = DEMO_TIME
        times = [at - 180, at - 120, at - 60, at, at + 60]
        rows = [
            {
                "station_id": "SIM-A", "available_docks": docks, "feed_updated_at": timestamp,
                "last_reported": timestamp, "is_installed": 1, "is_returning": 1,
                "capacity": 20, "latitude": 42.35, "longitude": -71.08,
            }
            for timestamp, docks in zip(times, (8, 8, 5, 5, 2))
        ]
        _, _, features = features_at(rows, times, at, True, {"SIM-A": "0"}, True)
        self.assertEqual(features["recent_change"], -3)
        self.assertEqual(features["recent_change_age_seconds"], 60)
        self.assertEqual(features["recent_change_flow_group_0"], -3)
        without_future = features_at(rows[:-1], times[:-1], at, True, {"SIM-A": "0"}, True)[2]
        self.assertEqual(without_future.keys(), features.keys())
        for key, value in features.items():
            if isinstance(value, float) and math.isnan(value):
                self.assertTrue(math.isnan(without_future[key]))
            else:
                self.assertEqual(without_future[key], value)

    def test_probability_calibration_and_metrics(self):
        actual = np.array([False, False, True, False, True, True])
        scores = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
        current = np.array([0, 0, 2, 2, 7, 7])
        sigmoid, base_rate, bins = fit_probability_models(actual, scores, current)
        self.assertEqual(base_rate, 0.5)
        self.assertEqual(bins, [1 / 6, 0.5, 0.5, 5 / 6])
        probabilities = sigmoid.predict_proba(scores.reshape(-1, 1))[:, 1]
        self.assertTrue(np.all(np.diff(probabilities) > 0))
        self.assertEqual(probability_metrics(actual, probabilities)["actual_full"], 3)
        self.assertEqual(probability_metrics(actual, actual.astype(float))["brier"], 0.0)
        self.assertIsNone(probability_metrics(actual[:0], probabilities[:0])["brier"])

    def test_logistic_uses_boolean_labels_and_reloads(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "demo.sqlite"
            reference = Path(folder) / "change.joblib"
            classifier = Path(folder) / "logistic.joblib"
            create_demo(database)
            with closing(open_store(database, "simulated")) as db, db:
                db.execute("UPDATE station_status SET available_docks=0 WHERE available_docks=1")
            train(database, 15, reference, allow_simulated=True, allow_missing_lags=True,
                  predict_change=True, boundaries=(START + 7 * 3600, START + 9 * 3600),
                  test_end=START + 12 * 3600)
            report = train_availability(database, joblib.load(reference), classifier, include_time_band=True)
            self.assertIn("time_band", report["features"])
            self.assertGreater(report["training_actual_full"], 0)
            self.assertGreater(report["test_at_cutoff"]["actual_full"], 0)
            artifact = joblib.load(classifier)
            self.assertEqual(artifact["pipeline"].classes_.tolist(), [False, True])
            result = predict_availability(artifact, database, "SIM-A", DEMO_TIME)
            self.assertEqual(result["status"], "ok")
            self.assertIsInstance(result["predicted_has_available_dock"], bool)
            self.assertEqual(result["time_band"], "16:00–20:00")
            self.assertNotIn("predicted_available_docks", result)
            self.assertEqual(predict_availability(artifact, database, "SIM-A", DEMO_TIME, 5)["status"],
                             "unsupported_horizon")

    def test_change_model_outputs_available_docks_after_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "demo.sqlite"
            model_path = Path(folder) / "change.joblib"
            create_demo(database)
            report = train(database, 15, model_path, allow_simulated=True,
                           allow_missing_lags=True, predict_change=True)
            self.assertEqual(report["regression_target"], "dock_change")
            artifact = joblib.load(model_path)
            _, history = load_history(database, "SIM-A", DEMO_TIME)
            rows = history["SIM-A"]
            _, current, features = features_at(
                rows, [row["observed_at"] for row in rows], DEMO_TIME, True, predict_change=True,
            )
            change = artifact["pipeline"].predict([features])[0]
            expected = min(20, max(0, current["available_docks"] + change))
            self.assertAlmostEqual(predict(artifact, database, "SIM-A", DEMO_TIME)[
                "predicted_available_docks"], expected)

    def test_flow_group_matches_training_and_reloaded_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "demo.sqlite"
            model_path = Path(folder) / "flow.joblib"
            create_demo(database)
            report = train(
                database, 15, model_path, allow_simulated=True,
                allow_missing_lags=True, flow_groups={"SIM-A": "0"},
            )
            self.assertIn("flow_group", report["features"])
            self.assertEqual(report["flow_group_test_coverage"]["mapped_stations"], 1)
            artifact = joblib.load(model_path)
            for station, group in (("SIM-A", "0"), ("SIM-B", "unknown")):
                forecast = predict(artifact, database, station, DEMO_TIME)
                self.assertEqual(forecast["status"], "ok")
                self.assertEqual(forecast["flow_group"], group)

    def test_return_threshold_uses_smallest_qualifying_step(self):
        actual = np.array([0, 0, 0, 3, 7])
        predicted = np.array([1.0, 2.0, 3.0, 2.1, 5.0])
        threshold = select_return_threshold(actual, predicted, 0.5)
        self.assertEqual(threshold, 2.5)
        report = return_decisions(actual, predicted, threshold)
        self.assertEqual(report["actual_full_meets_threshold"], 1)
        self.assertEqual(report["actual_open_below_threshold"], 1)
        with self.assertRaisesRegex(ValueError, "actual full stations"):
            select_return_threshold(np.array([2]), np.array([1.0]), 0.5)

    def test_training_and_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "demo.sqlite"
            model_path = Path(folder) / "demo.joblib"
            create_demo(database)
            _, history = load_history(database)
            rows = history["SIM-A"]
            times = [row["observed_at"] for row in rows]
            at = DEMO_TIME + 30
            end = 601
            self.assertEqual(features_at(rows, times, at), features_at(rows[:end], times[:end], at))
            groups, thresholds = time_splits(history, 15)
            for group, threshold in zip(groups, thresholds):
                self.assertTrue(all(max(base + 900, target) < threshold for _, _, base, target in group))
            self.assertEqual(bounded([-3, 22, 3.2], [{"capacity": 20}] * 3).tolist(), [0, 20, 3.2])
            with self.assertRaisesRegex(ValueError, "allow-simulated"):
                train(database, 15, model_path)
            report = train(database, 15, model_path, allow_simulated=True)
            self.assertEqual(report["test"]["persistence"]["all"]["n"],
                             report["test"]["regression"]["all"]["n"])
            artifact = joblib.load(model_path)
            forecast = predict(artifact, database, "SIM-A", DEMO_TIME)
            self.assertEqual(forecast["status"], "ok")
            self.assertEqual(forecast["data_kind"], "simulated")
            self.assertEqual(forecast["target_time"], "2026-07-01T22:15:00+00:00")
            self.assertGreaterEqual(forecast["predicted_available_docks"], 0)
            self.assertLessEqual(forecast["predicted_available_docks"], 20)
            self.assertEqual(forecast, predict(joblib.load(model_path), database, "SIM-A", DEMO_TIME))
            conservative = predict(artifact, database, "SIM-A", DEMO_TIME,
                                   minimum_recommendation_docks=21)
            self.assertFalse(conservative["meets_return_threshold"])
            self.assertEqual(conservative["predicted_available_docks"], forecast["predicted_available_docks"])
            self.assertEqual(predict(artifact, database, "SIM-A", DEMO_TIME, 5)["status"],
                             "unsupported_horizon")
            self.assertEqual(predict(artifact, database, "other", DEMO_TIME)["status"], "unknown_station")
            self.assertEqual(predict(artifact, database, "SIM-A", START)["status"], "missing_history")
            self.assertEqual(predict(artifact, database, "SIM-A", DEMO_TIME + 600)["status"],
                             "returns_unavailable")
            self.assertEqual(predict(artifact, database, "SIM-A", START + 24 * 3600)["status"], "stale_data")


if __name__ == "__main__":
    unittest.main()
