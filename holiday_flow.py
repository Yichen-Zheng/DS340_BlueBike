"""Compare Bluebikes holiday flow and test a multiplicative adjustment."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, timedelta
from pathlib import Path
import io
import json
import math
from zipfile import ZipFile

import numpy as np

YEARS = (2024, 2025)
BINS_PER_DAY = 96
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
# Skip weeks containing Veterans Day when constructing the November reference.
NOVEMBER_REFERENCE_MONDAYS = {2024: (4, 18), 2025: (3, 17)}


def thanksgiving(year):
    first = date(year, 11, 1)
    return first + timedelta(days=(3 - first.weekday()) % 7 + 21)


def days_from(first, count):
    return [first + timedelta(days=offset) for offset in range(count)]


def periods(year):
    holiday = days_from(thanksgiving(year) - timedelta(days=3), 7)
    november = [
        day
        for monday in NOVEMBER_REFERENCE_MONDAYS[year]
        for day in days_from(date(year, 11, monday), 7)
    ]
    start = date(year, 7, 1)
    summer = days_from(start, (date(year, 10, 1) - start).days)
    labor_day = date(year, 9, 1)
    labor_day += timedelta(days=(0 - labor_day.weekday()) % 7)
    summer = [day for day in summer if day not in (date(year, 7, 4), labor_day)]
    return {"thanksgiving": holiday, "november": november, "july_to_september": summer}


def count_csv(stream, calendar, events, departures):
    for row in csv.DictReader(stream):
        for time_key, station_key, direction in (
            ("started_at", "start_station_id", -1),
            ("ended_at", "end_station_id", 1),
        ):
            timestamp = row[time_key]
            day = calendar.get(timestamp[:10])
            if day is None:
                continue
            if direction == -1:
                departures[day] += 1
            station = row[station_key].strip()
            if station:
                slot = int(timestamp[11:13]) * 4 + int(timestamp[14:16]) // 15
                events[day][station, slot, direction] += 1


def daily_counts(folder, year, dates):
    calendar = {day.isoformat(): day for day in dates}
    events = defaultdict(Counter)
    departures = Counter()
    for month in sorted({(day.year, day.month) for day in dates}):
        prefix = f"{month[0]}{month[1]:02}"
        paths = sorted(folder.glob(f"{prefix}-bluebikes-tripdata*"))
        files = [path for path in paths if path.suffix in (".zip", ".csv")]
        if len(files) != 1:
            raise ValueError(f"Expected one official CSV or ZIP for {prefix}, found {len(files)}")
        path = files[0]
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                count_csv(stream, calendar, events, departures)
        else:
            with ZipFile(path) as archive:
                members = [name for name in archive.namelist()
                           if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")]
                if not members:
                    raise ValueError(f"No CSVs in {path}")
                for name in members:
                    with io.TextIOWrapper(archive.open(name), encoding="utf-8-sig", newline="") as stream:
                        count_csv(stream, calendar, events, departures)
    missing = sorted(day for day in dates if not departures[day])
    if missing:
        raise ValueError(f"No departures on expected dates in {year}: {missing}")
    return events, departures


def net_day(events, stations, day):
    net = np.zeros((len(stations), BINS_PER_DAY))
    for (station, slot, direction), count in events[day].items():
        if station in stations:
            net[stations[station], slot] += direction * count
    return net


def year_data(folder, year):
    groups = periods(year)
    events, departures = daily_counts(folder, year, set().union(*groups.values()))
    references = ("november", "july_to_september")
    # Use the same station cohort for both references and retain quiet holiday bins.
    coverage = [
        {station for day in groups[reference] for station, _, _ in events[day]}
        for reference in references
    ]
    stations = {station: i for i, station in enumerate(sorted(set.intersection(*coverage)))}
    if not stations:
        raise ValueError(f"No common station IDs in {year}")
    nets = {day: net_day(events, stations, day) for day in set().union(*groups.values())}
    daily = {}
    arrays = {}
    for holiday in groups["thanksgiving"]:
        index = holiday.weekday()
        weekday = WEEKDAYS[index]
        record = {"date": holiday.isoformat(), "holiday_trip_departures": departures[holiday]}
        averages = {}
        for reference in references:
            comparison = [day for day in groups[reference] if day.weekday() == index]
            record[f"{reference}_mean_trip_departures"] = float(np.mean(
                [departures[day] for day in comparison]
            ))
            averages[reference] = np.mean([nets[day] for day in comparison], axis=0)
            record[f"{reference}_mean_absolute_net_flow"] = float(np.mean([
                np.abs(nets[day]).mean() for day in comparison
            ]))
        record["holiday_mean_absolute_net_flow"] = float(np.abs(nets[holiday]).mean())
        daily[weekday] = record
        arrays[weekday] = (nets[holiday], averages["july_to_september"], averages["november"])
    return {"matched_stations": len(stations), "daily": daily}, arrays


def errors(actual, prediction):
    residual = actual - prediction
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
    }


def cross_year_test(results, weekdays):
    report = {}
    factors = [step / 20 for step in range(31)]
    for source, test in (YEARS, YEARS[::-1]):
        train_actual, train_summer, _ = (
            np.stack(series) for series in zip(*(results[source][1][day] for day in weekdays))
        )
        # MAE can favor zero when most station bins have no net flow.
        factor = min(factors, key=lambda value: (
            errors(train_actual, value * train_summer)["mae"], abs(value - 1)
        ))
        actual, summer, november = (
            np.stack(series) for series in zip(*(results[test][1][day] for day in weekdays))
        )
        thursday_actual, thursday_summer, _ = results[test][1]["Thursday"]
        report[f"{source}_fit_{test}_test"] = {
            "holiday_factor": factor,
            "test_station_bins": int(actual.size),
            "summer_unadjusted": errors(actual, summer),
            "summer_adjusted": errors(actual, factor * summer),
            "november_unadjusted": errors(actual, november),
            "zero_net_flow": errors(actual, 0),
            "thursday_with_fitted_factor": errors(thursday_actual, factor * thursday_summer),
        }
    return report


def compare(folder):
    results = {year: year_data(folder, year) for year in YEARS}
    return {
        "years": {year: result[0] for year, result in results.items()},
        "thursday": cross_year_test(results, ("Thursday",)),
        "whole_week": cross_year_test(results, WEEKDAYS),
    }


def estimate_available_docks(current_docks, expected_net_returns, capacity=None, holiday_factor=1.0):
    """Apply an explicit net-flow scalar to a current, returnable station snapshot."""
    if not math.isfinite(holiday_factor) or holiday_factor < 0:
        raise ValueError("holiday_factor must be finite and nonnegative")
    if not math.isfinite(current_docks) or current_docks < 0:
        raise ValueError("current_docks must be finite and nonnegative")
    if not math.isfinite(expected_net_returns):
        raise ValueError("expected_net_returns must be finite")
    if capacity is not None and (not math.isfinite(capacity) or capacity < current_docks):
        raise ValueError("capacity must be finite and at least current_docks")
    estimate = max(0.0, current_docks - holiday_factor * expected_net_returns)
    return min(estimate, capacity) if capacity is not None else estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analysis = subparsers.add_parser("compare")
    analysis.add_argument("--data-dir", required=True, type=Path)
    estimate = subparsers.add_parser("estimate")
    estimate.add_argument("--current-docks", required=True, type=float)
    estimate.add_argument("--expected-net-returns", required=True, type=float)
    estimate.add_argument("--capacity", type=float)
    estimate.add_argument("--holiday-factor", default=1.0, type=float)
    args = parser.parse_args()
    try:
        if args.command == "compare":
            payload = compare(args.data_dir)
        else:
            payload = {
                "estimated_available_docks": estimate_available_docks(
                    args.current_docks, args.expected_net_returns, args.capacity, args.holiday_factor
                ),
                "holiday_factor": args.holiday_factor,
            }
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
