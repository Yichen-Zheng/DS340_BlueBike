"""Collect timestamped Bluebikes GBFS snapshots into SQLite."""
import argparse
from collections import Counter
import json
import logging
import sqlite3
import time
from pathlib import Path
from urllib.request import urlopen

FORECASTING_DIR = Path(__file__).resolve().parent
DEFAULT_TRIP_FILES = tuple(FORECASTING_DIR.parent / name for name in ("2607.csv", "2608.csv"))
DEFAULT_STATUS_DB = FORECASTING_DIR / "data" / "status.sqlite"

DISCOVERY = "https://gbfs.bluebikes.com/gbfs/gbfs.json"
SCHEMA = """
CREATE TABLE IF NOT EXISTS station_status (
    station_id TEXT NOT NULL,
    observed_at INTEGER NOT NULL,
    feed_updated_at INTEGER NOT NULL,
    last_reported INTEGER,
    available_docks INTEGER,
    available_bikes INTEGER,
    is_installed INTEGER,
    is_renting INTEGER,
    is_returning INTEGER,
    capacity INTEGER,
    latitude REAL,
    longitude REAL,
    PRIMARY KEY (station_id, observed_at)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def open_store(path, kind, source=None):
    source = source or ("official_gbfs" if kind == "real" else kind)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)
        with db:
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('data_kind', ?)", (kind,))
            if db.execute("SELECT value FROM metadata WHERE key='data_kind'").fetchone()[0] != kind:
                raise ValueError("Database data_kind does not match")
            # Existing real stores without provenance were created by the live collector.
            recorded = db.execute("SELECT value FROM metadata WHERE key='data_source'").fetchone()
            if recorded is None and db.execute("SELECT 1 FROM station_status LIMIT 1").fetchone():
                recorded = ("official_gbfs" if kind == "real" else kind,)
            if recorded is not None and recorded[0] != source:
                raise ValueError("Database data_source does not match")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('data_source', ?)", (source,))
    except (ValueError, sqlite3.Error):
        db.close()
        raise
    return db


def station_map(path):
    """Map unambiguous official GBFS IDs to trip CSV short names."""
    stations = json.loads(path.read_text())["data"]["stations"]
    pairs = [(str(station["station_id"]), station["short_name"].strip()) for station in stations
             if station.get("short_name")]
    source_counts = Counter(source for source, _ in pairs)
    trip_counts = Counter(trip for _, trip in pairs)
    return {source: trip for source, trip in pairs
            if source_counts[source] == 1 and trip_counts[trip] == 1}


def fetch(url):
    with urlopen(url, timeout=20) as response:
        return json.load(response)


def snapshot(db, discovery=DISCOVERY):
    feeds = fetch(discovery)["data"]
    feeds = feeds["feeds"] if "feeds" in feeds else feeds["en"]["feeds"]
    urls = {item["name"]: item["url"] for item in feeds}
    info = fetch(urls["station_information"])["data"]["stations"]
    metadata = {str(station["station_id"]): station for station in info}
    status = fetch(urls["station_status"])
    observed = int(time.time())
    updated = status["last_updated"]
    if type(updated) is not int or not 0 <= observed - updated <= 300:
        raise ValueError("GBFS feed timestamp is invalid or stale")
    rows = []
    for station in status["data"]["stations"]:
        station_id = str(station["station_id"])
        details = metadata.get(station_id, {})
        rows.append((
            station_id, observed, updated, station.get("last_reported"),
            station.get("num_docks_available"),
            station.get("num_vehicles_available", station.get("num_bikes_available")),
            station.get("is_installed"), station.get("is_renting"), station.get("is_returning"),
            details.get("capacity"), details.get("lat"), details.get("lon"),
        ))
    if not rows:
        raise ValueError("GBFS returned no stations")
    with db:
        db.executemany("INSERT OR IGNORE INTO station_status VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_STATUS_DB, type=Path)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 60:
        parser.error("--interval must be at least 60 seconds")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db = open_store(args.db, "real")
    try:
        while True:
            started = time.monotonic()
            try:
                logging.info("Received %d station records", snapshot(db))
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
                logging.warning("Snapshot failed: %s", error)
                if args.once:
                    return 1
            if args.once:
                return 0
            time.sleep(max(0, args.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
