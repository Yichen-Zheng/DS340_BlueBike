"""Import a CityBikes Bluebikes status archive into a separate regression store."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path

import pyarrow.parquet as pq

from collect_status import open_store

COLUMNS = ("tag", "nuid", "timestamp", "bikes", "free", "extra", "latitude", "longitude")


def as_utc_seconds(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())


def import_archive(archive, database):
    with closing(open_store(database, "real", "citybikes")) as db, db:
        db.execute("INSERT OR IGNORE INTO metadata VALUES (?, ?)",
                   (f"archive_file:{archive.name}", archive.name))
        changes_before = db.total_changes
        for batch in pq.ParquetFile(archive).iter_batches(columns=COLUMNS):
            rows = []
            for item in batch.to_pylist():
                if item["tag"] != "blue-bikes":
                    raise ValueError("Expected Boston Bluebikes records only")
                extra = json.loads(item["extra"])
                if extra.get("virtual") is not False or not item["nuid"]:
                    continue
                observed = as_utc_seconds(item["timestamp"])
                reported = extra.get("last_updated")
                if reported:
                    reported = as_utc_seconds(datetime.fromisoformat(reported))
                capacity = extra.get("slots")
                free = item["free"]
                if free is not None and (free < 0 or (capacity is not None and free > capacity)):
                    free = None
                # Archive observation time is the closest available feed timestamp.
                rows.append((
                    item["nuid"], observed, observed, reported, free, item["bikes"],
                    None, extra.get("renting"), extra.get("returning"), capacity,
                    item["latitude"], item["longitude"],
                ))
            db.executemany("INSERT OR IGNORE INTO station_status VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        rows_loaded = db.total_changes - changes_before
    return rows_loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    args = parser.parse_args()
    print(f"Imported {import_archive(args.archive, args.db):,} new station observations")


if __name__ == "__main__":
    main()
