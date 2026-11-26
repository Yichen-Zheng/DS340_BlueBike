# DS340_BlueBike

[简体中文](README_ZH.md)

## Phase 1 · Forecasting available return docks

Given a station and a reference time, forecast how many docks will be available 15 minutes later. The existing `Bluebike.ipynb` reads trip data. Use the CLI or the Python function `predict(artifact, db, station_id, at, horizon=15)` to connect forecasts to a station list. Display forecasts from real data with `status == "ok"` alongside their `target_time`. Stations that are uninstalled or closed to returns do not produce a usable forecast.

### Data inventory

| Local file | Trips | Departure timestamp range | Arrival timestamp range | Distinct non-empty origin or destination station IDs |
| --- | ---: | --- | --- | ---: |
| 2607.csv | 530,297 | 2026-06-30 04:33 to 07-31 23:56 | 07-01 00:00 to 07-31 23:59 | 606 |
| 2608.csv | 582,663 | 2026-07-31 01:38 to 08-31 23:56 | 08-01 00:00 to 08-31 23:59 | 617 |
| 2609.csv | 555,205 | 2026-08-30 23:38 to 09-30 23:54 | 09-01 00:00 to 09-30 23:59 | 621 |

The table shows the original timestamps to the minute. The CSVs contain no time zone offsets. They include ride ID, bike type, start and end times, station names and IDs, coordinates, and membership type. Each row is a trip event. The files omit station status, disabled docks, and bike rebalancing, so they cannot reconstruct actual dock availability on their own. Training and prediction use GBFS station IDs. Verify the ID mapping if a station list uses trip CSV IDs.

The operator publishes trip history, a station inventory with capacity, and a live GBFS feed. The project did not find a downloadable station status history from the operator. The first live capture covered 595 stations, and at 195 of them the reported available docks differed from capacity minus available bikes. The prediction target is the reported `num_docks_available`, with missing values kept as missing. [Bluebikes System Data](https://bluebikes.com/system-data) · [Live GBFS discovery feed](https://gbfs.bluebikes.com/gbfs/gbfs.json)

### Live collection

The collector uses the Python standard library. Modeling was verified with scikit-learn 1.4.2, NumPy 1.26.4, and joblib in the project's Anaconda environment. The commands below use `python` from that environment, such as `/opt/anaconda3/bin/python` on the project machine.

```bash
python collect_status.py --once
python collect_status.py --db data/status.sqlite --interval 60
```

Continuous collection runs in the foreground and stops with Ctrl+C. Use the system's service manager if collection needs to survive terminal exits or restarts. A restarted collector can resume writing to the same database. The script logs network and data errors and retries on the next cycle.

SQLite stores collection time, feed update time, station report time, station ID, available docks and bikes, installation and return/rental status, capacity, and location. Timestamps are Unix seconds in UTC. The primary key deduplicates records for the same station and second. Each snapshot is written in a single transaction. The default polling interval is 60 seconds, and matching uses observed timestamps. The collector rejects feeds more than five minutes old. Git ignores generated database and model files.

### Training, evaluation, and prediction

Collect continuous real snapshots across weekdays, weekends, and different times of day before training. A single snapshot cannot support training.

```bash
python dock_regression.py train --db data/status.sqlite --model artifacts/real-15.joblib --horizon 15
python dock_regression.py evaluate --model artifacts/real-15.joblib
python dock_regression.py predict --db data/status.sqlite --model artifacts/real-15.joblib \
  --station-id "$GBFS_STATION_ID" --at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --horizon 15
```

Set `GBFS_STATION_ID` to a station covered by training. Supply the reference time as ISO 8601 with an explicit offset. A response includes the feed update time, station report time, target time, continuous forecast, model version, and data source. If a forecast is unavailable, the CLI returns a null prediction and a status such as `unsupported_horizon`, `unknown_station`, `missing_history`, `stale_data`, `missing_dock_count`, or `returns_unavailable`, and exits with code 2. Load only joblib files you produced or trust, using the same dependency versions as training.

- **Matching and features.** The target is the closest real observation at the same station within ±90 seconds of the requested horizon. Ties go to the earlier observation. A gap of more than 180 seconds between consecutive records invalidates the target window. The 5-, 15-, and 30-minute history features look backward only, with up to 120 seconds of lag tolerance and the same gap check. The current and matched observations require both the feed and station report to be no more than five minutes old. Features include Boston-local time of day, day of week, weekend status, current docks, five-minute change, station ID, capacity, and coordinates.
- **Evaluation.** The default chronological split is 60/20/20 across distinct observation timestamps. Rows are dropped if the requested or matched target crosses the end of the split. Validation and testing cover stations seen in training. Encoding, imputation, and scaling are fitted on training data. Validation MAE selects the Ridge penalty. The final test uses the same rows for persistence and regression, clips predictions at zero and at known capacity, and keeps fractional values. The saved report includes sample counts, MAE, and RMSE for the full test set and for observations with 0–2 available docks.
- **Forecast horizon.** Train and evaluate a separate model with `--horizon` for horizons such as 5 or 30 minutes. Arbitrary arrival times require additional validated horizons or a model that takes the horizon as input. Probability forecasting requires separate validation.

### Synthetic smoke test

The database and model below are explicitly labeled `simulated`. Use them to verify the pipeline. Real test results from station archives are described later in this README.

```bash
python -m unittest -v
python -c "from test_dock_regression import create_demo; create_demo('data/simulated.sqlite')"
python dock_regression.py train --db data/simulated.sqlite --model artifacts/simulated-15.joblib --allow-simulated
python dock_regression.py evaluate --model artifacts/simulated-15.joblib
python dock_regression.py predict --db data/simulated.sqlite --model artifacts/simulated-15.joblib \
  --station-id SIM-A --at 2026-07-01T18:00:00-04:00
```

## Thanksgiving trip flow and adjustment factor

This experiment uses the July, August, September, and November 2024–2025 trip files and the December 2024 file. The original files were downloaded to a temporary directory. Thanksgiving week ran from November 25 to December 1 in 2024 and November 24 to 30 in 2025. [Official Bluebikes trip archive](https://s3.amazonaws.com/hubway-data/index.html)

| Year | Thanksgiving-week departures | Same-year November reference | Same-year July–September reference |
| --- | ---: | ---: | ---: |
| 2024 | 49,816 | 101,954 | 126,916 |
| 2025 | 52,134 | 90,830 | 127,734 |

Weekly references sum the averages for each day of the week and are rounded in the table. The November reference weeks begin November 4 and 18 in 2024 and November 3 and 17 in 2025, avoiding Veterans Day weeks. The summer reference excludes July 4 and Labor Day. Comparisons use stations with trip records in both reference periods, giving 505 stations in 2024 and 581 in 2025. Each local day has 96 aligned 15-minute windows. Net returns are arrivals minus departures. Windows without recorded events have zero trip flow. The trip data cannot establish whether those stations were open.

| Thanksgiving Day test | Station windows | Unadjusted summer MAE / RMSE | Ordinary November Thursday MAE / RMSE | Summer scaled by factor fitted on the other year MAE / RMSE |
| --- | ---: | ---: | ---: | ---: |
| 2024 | 48,480 | 0.277 / 0.627 | 0.330 / 0.742 | 0.066 / 0.302 |
| 2025 | 55,776 | 0.285 / 0.606 | 0.350 / 0.761 | 0.124 / 0.446 |

Errors are in bikes per station per 15 minutes. The `holiday_factor` scales the summer mean net returns for the same station, weekday, and time window. The experiment selects a factor from 0 to 1.5 in steps of 0.05 by MAE on one Thanksgiving Day and tests it on the other. Both directions select 0, equivalent to a zero-net-flow baseline. The result applies to the evaluated windows and historical mean baseline. Other dates, arbitrary departure minutes, and real dock availability need separate validation. The CLI defaults to a factor of 1. A factor of 0 keeps the current dock count.

```bash
python holiday_flow.py compare --data-dir /tmp/bluebikes-holiday > artifacts/holiday-flow.json
python holiday_flow.py estimate --current-docks 6 --expected-net-returns 4 --capacity 20 --holiday-factor 0.5
```

The comparison reads the official CSV or ZIP filenames and reports daily and cross-year results. The estimator computes current available docks minus the factor times expected net returns, subject to the bounds. Use a fresh snapshot from a station that accepts returns. For GBFS validation, compare factors 0 and 1 and any other candidates on the same station and forecast window. Measure the errors against official observations 15 minutes later and report the subset with 0–2 actual docks separately.

Fitting on Monday through Sunday in one year and testing on the other year's full week also selects 0 in both directions.

| Full-week test | Station windows | Unadjusted summer MAE / RMSE | Summer scaled by full-week factor MAE / RMSE |
| --- | ---: | ---: | ---: |
| 2024 Thanksgiving week | 339,360 | 0.336 / 0.709 | 0.196 / 0.590 |
| 2025 Thanksgiving week | 390,432 | 0.313 / 0.645 | 0.186 / 0.562 |

The full-week factor gives the same result on Thursday as the Thursday-only factor. The JSON contains `whole_week` and `thursday` results. A pooled weekly result does not establish that every day would select the same factor on its own.

## Ordinary-week validation against real dock availability

The fixed evaluation period is September 14–27, 2026, Boston time. Observed availability comes from the archive's `free` field. The historical net-flow reference uses only `2607.csv` and `2608.csv`, excluding July 4. It averages by station, weekday, and minute of day, then sums the following 15 minutes starting on the first whole minute at or after the observation. Factor 0 uses current availability. Factor 1 subtracts historical net returns and clips to the current capacity. Both methods use identical test samples, with no parameter tuning on the test set.

The official station inventory maps GBFS `station_id` to trip `short_name`. CityBikes `nuid` joins on the GBFS ID. Only unique mappings with historical flow data are retained. The closest target must be within ±90 seconds. Both observations must be from physical stations accepting returns, reported in the last five minutes, with dock counts within capacity. Windows that cross a week boundary or have an observation gap greater than three minutes are excluded. The archive records status changes at irregular intervals, so filtering favors busy stations and times. Neighboring windows can overlap.

Download the archive and station inventory before running with pandas and PyArrow installed. Downloading a later revision of the station inventory can change station coverage. [September 2026 CityBikes archive](https://data.citybik.es/dumps/by-network/2026/202609-blue-bikes-stats.parquet) · [Official Bluebikes station inventory](https://gbfs.lyft.com/gbfs/1.1/bos/en/station_information.json)

```bash
mkdir -p artifacts
python validate_ordinary_weeks.py \
  --archive /tmp/202609-blue-bikes-stats.parquet \
  --station-information /tmp/bluebikes-station-information.json \
  --trips 2607.csv 2608.csv > artifacts/ordinary-weeks.json
```

Bike-share data by CityBikes contributors and available from https://data.citybik.es.

## Regression on real station snapshots · September 2026

The importer uses CityBikes `nuid` as the station ID, `free` as available docks, and `extra.last_updated` as station report time. The archive has no separate feed-update timestamp or installation flag. The importer uses archive capture time as the feed-update timestamp, keeps records explicitly marked as physical stations, and uses `returning=1` for return eligibility. Archive and live GBFS records use separate SQLite files.

```bash
python import_citybikes.py --archive /tmp/202609-blue-bikes-stats.parquet \
  --db data/citybikes-202609.sqlite
python dock_regression.py train --db data/citybikes-202609.sqlite \
  --model artifacts/citybikes-real-15.joblib \
  --validation-start 2026-09-10T00:00:00-04:00 \
  --test-start 2026-09-14T00:00:00-04:00 \
  --test-end 2026-09-28T00:00:00-04:00 > artifacts/citybikes-real-15.json
python dock_regression.py evaluate --model artifacts/citybikes-real-15.joblib
python dock_regression.py predict --db data/citybikes-202609.sqlite \
  --model artifacts/citybikes-real-15.joblib \
  --station-id 2012646293885032702 --at 2026-09-14T21:18:48Z
```

The splits use Boston time. September 1–9 supplies 384 training samples, September 10–13 is for validation, and September 14–27 is for testing. Validation selects Ridge `alpha=100` from three candidates. Requiring usable 5-, 15-, and 30-minute history leaves 755 test samples across 10 stations seen in training. The ordinary-week experiment above used a less restrictive set of features.

| Test period by forecast reference time | Samples | Persistence MAE / RMSE | Ridge MAE / RMSE |
| --- | ---: | ---: | ---: |
| September 14–20 | 428 | 4.079 / 5.104 | 3.632 / 4.847 |
| September 21–27 | 327 | 2.844 / 3.854 | 3.412 / 4.249 |
| Combined | 755 | 3.544 / 4.604 | 3.537 / 4.598 |

On the 61 samples with 0–2 actual docks, persistence MAE is 4.902 and Ridge MAE is 5.309. Using at least one predicted dock as the screening rule, persistence and Ridge passed 16 and 20 of the 20 actually full stations, respectively. Persistence remains the baseline. The analysis does not use an academic-term variable. One month covering a single semester transition cannot establish whether such a feature generalizes.

## Missing lag features and the October holdout

Use `--allow-missing-lags` to train Ridge with missing history. The 5-, 15-, and 30-minute lag features each get a missingness indicator and an observation age in seconds. A lag is missing when it fails tolerance or freshness checks, or is separated from the current observation by a gap longer than three minutes. The imputer is fitted on training data. Current snapshot freshness, return eligibility, and target matching still apply. Existing models keep their original history requirements.

Import both monthly files into the same archive database. In Boston local time, the splits are September 1–October 4 for training, October 5–18 for validation, and October 19–25 for the original holdout. The validation set selects the Ridge penalty. Both methods use the same test samples and retain fractional predictions. [September 2026 CityBikes archive](https://data.citybik.es/dumps/by-network/2026/202609-blue-bikes-stats.parquet) · [October 2026 CityBikes archive](https://data.citybik.es/dumps/by-network/2026/202610-blue-bikes-stats.parquet)

```bash
mkdir -p artifacts
python import_citybikes.py --archive /tmp/202609-blue-bikes-stats.parquet \
  --db data/citybikes-202609-10.sqlite
python import_citybikes.py --archive /tmp/202610-blue-bikes-stats.parquet \
  --db data/citybikes-202609-10.sqlite
python dock_regression.py train --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-optional-lags-15.joblib --allow-missing-lags \
  --validation-start 2026-10-05T00:00:00-04:00 \
  --test-start 2026-10-19T00:00:00-04:00 \
  --test-end 2026-10-26T00:00:00-04:00 > artifacts/citybikes-optional-lags-15.json
python dock_regression.py evaluate --model artifacts/citybikes-optional-lags-15.joblib
```

Training uses 30,994 samples and validation selects `alpha=1`. Both methods use 5,057 test samples from 144 stations. Only 381 samples across 10 stations have all three lag features. The archive's irregular sampling favors station-times with frequent observations in the target window.

| Test metric | Persistence | Ridge with optional lags |
| --- | ---: | ---: |
| Overall MAE / RMSE | 3.728 / 4.886 | 3.635 / 4.649 |
| MAE / RMSE for 0–2 actual docks (498 samples) | 3.813 / 5.351 | 4.633 / 5.609 |
| Actually full but predicted at least 1 dock (127 samples) | 119 | 127 |
| Actually had at least 1 dock but predicted below 1 | 80 | 8 |

The lower overall error did not improve near-full performance. Keep persistence as the comparison baseline. Both methods passed many actually full stations at the one-dock cutoff.

## Minimum predicted docks for screening

Set `--minimum-recommendation-docks` on `predict` to control the minimum forecast needed to pass screening. The dock forecast stays continuous. The response field `meets_return_threshold` is true at or above the threshold, false below it, and null when no valid estimate is available. Uncalibrated models default to 1. A calibrated artifact uses its saved threshold unless the CLI overrides it.

The example selects the smallest threshold in steps of 0.5 that passes no more than 50% of actually full samples on the October 5–18 validation set. This percentage is a configurable experimental target. October 26–30 supplies 4,386 held-out archive samples. The calibrated artifact is saved separately and the numerical regression forecast stays the same.

```bash
python dock_regression.py calibrate-return-threshold \
  --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-optional-lags-15.joblib \
  --output-model artifacts/citybikes-threshold-15.joblib \
  --test-start 2026-10-26T00:00:00-04:00 --test-end 2026-10-31T00:00:00-04:00 \
  --max-full-station-false-open-rate 0.5 > artifacts/citybikes-threshold-15.json
python dock_regression.py predict --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-threshold-15.joblib \
  --station-id 00802fe6-4268-4dc8-8f55-0dbbc1665ce6 --at 2026-10-19T04:04:22Z \
  --minimum-recommendation-docks 9
```

| Holdout method | Actually full but passed screening (157 total) | Had a dock but failed screening (4,229 total) |
| --- | ---: | ---: |
| Ridge, original threshold 1 | 156 | 19 |
| Ridge, calibrated threshold 5.5 | 52 | 783 |
| Persistence, calibrated threshold 4.5 | 59 | 802 |

These results show the trade-off between passing actually full stations and screening out usable ones. The counts describe this historical test set.

## Station groups from historical trip flow

The exploratory script uses July–August 2026 trips, excluding July 3 and 4. It summarizes checkouts and returns in four time bands, separately for weekdays and weekends. K-means uses standardized flow volume and directional patterns to form four groups. The official GBFS inventory is used for the station ID join. A group gets its own Ridge fit when it has at least 500 training samples and five training stations. Stations without enough history or a mapping use the global model.

```bash
python flow_type_experiment.py --trips 2607.csv 2608.csv \
  --station-information /tmp/bluebikes-station-information.json \
  --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-optional-lags-15.joblib \
  --classifier artifacts/citybikes-flow-classifier.joblib \
  > artifacts/citybikes-flow-types-exploratory.json
```

The saved grouping artifact contains the scaler, cluster centers, and trip IDs for grouped stations. On 10,408 shared validation samples from October 5–18, the global and group-specific models obtain overall MAEs of 3.418 and 3.349, respectively. On samples with 0–2 actual docks, their MAEs are 4.645 and 4.867. This is an exploratory comparison. The existing regression and screening threshold stay in use.

### Adding the flow group to a single Ridge model

Train with `--flow-classifier` and `--station-information` to use the saved flow group as a categorical feature in a shared model. Stations with training data but no mapped group receive the `unknown` category. The trained artifact stores the mapping, so prediction does not require the classifier or station inventory files.

```bash
python dock_regression.py train --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-pooled-flow-group-15.joblib --allow-missing-lags \
  --flow-classifier artifacts/citybikes-flow-classifier.joblib \
  --station-information /tmp/bluebikes-station-information.json \
  --validation-start 2026-10-05T00:00:00-04:00 \
  --test-start 2026-10-19T00:00:00-04:00 --test-end 2026-10-26T00:00:00-04:00 \
  > artifacts/citybikes-pooled-flow-group-15.json
```

Both versions are compared on the same, previously used October 19–25 diagnostic samples. Of 5,057 samples, 4,971 map to a group, covering 135 of 144 stations. Both versions have overall MAE 3.635 and near-full MAE 4.633 after rounding. At the one-dock threshold, both pass all 127 actually full samples. The group-feature candidate is saved separately. The calibrated threshold is attached to the earlier model, and the candidate needs a fresh holdout.

## Dock-change regression and equal-coverage comparisons

`--predict-change` sets the target to the difference between future and current docks. At prediction time, add the predicted change to the current count and clip to the feasible range. Features include the most recent observed change, how long ago the new count first appeared, and its interaction with flow group. They use observations through the reference time and look back roughly half an hour. A long gap, stale data, or lack of an observed change gives a missing feature.

```bash
python dock_regression.py train --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-dock-change-15.joblib --allow-missing-lags --predict-change \
  --flow-classifier artifacts/citybikes-flow-classifier.joblib \
  --station-information /tmp/bluebikes-station-information.json \
  --validation-start 2026-10-05T00:00:00-04:00 \
  --test-start 2026-10-19T00:00:00-04:00 --test-end 2026-10-26T00:00:00-04:00 \
  > artifacts/citybikes-dock-change-15.json

python compare_return_coverage.py --db data/citybikes-202609-10.sqlite \
  --direct-model artifacts/citybikes-pooled-flow-group-15.joblib \
  --change-model artifacts/citybikes-dock-change-15.joblib \
  --start 2026-10-19T00:00:00-04:00 --end 2026-10-26T00:00:00-04:00 \
  > artifacts/citybikes-change-coverage.json
```

All methods use the same station-times. Rank by predicted docks and select the same number of observations across the full sample, by default about 80% and 90%. A fixed random ordering breaks ties consistently across methods.

| Previously used October 19–25 archive samples | Persistence | Direct dock-count regression | Dock-change regression |
| --- | ---: | ---: | ---: |
| Overall MAE (5,057 samples) | 3.728 | 3.635 | 3.615 |
| MAE for 0–2 actual docks (498 samples) | 3.813 | 4.633 | 4.542 |
| Actually full among 4,045 selected (about 80% coverage) | 38 | 44 | 44 |
| Actually full among 4,551 selected (about 90% coverage) | 72 | 67 | 70 |

The project's minute-by-minute official GBFS collection was started in a terminal session, with database `data/status.sqlite` and log `data/status-collection.log`. The collector depends on the session and its network connection. To resume, use the continuous collection command above after checking for another collector. Once an ordinary week is available for independent evaluation, use the comparison command above with `data/status.sqlite`, the week's dates, and `--allow-live-gbfs`. Report the common station coverage and evaluation dates when comparing archive-trained models on live-feed data.

## Logistic classification · At least one dock in 15 minutes

The classifier uses the dock-change regression features, station coverage, chronological splits, and observed target matching. The label is `target available docks >= 1`. Return restrictions, missing data, and stale data are handled explicitly. The model uses class balancing and picks regularization by average precision for actual full stations on validation. The runtime cutoff is fitted on validation to retain about 80% of candidates. The API returns the boolean `predicted_has_available_dock`, or null when prediction is unavailable.

```bash
python dock_classifier.py train --db data/citybikes-202609-10.sqlite \
  --reference-regression artifacts/citybikes-dock-change-15.joblib \
  --model artifacts/citybikes-logistic-15.joblib > artifacts/citybikes-logistic-15.json
python dock_classifier.py evaluate --model artifacts/citybikes-logistic-15.joblib
python dock_classifier.py predict --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-logistic-15.joblib \
  --station-id 00802fe6-4268-4dc8-8f55-0dbbc1665ce6 --at 2026-10-19T04:04:22Z --horizon 15
```

Add `--classifier artifacts/citybikes-logistic-15.joblib` to the comparison command to evaluate the classifier and regression on the same samples. For a fresh ordinary week of official GBFS data, use `data/status.sqlite`, the evaluation dates, and `--allow-live-gbfs`. Equal-coverage comparisons rank by classifier score. See the classifier training report for results at the saved cutoff.

| October 19–25 archive diagnostic (5,057 samples) | Persistence | Direct regression | Dock-change regression | Logistic |
| --- | ---: | ---: | ---: | ---: |
| Actually full among 4,045 selected (about 80% coverage) | 38 | 44 | 44 | 51 |
| Actually full among 4,551 selected (about 90% coverage) | 72 | 67 | 70 | 75 |

The archived test week had already been used for diagnostics. Logistic is saved as a separate candidate, and the usual recommendation rule has not switched to it.

### Evaluation by time of day

Add `--by-time-of-day` to the equal-coverage comparison. It reports sample counts across all 24 hours and results for five Boston-local reference-time bands. Each method selects the same number within each band. The overall report ranks across the full sample, so summing the band-level full-station errors can give a different total. A band without actual full samples cannot establish how well a method recognizes full stations.

### Categorical time bands for Logistic

Train with `--time-band` to add a Boston-local categorical feature with boundaries at 00:00, 06:00, 10:00, 16:00, 20:00, and 24:00. Continuous cyclical hour features, weekday, and weekend remain available. Training, the time-of-day report, and prediction use the same band boundaries based on the forecast reference time. Predictions from this model include `time_band`.

```bash
python dock_classifier.py train --db data/citybikes-202609-10.sqlite \
  --reference-regression artifacts/citybikes-dock-change-15.joblib \
  --model artifacts/citybikes-logistic-time-band-15.joblib --time-band \
  > artifacts/citybikes-logistic-time-band-15.json
```

Point `--classifier` at this model and add `--by-time-of-day` to reproduce the breakdown. The model is saved separately. The comparison reuses the October 19–25 archive diagnostic set.

### Evaluating low-dock and sustained-decline rules

`evaluate_return_guards.py` compares rules offline at matched coverage. On the classifier's original validation period, it tests moving candidates with at most 2, 3, or 5 current docks behind other candidates in the Logistic ranking. It also tests subtracting 1, 2, or 3 docks from the current count when the station shows sustained decline. Decline requires at least two observed decreases without a rebound in the previous 5 or 10 minutes. Missing or stale history and long observation gaps make the trend unknown. Each rule family selects the parameters with the fewest full-station errors on validation and fixes them for testing. All four methods recommend the same number of candidates, about 80%.

```bash
python evaluate_return_guards.py --classifier artifacts/citybikes-logistic-time-band-15.joblib \
  --validation-db data/citybikes-202609-10.sqlite \
  --test-db data/citybikes-202609-10.sqlite \
  --start 2026-10-19T00:00:00-04:00 --end 2026-10-26T00:00:00-04:00 \
  > artifacts/citybikes-return-guards-archive.json
```

For an independent ordinary-week test, use `--test-db data/status.sqlite --allow-live-gbfs` and the week's start and end. Keep the archive as the validation database. The script reports errors by starting dock count, and separately reports sustained-decline samples and subsequent full stations between 16:00 and 20:00. Short live runs are used to verify the data and workflow.

October validation selected demoting stations with at most 3 docks and subtracting 3 docks when a 5-minute sustained decline is observed. On the October 19–25 archive diagnostic, the numbers of actually full samples selected were 38 for current docks, 52 for Logistic, 43 for Logistic with low-dock demotion, and 41 for current docks with a decline penalty. In the November 25 Thanksgiving-eve official GBFS trial from 15:35 to 17:35, the corresponding counts were 16, 1,011, 43, and 19. The live trial used 27,445 candidates at 259 common stations, including 1,857 actual full labels. Minute-level forecast windows overlap. An independent ordinary-week test remains pending.

### Offline probability calibration

Fit a sigmoid to the time-band Logistic score on October 26–28, then evaluate on October 29–30. The fitted mapping is also checked against official GBFS snapshots from November 25, 15:35–17:35 Boston time. The offline report compares a constant calibration-period availability rate, a baseline by current dock-count band, and calibrated Logistic. It includes Brier score, log loss, reliability bins, and separate results for the late peak and for 0–2 current docks.

```bash
python calibrate_dock_probability.py calibrate --db data/citybikes-202609-10.sqlite \
  --classifier artifacts/citybikes-logistic-time-band-15.joblib \
  --output-model artifacts/citybikes-dock-probability-calibration.joblib \
  --calibration-start 2026-10-26T00:00:00-04:00 --calibration-end 2026-10-29T00:00:00-04:00 \
  --test-start 2026-10-29T00:00:00-04:00 --test-end 2026-10-31T00:00:00-04:00 \
  > artifacts/citybikes-dock-probability-archive.json
python calibrate_dock_probability.py evaluate --db data/status.sqlite --allow-live-gbfs \
  --model artifacts/citybikes-dock-probability-calibration.joblib \
  --test-start 2026-11-25T15:35:00-05:00 --test-end 2026-11-25T17:35:55-05:00 \
  > artifacts/bluebikes-dock-probability-live-nov25.json
```

Archive Brier scores were 0.0330 for both calibrated Logistic and the dock-count baseline after rounding. In the short live trial, the corresponding scores were 0.0671 and 0.0549. For stations starting with 0–2 docks, calibrated Logistic averaged 95.6% while the live observed fraction was 61.4%. This calibration remains an offline experiment. The prediction interface continues to return the boolean result, and an independent ordinary-week probability test remains pending.

### Spring probability experiment · May and June 2026 tests

This experiment uses March for training and model selection, April 1–7 for the initial classifier check, and April 8–30 to calibrate its scores. May and June are used for probability evaluation. The spring models use the same status-based features and time bands. The flow-group feature is omitted because its saved groups were derived from July–August trips.

Download the [March](https://data.citybik.es/dumps/by-network/2026/202603-blue-bikes-stats.parquet), [April](https://data.citybik.es/dumps/by-network/2026/202604-blue-bikes-stats.parquet), [May](https://data.citybik.es/dumps/by-network/2026/202605-blue-bikes-stats.parquet), and [June](https://data.citybik.es/dumps/by-network/2026/202606-blue-bikes-stats.parquet) CityBikes snapshots to `/tmp` before running:

```bash
mkdir -p artifacts
db=data/citybikes-202603-06.sqlite
for month in 202603 202604 202605 202606; do
  python import_citybikes.py --archive "/tmp/$month-blue-bikes-stats.parquet" --db "$db"
done

python dock_regression.py train --db "$db" --allow-missing-lags --predict-change \
  --model artifacts/citybikes-spring-dock-change-15.joblib \
  --validation-start 2026-03-23T00:00:00-04:00 \
  --test-start 2026-04-01T00:00:00-04:00 --test-end 2026-04-08T00:00:00-04:00

python dock_classifier.py train --db "$db" --time-band \
  --reference-regression artifacts/citybikes-spring-dock-change-15.joblib \
  --model artifacts/citybikes-spring-logistic-time-band-15.joblib

python calibrate_dock_probability.py calibrate --db "$db" \
  --classifier artifacts/citybikes-spring-logistic-time-band-15.joblib \
  --output-model artifacts/citybikes-spring-probability.joblib \
  --calibration-start 2026-04-08T00:00:00-04:00 --calibration-end 2026-05-01T00:00:00-04:00 \
  --test-start 2026-05-01T00:00:00-04:00 --test-end 2026-07-01T00:00:00Z \
  > artifacts/citybikes-may-june-probability.json

for month in 05 06; do
  if [ "$month" = 05 ]; then
    end=2026-06-01T00:00:00-04:00
  else
    end=2026-07-01T00:00:00Z
  fi
  python calibrate_dock_probability.py evaluate --db "$db" \
    --model artifacts/citybikes-spring-probability.joblib \
    --test-start "2026-$month-01T00:00:00-04:00" --test-end "$end" \
    > "artifacts/citybikes-$month-probability.json"
done
```

The June source file ends at July 1 00:00 UTC, so the test excludes the final four hours of June 30 Boston time. The combined report can include windows spanning the May–June boundary. Monthly reports exclude those boundary-crossing windows.

May has 8,868 matched test samples and June has 11,011. Their calibrated Logistic Brier scores are 0.0268 and 0.0365. The corresponding current-dock-bin baseline scores are 0.0261 and 0.0355. Across the two months, Logistic predicts 95.5% availability on average when the station starts with 0–2 docks, compared with 88.0% observed. The current-dock-bin baseline averages 90.6% in that group. These remain offline results.

## Repository files and source downloads

This repository includes the analysis code, notebook, reports, presentations, and fitted model artifacts. Recreate local real-data SQLite databases with the import and collection commands above. Download original trip data from Bluebikes and keep it in your local workspace:

| Local filename | Official download |
| --- | --- |
| `2607.csv` | [July 2026 trips](https://s3.amazonaws.com/hubway-data/202607-bluebikes-tripdata.zip) |
| `2608.csv` | [August 2026 trips](https://s3.amazonaws.com/hubway-data/202608-bluebikes-tripdata.zip) |
| `2609.csv` | [September 2026 trips](https://s3.amazonaws.com/hubway-data/202609-bluebikes-tripdata.csv) |

Extract the CSV from each ZIP and use the local filename shown in the table. The monthly CityBikes snapshot links for the experiments appear above. The distribution terms apply to the raw data. [Bluebikes Data License Agreement](https://bluebikes.com/data-license-agreement) · [CityBikes data terms](https://data.citybik.es/#terms)
