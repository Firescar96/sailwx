# Sailing Weather Database & Forecast-Accuracy Tracking — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Replace the ad-hoc pixel-scraped CSV with a proper database that (1) ingests raw
observations from MIT Sailing Pavilion + Boston Harbor buoys + Boston airport (KBOS) into a
common schema for months of continuous collection, and (2) ingests forecasts from HRRR, GFS,
and ECMWF on a rolling basis so we can measure how each model's accuracy evolves both across
lead time and across successive forecast runs for the same valid time.

**Architecture:** DuckDB database (`db/weather.duckdb`, embedded/file-based, no server process)
with two logical halves: an `observations` table (tidy/long format: one row per
location+timestamp+variable) fed by lightweight no-LLM cron scripts hitting official
machine-readable feeds, and a `forecast_runs` / `forecast_values` pair that snapshots each
model's forecast *as issued* so later verification can join forecast-at-lead-time against
observed-truth-at-valid-time.

**Tech Stack:** Python 3.11 (via pyenv, project-local `.python-version`) + `duckdb` (pip) +
`requests` (or urllib) + `Pillow` (existing pavilion scraper) + Open-Meteo API (unified access
to GFS/ECMWF/HRRR forecasts + historical archive) + NOAA NDBC realtime2 buoy feeds + NOAA
Aviation Weather Center METAR API. All ingestion scripts are `no_agent` cron jobs (script-only,
no LLM token cost) similar to the existing `extract_wind.py` job.

---

## Why DuckDB (not SQLite or Postgres) for this scale

Back-of-envelope volume over "a few months" (call it 4 months ≈ 120 days):

- Observations: 4 sources × ~6 variables × 1 reading/10min ≈ 4 × 6 × 144/day × 120 days ≈ **414k rows**
- Forecasts: 3 models × ~8 variables × ~130-150 points/run × 4 runs/day (all three models cycle 6-hourly) × 120 days ≈ low millions of rows

This is comfortably inside DuckDB's competence and it's zero-ops just like SQLite: no server
process, no auth, no separate backup story — just a `.duckdb` file we can copy. We picked DuckDB
over SQLite specifically because it's **built for analytical queries** — columnar storage,
vectorized execution — which is exactly the access pattern our two reports need (scan
millions of rows, group by model/variable/lead-time-bucket, compute MAE/RMSE; or scan all
runs covering one valid_time and watch the forecast converge). SQLite would work but is
row-store-oriented and slower at these aggregations; Postgres/Timescale would also work but
require running and maintaining an always-on server process for a personal single-machine
project with low write concurrency — unnecessary ops overhead here.

**Decision locked in 2026-09-05: DuckDB.** Installed via pyenv-managed Python 3.11.1,
project-local `.python-version`, `pip install duckdb`.

---

## Data Sources

| Source | What | Format | Update cadence | Notes |
|---|---|---|---|---|
| MIT Sailing Pavilion | Wind speed/gust (existing) | Pixel-scraped PNG → CSV | Continuous, scraped 2x/day | Keep as-is, migrate output into common schema. Only source without an official machine-readable feed. |
| NOAA NDBC Buoy 44013 | "Boston 16NM East" — wind, gust, dir, pressure, air/water temp, wave height/period | Plain text, fixed-width columns, official | Updated ~every 10 min | `https://www.ndbc.noaa.gov/data/realtime2/44013.txt` — no scraping needed, just parse. Confirmed live format via fetch. |
| NOAA CO-OPS Station 8443970 (Boston) | Wind, air/water temp, water level — actual harbor-adjacent tide station | JSON via `api.tidesandcurrents.noaa.gov/api/prod/datagetter` | 6-min | Closer to the harbor itself than 44013; use as second buoy-class source ("Boston Harbor" per your ask). |
| KBOS (Logan) METAR | Wind, visibility, ceiling, temp, pressure | JSON via `aviationweather.gov/api/data/metar?ids=KBOS&format=json` | Hourly (+ special obs) | Standard aviation feed. |
| Open-Meteo Forecast API | GFS, ECMWF (IFS), + ICON as bonus | JSON, `models=gfs_seamless,ecmwf_ifs04,icon_seamless` | Poll hourly; new *run* appears per model's own cadence (GFS 4x/day, ECMWF 2x/day) | Also has a **Historical Forecast / Single Runs API** for verification-grade archived-as-issued forecasts — worth using to backfill before this project started, not just going forward. |
| HRRR | High-resolution, hourly-updating, short-horizon (18h) US model | Also available through Open-Meteo (`models=hrrr`... check exact model slug at ingestion time) or raw NOMADS GRIB if we need fields Open-Meteo doesn't expose | Hourly | Start with Open-Meteo for simplicity; only drop to raw GRIB parsing if a needed field is missing. |

Decision: **use Open-Meteo as the single ingestion point for all three forecast models** (GFS,
ECMWF, HRRR). It normalizes units/format across models, exposes the model-run/init-time
metadata we need, and has both a live forecast endpoint and a historical-archive endpoint for
verification. This avoids writing three separate GRIB parsers. Fall back to native
NOMADS/ECMWF-open-data only if a required field turns out to be missing from Open-Meteo.

### Confirmed model specs (researched 2026-09-05, via Open-Meteo docs + NOAA/ECMWF sources)

All three models happen to cycle on the **same 6-hour clock (00/06/12/18 UTC)**, which is why a
uniform 6-hourly poll (rather than 3 different per-model schedules) is the right call:

| Model | Runs per day | Cycle times | Max forecast horizon | Native resolution near horizon | Open-Meteo `models=` slug |
|---|---|---|---|---|---|
| **GFS** | 4 | 00/06/12/18 UTC | **16 days (384h)** | hourly to 120h, then 3-hourly to 384h | `gfs_seamless` (or `ncep_gfs_seamless`) |
| **ECMWF IFS HRES** | 4 | 00/06/12/18 UTC | **15 days (360h)** | 1-hourly to 90h, 3-hourly to 144h, 6-hourly to 360h | `ecmwf_ifs` (full 9km res) or `ecmwf_ifs025` (open-data 0.25°, ~2h extra delivery lag) |
| **HRRR** | 24 (hourly) | every hour, but only the **00/06/12/18 UTC runs extend to 48h** — the other 20 runs/day are capped at 18h | **48h at the 4 synoptic hours**, 18h otherwise | hourly throughout | `gfs_hrrr` (confirmed live 2026-09-05 — `hrrr_conus` does NOT work, returns "Cannot initialize MultiDomains" error) |

Practical implication for HRRR: polling every 6 hours at 00/06/12/18 UTC naturally lines up with
*exactly the runs that get the extended 48h horizon* — we don't lose anything by only sampling
HRRR every 6h instead of hourly, because the in-between hourly runs are the short 18h ones we'd
be less interested in for multi-day accuracy tracking. (If later you also want the short-horizon
hourly HRRR nowcast behavior, that would be a separate, denser polling job — flagged as a
possible Phase 6 follow-up, not needed for the accuracy-tracking goal.)

### Ingestion cadence & horizon (per your request)

- **Poll every 6 hours** (`0 */6 * * *`, i.e. ~02:xx/08:xx/14:xx/20:xx local after allowing for
  model processing lag — see delay note below), once per model, requesting the **maximum
  forecast length that model supports**:
  - GFS: `forecast_days=16`
  - ECMWF: `forecast_days=15` (16 rejected — API caps at 15 for this endpoint)
  - HRRR: `forecast_hours=48` (only meaningful at the 4 synoptic cycles; a poll that lands on an
    off-cycle HRRR run will just get an 18h return, which is fine — insert whatever's given)
- **Processing/dissemination lag**: real-world data isn't available the instant the cycle time
  ticks over. ECMWF's own dissemination schedule (confirmed via ecmwf.int) shows the 00Z run's
  atmospheric fields aren't fully out until ~07:34 UTC; GFS/HRRR are faster but not instant.
  Build in a **~2-3 hour delay** between cycle time and poll time so we're not fetching a
  half-published run. Concretely: schedule the cron job's poll times a few hours after each
  00/06/12/18 UTC boundary (e.g. `0 3,9,15,21 * * *` UTC) rather than exactly on the boundary,
  and resolve `init_time_utc` (Task 9) independently of poll time regardless.
- This changes the earlier volume estimate: instead of ~70 lead-time steps/run, GFS/ECMWF runs
  now carry **~130-150 forecast points per run** (120 hourly + ~88 three-hourly for GFS's 384h;
  similar math for ECMWF's tiered resolution) and HRRR carries ~48. At 4 runs/day × 3 models ×
  ~8 variables × ~120 days, this is still comfortably low-millions of rows for SQLite — no
  change to the "SQLite is fine" conclusion above, just a bigger multiplier than the original
  rough estimate.

---

## Schema

```sql
-- One row per physical sensor/model-grid-point location we care about
CREATE TABLE locations (
    location_id   TEXT PRIMARY KEY,   -- e.g. 'mit_pavilion', 'ndbc_44013', 'coops_8443970', 'kbos'
    name          TEXT NOT NULL,
    lat           DOUBLE NOT NULL,
    lon           DOUBLE NOT NULL,
    kind          TEXT NOT NULL       -- 'coastal_station' | 'buoy' | 'airport'
);

-- Raw observations, tidy/long format so heterogeneous sources share one table
CREATE TABLE observations (
    location_id     TEXT NOT NULL REFERENCES locations(location_id),
    ts_utc          TIMESTAMP NOT NULL,  -- UTC timestamp
    variable        TEXT NOT NULL,       -- 'wind_speed_mph','wind_gust_mph','wind_dir_deg',
                                         -- 'pressure_hpa','air_temp_c','water_temp_c', ...
    value           DOUBLE,
    source          TEXT NOT NULL,       -- 'mit_pixel_scrape' | 'ndbc_realtime2' | 'coops_api' | 'metar'
    PRIMARY KEY (location_id, ts_utc, variable)
);
CREATE INDEX idx_obs_var_ts ON observations(variable, ts_utc);

-- One row per (model, init_time, location) forecast *run* we ingested.
-- DuckDB has no AUTOINCREMENT; use a SEQUENCE for the surrogate key.
CREATE SEQUENCE seq_forecast_run_id START 1;
CREATE TABLE forecast_runs (
    run_id        BIGINT PRIMARY KEY DEFAULT nextval('seq_forecast_run_id'),
    model         TEXT NOT NULL,      -- 'gfs' | 'ecmwf' | 'hrrr'
    location_id   TEXT NOT NULL REFERENCES locations(location_id),
    init_time_utc TIMESTAMP NOT NULL, -- model cycle time, e.g. 2026-09-05 12:00:00
    retrieved_at  TIMESTAMP NOT NULL, -- when WE pulled it (may lag init_time by hours)
    UNIQUE (model, location_id, init_time_utc)
);

-- The actual forecast values for a run, one row per valid_time x variable
CREATE TABLE forecast_values (
    run_id         BIGINT NOT NULL REFERENCES forecast_runs(run_id),
    valid_time_utc TIMESTAMP NOT NULL,  -- the time this forecast point is FOR
    variable       TEXT NOT NULL,
    value          DOUBLE,
    PRIMARY KEY (run_id, valid_time_utc, variable)
);
CREATE INDEX idx_fv_valid ON forecast_values(valid_time_utc, variable);
```

`lead_hours = (valid_time_utc - init_time_utc)` is computed at query time, not stored — keeps
the schema normalized and avoids drift if we ever need to recompute. Storing real `TIMESTAMP`
columns (rather than ISO text, as a plain-SQLite version of this schema would need) means this
subtraction is native interval arithmetic in DuckDB, no string-to-julianday conversion needed.

### Verification query pattern (accuracy over lead time)

```sql
SELECT
    fr.model,
    fv.variable,
    date_diff('hour', fr.init_time_utc, fv.valid_time_utc) AS lead_hours,
    fv.value AS forecast_value,
    o.value  AS observed_value,
    fv.value - o.value AS error
FROM forecast_values fv
JOIN forecast_runs fr ON fr.run_id = fv.run_id
JOIN observations o
    ON o.location_id = fr.location_id
   AND o.variable    = fv.variable
   AND o.ts_utc       = fv.valid_time_utc   -- exact match; see note below on nearest-time join
WHERE fr.location_id = 'mit_pavilion' AND fv.variable = 'wind_speed_mph';
```

Because observation timestamps (5/10-min buckets) won't always land exactly on forecast valid
times (hourly), the real implementation will do a **nearest-within-tolerance join** (e.g. ±7.5
min) rather than exact equality — flagged as a task below.

This lets us answer the two things you actually care about:
1. **"How good is model X at N hours out?"** — group by `lead_hours`, compute MAE/RMSE per model/variable, plot error vs. lead time (classic degrading-accuracy curve).
2. **"How does the forecast for a specific upcoming race day change as we get closer to it?"** — fix `valid_time_utc`, look across multiple `init_time_utc` runs, watch the forecast converge (or not) as `retrieved_at` approaches `valid_time`. This is the "run-to-run consistency" view sailors actually want the night before racing.

---

## Step-by-Step Tasks

### Phase 1 — Foundation

**Task 1: Create project structure**
- Create: `db/` (db file, `common.py` helper module) under the project root
- Create: `db/schema.sql` with the DDL above (DuckDB dialect)
- Create: `db/init_db.py` — runs schema.sql against `db/weather.duckdb` if not present (idempotent, `CREATE TABLE IF NOT EXISTS`)
- Verify: `python3 db/init_db.py` then check tables exist via `duckdb.connect(...).sql("SHOW TABLES")`

**Task 2: Seed `locations` table**
- Add rows for `mit_pavilion`, `ndbc_44013`, `coops_8443970`, `kbos` with real lat/lon
- Verify: `SELECT * FROM locations;` returns 4 rows

### Phase 2 — Observation ingestion (migrate + add sources)

**Task 3: Migrate existing MIT pavilion scraper**
- Modify: `~/mit_weather_archive/extract_wind.py` (or write a thin wrapper) to write into
  `observations` (location_id='mit_pavilion', variable in {'wind_speed_mph','wind_gust_mph'})
  instead of / in addition to the standalone CSV
- Keep the CSV write too during a transition window so nothing is lost if the DB write has bugs
- Verify: after a manual run, `SELECT count(*) FROM observations WHERE location_id='mit_pavilion';` grows
- **DONE 2026-09-05**: `scripts/extract_wind.py` now calls `write_rows_to_db()` alongside
  `append_rows()` (CSV). Verified: 355 rows in `observations` for `mit_pavilion` after a live run.
- **CORRECTED 2026-09-09 — real bug found via user's on-the-water report.** User was sailing
  the afternoon of 2026-09-08 and said "the forecasts were totally wrong" after seeing a
  logged 26.1kt gust in the dashboard that didn't match their experience. Investigation (see
  session history) found:
  1. **26.1 kt converts back to exactly 30.0 mph** — the pixel-scraper's hardcoded Y-axis
     ceiling (`Y_MAX_MPH = 30.0`). Confirmed root cause by inspecting the raw day-graph pixels
     directly: the gust line's topmost color-matched pixel sits flat at row 24 (the plot's
     hard ceiling) across 3+ consecutive columns around 2026-09-08 16:20-16:25 UTC — a real
     peak-and-decay curve doesn't flatline like that; this is graph clipping. When true wind
     exceeds the graph's 30 mph axis max, the plotted line visually caps at the top and the
     extracted number silently reads back as "exactly 30 mph" even though the real gust could
     have been higher. **The recorded value is a floor, not the true peak — this was NOT a
     forecast-accuracy problem, it was an observation-data problem** (the forecasts weren't
     necessarily as wrong as the flawed 26.1kt "ground truth" made them look).
  2. **Cross-check confirmed the fault was in the mit_pavilion reading, not real weather**:
     NDBC 44013 at the exact same timestamps showed only 4-8kt gusts — two stations don't
     genuinely disagree by 18+ knots on real wind.
  3. Separately, the resampling was **throwing away real resolution**: the day graph's plot
     area is 634px wide covering 24h = ~2.27 min/pixel native resolution, but the old script
     bucketed everything down to coarse 5-minute averages, discarding detail the source image
     actually had.
  - **Fixes applied**: rewrote `scripts/extract_wind.py` — (a) day-graph-only per explicit user
    instruction ("only ever use the day graph for readings unless I specifically ask you to
    check something else" — the week graph is also a rolling window that shifts on every
    fetch, making it unreliable for reconstructing historical timestamps after the fact,
    which is exactly how the original investigation into this bug briefly went sideways);
    (b) resamples to native ~2.27-min resolution instead of 5-min buckets, so no detail is
    discarded; (c) added a `quality` column to `observations` (schema.sql + live `ALTER
    TABLE` migration for the already-existing table) — any reading whose topmost matched
    pixel lands within 1 row of the plot ceiling gets `quality='clipped_high'`, flagging it
    as a **known-underestimated floor value**, not exact truth, so downstream analysis
    (forecast comparisons, flag correlation) can filter/handle these explicitly instead of
    silently trusting them.
  - **Data handling**: cleared and re-ingested only the last 24h of `mit_pavilion` wind
    data (per user: "delete only from the past day") — older history keeps its original
    5-min-bucket resolution since the source images that produced it are gone and can't be
    reprocessed at finer resolution retroactively.
  - Verified live: re-ingestion correctly re-flagged the original 26.1kt readings from
    2026-09-08 16:24-16:29 UTC as `clipped_high`, AND caught a second previously-unflagged
    clipping episode at 20:40-20:45 UTC the same day. Fresh native-resolution rows (e.g.
    12:06, 12:09, 12:11 UTC) confirm the ~2-3 min spacing is now real, not a rigid 5-min grid.
    Cron job `f6891b01521e` re-run live and confirmed working end-to-end with the new script.

**Task 4: NDBC buoy 44013 ingester**
- Create: `scripts/ingest_ndbc.py`
- Fetch `https://www.ndbc.noaa.gov/data/realtime2/44013.txt`, parse fixed-width/whitespace columns (header rows start with `#`), map `WDIR,WSPD,GST,PRES,ATMP,WTMP` → variables, convert m/s → mph, skip `MM` (missing) values
- Only insert timestamps not already present (`INSERT OR IGNORE`)
- Verify: run once, confirm rows appear for `location_id='ndbc_44013'`
- **DONE 2026-09-05**: verified live, 39,486 rows ingested (feed covers back to
  2026-07-22). **Important performance finding** — see "DuckDB bulk-insert gotcha" below;
  this ingester needed a non-obvious fix to avoid a multi-minute hang.

**Task 5: CO-OPS Boston Harbor station ingester**
- Create: `scripts/ingest_coops.py`
- Call `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter` with `station=8443970`, `product=wind` (and `product=air_temperature`, `water_temperature` as separate calls — CO-OPS is single-product-per-call), `date_range` covering since-last-ingested
- Verify: rows appear for `location_id='coops_8443970'`
- **DONE 2026-09-05, with a correction**: confirmed live that station 8443970 does NOT
  have a `wind` or `water_temperature` sensor (both return "No data was found") — it's
  primarily a tide gauge. Implemented with the two products that DO work:
  `air_temperature` and `water_level`. If a wind sensor is ever added at this station,
  extend `PRODUCTS` in the script.

**Task 6: KBOS METAR ingester**
- Create: `scripts/ingest_metar.py`
- Call `https://aviationweather.gov/api/data/metar?ids=KBOS&format=json&hours=3`, parse `wdir`,`wspd`,`wgst`,`altim`,`temp`
- Verify: rows appear for `location_id='kbos'`
- **DONE 2026-09-05**: verified live, 12 inserts across 3 obs on first run.

**Task 7: Schedule all four as cron jobs**
- Use the `cronjob` tool, `no_agent=True`, one job per script, schedule matched to source cadence:
  - MIT pavilion: keep existing `0 11,23 * * *` (or tighten now that we know it works)
  - NDBC: every 15-20 min (`*/15 * * * *`) — feed updates every ~10 min, no need to hammer it
  - CO-OPS: every 15-20 min
  - METAR: hourly, offset a few minutes after the hour (`5 * * * *`) to let the ob post
- Verify: `cronjob(action='list')` shows all 4, `last_status: ok` after first fire
- **DONE 2026-09-05**: all 4 jobs created —
  `f6891b01521e` (MIT pavilion, `0 11,23 * * *`), `f599c21739d2` (NDBC, `*/15 * * * *`),
  `ebe54790605e` (CO-OPS, `*/15 * * * *`), `340186ccc706` (METAR, `5 * * * *`). All `no_agent`.

### DuckDB bulk-insert gotcha (found while building Task 4)

The NDBC ingester needed to insert ~39k rows per run. Two approaches that seemed reasonable
both hung for minutes / timed out:
1. `con.executemany()` with an `ON CONFLICT DO NOTHING` insert directly into the
   PK/FK-constrained `observations` table — measured ~3-4ms/row, i.e. minutes for this feed.
2. A single parameterized multi-row `INSERT ... VALUES (?,?,?,?), (?,?,?,?), ...` statement
   (tens of thousands of `?` placeholders bound via the Python list-of-params interface) —
   this ALSO hung well past a minute, contrary to the initial assumption that "just batch it
   into one statement" would fix approach 1.

What actually worked (~0.9s for 38,827 rows): a single INSERT statement with **literal values
inlined directly into the VALUES clause** (no `?` placeholders at all), built by
`common.bulk_insert()`. Safe here because bulk_insert is only fed parsed government
data-feed values (numbers/timestamps/short strings), not arbitrary user input — string values
are still escaped (doubled single quotes) before inlining.

Practical pattern used in `ingest_ndbc.py`: bulk-load the whole feed into an *unconstrained*
temp staging table via `bulk_insert()`, then do one set-based `INSERT ... SELECT ... ANTI JOIN
observations` to filter out already-seen rows and land only new ones into the real
constrained table — avoids paying a per-row PK/FK check at insert time entirely.

CO-OPS and METAR ingest a handful of rows per poll (single digits to low tens), so they keep
simple per-row `INSERT ... ON CONFLICT DO NOTHING` calls — this gotcha only bites at NDBC's
~39k-row-per-poll scale (or the future forecast ingester's higher row counts — flag this for
Task 8's implementation too).

### Phase 3 — Forecast ingestion

**Task 8: Forecast fetch/store script**
- Create: `~/weather_db/ingest_forecasts.py`
- For `mit_pavilion`'s lat/lon (the single point of interest — sailors care about the pavilion
  site's forecast, not the buoy's), make **three separate Open-Meteo calls, one per model**
  (never `models=auto` — we need to know exactly which model produced which numbers):
  - GFS: `models=gfs_seamless&forecast_days=16` (confirmed live: 384h/16-day range returned)
  - ECMWF: `models=ecmwf_ifs&forecast_days=15` (confirmed live: 360h/15-day range returned)
  - HRRR: `models=gfs_hrrr&forecast_hours=48` (confirmed live 2026-09-05; NOTE the slug is
    `gfs_hrrr`, NOT `hrrr_conus` as the docs page's model list implies — that slug errors out)
  - Shared hourly variables: `wind_speed_10m,wind_gusts_10m,wind_direction_10m,pressure_msl,temperature_2m`
- Insert one `forecast_runs` row + N `forecast_values` rows per model per fetch
- Verify: run once, `SELECT model, count(*) FROM forecast_values fv JOIN forecast_runs fr USING(run_id) GROUP BY model;` shows 3 models with rows, and GFS/ECMWF row counts are noticeably larger than HRRR's (longer horizon)
- **DONE 2026-09-05**: `scripts/ingest_forecasts.py`. Verified live: gfs 1883 rows, ecmwf 1745
  rows, hrrr 215 rows for a single poll — matches expectation (GFS/ECMWF >> HRRR).
- **CORRECTED 2026-09-07** — originally pulled forecasts for `mit_pavilion` only and compared
  them against observations from 4 physically different locations (buoy 16nm offshore, tide
  station, airport miles away) — not a like-to-like comparison. Fixed: `ingest_forecasts.py`
  now loops over **all 4 locations** (`mit_pavilion`, `ndbc_44013`, `coops_8443970`, `kbos`),
  making one forecast API call per (location × model) = 12 calls/poll instead of 3. The
  metadata API call (init-time lookup) stays per-model only, since a model's cycle time is
  the same everywhere — no need to repeat it per location. Added `idx_fr_location_model` and
  `idx_fv_run` indexes to `schema.sql` to keep per-location queries fast at the higher row
  count. Verified live: all 4 locations × 3 models now have forecast_values rows
  (`ndbc_44013`/`coops_8443970`/`kbos` each: gfs 1920, ecmwf 1800, hrrr 205 — matches
  `mit_pavilion`'s per-poll counts exactly, confirming the pull is now symmetric across
  locations). This is what makes Task 13's accuracy-by-lead-time numbers actually meaningful
  per location, rather than comparing one shared forecast against 4 different truths.

**Task 9: Resolve model init/cycle time**
- All three models cycle on 00/06/12/18 UTC (confirmed via Open-Meteo + NOAA/ECMWF docs — see
  "Confirmed model specs" table above), so `init_time_utc` can be approximated by rounding
  `retrieved_at` **down** to the most recent 6-hour boundary, adjusted per model's known
  dissemination lag:
  - GFS: ~4-5h lag (full 384h product isn't out instantly)
  - ECMWF IFS HRES: confirmed via ecmwf.int dissemination schedule — 00Z run's atmospheric
    fields land ~05:45-07:34 UTC, i.e. ~6-7.5h lag depending on which forecast step tranche
  - HRRR: ~1-2h lag
- Prefer switching to Open-Meteo's **Historical Forecast / Single Runs API** instead of this
  approximation wherever practical — it's purpose-built for verification and returns the actual
  init time, removing the guesswork entirely. Use the approximation above only as a fallback if
  the Single Runs API doesn't support live/near-real-time polling the way we need.
- This is a spike/research task — resolve it **before** finalizing Task 8, not after, since
  every downstream accuracy number depends on init_time_utc being right
- Verify: cross-check one resolved init_time against Open-Meteo's own model-run metadata or a
  known public run time (e.g. ECMWF's published dissemination schedule above)
- **DONE 2026-09-05 — solved exactly, no approximation needed.** Open-Meteo has a dedicated
  **metadata API** for exactly this: `https://api.open-meteo.com/data/<meta_id>/static/meta.json`
  returns `last_run_initialisation_time` as a Unix timestamp — the true init time of the run
  currently being served. Confirmed live. One catch: **the metadata API's internal model id is
  NOT the same string as the forecast API's `models=` slug** —
  - `gfs_seamless` (forecast) → `ncep_gfs013` (metadata)
  - `ecmwf_ifs` (forecast) → `ecmwf_ifs` (metadata, matches)
  - `gfs_hrrr` (forecast) → `ncep_hrrr_conus` (metadata)
  `ingest_forecasts.py` calls the metadata endpoint once per model per poll (uncounted against
  API rate limits per Open-Meteo's docs) and uses the exact returned init time — no dissemination-
  lag guesswork was needed after all.

**Task 10: Dedup logic for repeated forecast polls**
- We poll every 6h and each model issues a new run every 6h too, so in the steady state each
  poll should see exactly one new run per model (not "mostly duplicates" as in hourly polling)
- `forecast_runs` has a `UNIQUE(model, location_id, init_time_utc)` constraint — use `INSERT OR IGNORE` as a safety net in case a poll is retried or fires slightly early/late relative to the cycle boundary
- Verify: run the ingester twice in a row without a new cycle having elapsed, confirm `forecast_runs` row count doesn't grow on the second run
- **DONE 2026-09-05**: verified — running the ingester twice in the same cycle produces
  "already ingested, skipping values" for all 3 models and 0 new `forecast_values` rows.

**Task 11: Schedule forecast ingestion cron job**
- One `no_agent` cron job, every 6 hours offset for dissemination lag — `0 3,9,15,21 * * *` UTC
  (i.e., poll ~3h after each 00/06/12/18 UTC cycle boundary) — calling `ingest_forecasts.py`
- Verify via `cronjob(action='list')`
- **DONE 2026-09-05**: job `7d8d1900e5c9`, schedule `0 3,9,15,21 * * *`, script `sw_ingest_forecasts.sh`.
- **CORRECTED 2026-09-09 — real gap found via user's "the weather didn't match predictions"
  report for 2026-09-08 afternoon.** Investigating that afternoon's forecast-vs-observed gap
  turned up a second, unrelated bug: **ECMWF had zero forecast_values for that entire
  afternoon** (19:00-23:00 UTC 2026-09-08). Root cause: ECMWF disseminates slower than
  GFS/HRRR (~6-7.5h lag vs ~4-5h, per the Task 9 research). If a poll lands before that lag
  has elapsed, the newest ECMWF run isn't ready yet; if the *next* poll happens after an
  even-newer cycle has already superseded it (since the ingester only ever fetches "whatever
  is currently newest," not a specific historical cycle), the in-between run is silently
  skipped forever with no error — it's a genuine hole in coverage, not visible unless you go
  looking for a specific missing window like this one. **Fix**: tightened the forecast
  ingester's poll cadence from every 6h to **every 4h** (`0 */4 * * *`) — since all 3 models
  cycle every 6h, polling more often than that cycle length gives multiple chances to catch a
  slow-disseminating run before a newer one supersedes it, directly narrowing the window in
  which this kind of silent gap can occur. Verified live: re-run via the real cron mechanism
  on the new schedule, correctly deduped already-current GFS/ECMWF runs and picked up a new
  HRRR run (940 new forecast_values rows) — confirms 4-hourly polling works cleanly end to end.
  (Separately, clarified 2026-09-09: `observations.quality='clipped_high'` readings — see
  Task 3's correction — are still legitimate data, not excluded from analysis by default; the
  flag just documents that the graph's line hit its axis ceiling, not that the reading itself
  is invalid.)

### Cron scheduling gotcha found while wiring up Tasks 7 & 11

The `cronjob` tool's `script` field for `no_agent` jobs is **not a shell command line** —
it's resolved as a bare filename relative to `~/.hermes/scripts/`, so `python3 /path/to/foo.py`
or `env VAR=x python3 foo.py` both fail with "Script not found" (the whole string gets treated
as one filename). Fix: create a small executable wrapper shell script per job under
`~/.hermes/scripts/` (e.g. `sw_ingest_ndbc.sh`) that `exec`s the pyenv 3.11.1 interpreter
(which has `duckdb`/`Pillow` installed — system `python3` does not) against the real project
script, `chmod +x` it, and reference just the wrapper's filename in the cron job's `script`
field. All 5 sailing-weather cron jobs use this pattern now (`sw_extract_wind.sh`,
`sw_ingest_ndbc.sh`, `sw_ingest_coops.sh`, `sw_ingest_metar.sh`, `sw_ingest_forecasts.sh`).

### DuckDB concurrent-writer gotcha found while testing the cron jobs

Firing multiple ingesters back-to-back (as a burst of manual test runs, but this WILL also
happen for real whenever two cron schedules land in the same minute) produced real failures:
`IOException: Could not set lock on file ... Conflicting lock is held by ...`. DuckDB's
single-writer model means a second connection attempt while another is open fails immediately
— it does not queue/block and wait. Fixed in `common.get_connection()` with a small
retry-with-delay loop (5 attempts, 2s apart by default) since our ingesters are short-lived
(open, write, close within seconds), so a brief retry is enough — no real lock broker needed.
Also staggered the CO-OPS cron schedule (`7,22,37,52 * * * *`) off of NDBC's (`*/15 * * * *`)
so they don't routinely collide on the same minute even before hitting the retry path.

### Phase 4 — Verification / accuracy tracking

**Canonical units decision (found needed while building this phase):** observations and
forecasts must share units for any comparison query to be meaningful. Standardized on
**knots** for all wind speed/gust values (the sailing standard, per user preference) across
every source — MIT pavilion (pixel-scraped mph → converted), NDBC (native m/s → converted),
METAR (native knots, kept as-is), and the Open-Meteo forecast ingester (native km/h →
converted). Variable names are `wind_speed_kt` / `wind_gust_kt` everywhere now (previously
`_mph` in observations vs. `_kmh` in forecasts — a real mismatch caught before it corrupted
any comparison). All historical wind rows were cleared and re-ingested fresh in knots on
2026-09-06.

**Task 12: Nearest-time join helper**
- Add: `~/weather_db/verify.py` with a function that, given a variable + location, joins each
  `forecast_values.valid_time_utc` to the *nearest* `observations.ts_utc` within a configurable
  tolerance window (default 7.5 min), rather than requiring exact timestamp equality
- Verify: unit-test against a handful of hand-picked rows with known expected pairing
- **DONE 2026-09-06**: `db/verify.py`. Uses a window-function nearest-match (rank candidates
  within the tolerance window, keep rank 1) and exposes both a raw query function
  (`matched_forecast_observation_view`) and a materialized DuckDB VIEW helper
  (`create_matched_view` → `v_forecast_vs_observed`) for ad hoc querying. Verified live:
  correctly matched 34 pairs, including exact-timestamp matches (time_diff_seconds=0) and
  near matches within the window (e.g. 300s off).

**Task 13: Accuracy-by-lead-time report**
- Add: query/report producing MAE and RMSE per `(model, variable, lead_hours bucket)` — bucket
  lead hours into e.g. 0-6h, 6-12h, 12-24h, 24-48h, 48-72h+ bins
- Output as a simple text/CSV report (or a chart if you want visuals later — flag as follow-up)
- Verify: run against a few weeks of real data once ingestion has been running, sanity-check numbers look reasonable (error should generally increase with lead time)
- **DONE 2026-09-06**: `scripts/report_accuracy_by_lead_time.py` (live query/print version).
  Verified live: produces MAE/RMSE rows per model/variable/bucket. Sample sizes were tiny at
  first (only a few hours of history) — expected, improves as ingestion accumulates history
  over the 10-year retention window.
- **CORRECTED 2026-09-08** — originally also had a `run_accuracy_report.py` cron job that
  wrote a fresh dated CSV file to `reports/` every day (and, briefly, an
  `accuracy_snapshots` DB table meant to "snapshot" the computation over time). Both were
  the wrong call: the MAE/RMSE numbers are **fully recomputable at any moment** from
  `forecast_values`/`observations` (the permanent 10-year-retained raw data) — there's
  nothing to snapshot, and generating files on a schedule just piles up stale duplicates of
  a number one query already gives you. Fixed properly: the whole computation is now a
  **live DuckDB view**, `v_accuracy_by_lead_time` (defined in `db/schema.sql`), queryable
  directly with plain SQL (`SELECT * FROM v_accuracy_by_lead_time;`) at any time, by any
  tool, with zero staleness. `report_accuracy_by_lead_time.py` is now a thin pretty-print
  wrapper around the view; `export_report_csv.py` is a genuinely **on-demand** CSV export
  (run manually when you want a file — never scheduled). Removed: the daily cron job
  (`f52b430317a1`), `run_accuracy_report.py`, the `sw_run_accuracy_report.sh` wrapper, the
  `accuracy_snapshots` table, and the `reports/` directory (all 5 previously-generated CSVs
  deleted).

**Task 14: Forecast-convergence report ("how did tonight's forecast for Saturday change")**
- Add: query that fixes a `valid_time_utc` and lists every `forecast_runs` row (across
  `init_time_utc`) whose `forecast_values` cover that valid time, ordered by `init_time_utc`,
  showing how the predicted wind speed/gust changed run-over-run as the valid time approached
- Verify: pick a recent day, confirm the output tells a sensible story (e.g. gust prediction
  tightening from a wide range 5 days out to a narrow range 12 hours out)
- **DONE 2026-09-06**: `scripts/report_forecast_convergence.py` — takes a valid_time + variable
  (CLI args, or defaults to ~24h out / wind_speed_kt for a standalone smoke test) and lists each
  model's prediction for that moment across all ingested runs. Verified live: currently shows
  one row per model (only one forecast poll ingested so far — GFS/ECMWF at 33h lead, HRRR at
  25h lead, for the same target hour). Will show the actual "narrowing" story once multiple
  6-hourly polls accumulate for the same valid_time. Not yet scheduled as its own cron job
  (it's parameterized per query, not a fixed daily report) — run on demand, or ask me to wire
  up a scheduled version for a specific recurring valid_time (e.g. "every upcoming Saturday
  race start") if useful.

### Phase 5 — Backfill & housekeeping

**Task 15: Backfill historical forecasts via Open-Meteo's archive**
- Since Open-Meteo's Historical Forecast / Single Runs API can return forecasts *as they were
  issued* in the past, use it to backfill several weeks/months of forecast history retroactively
  rather than waiting months for enough live-polled data to accumulate
- Verify: backfilled `forecast_runs` rows exist with `init_time_utc` well before project start
- **SKIPPED per user 2026-09-06** — not doing a backfill; data accumulates from live ingestion
  going forward only.

**Task 16: Retention / backup**
- Decide a retention policy (keep everything, given the modest row-count estimate above) and
  add a simple weekly `sqlite3 weather.db .dump | gzip > backup-$(date +%F).sql.gz` cron job
- Verify: backup file appears after first scheduled run
- **UPDATED per user 2026-09-06**: retention policy is explicitly **keep all data for 10
  years** (not just "keep everything" open-endedly) — no pruning job for the *data itself*.
- **DONE 2026-09-06**: `scripts/backup_db.py` — DuckDB-appropriate approach (no SQLite-style
  `.dump`): `CHECKPOINT`s the live DB (flushes WAL into the main file for a consistent
  snapshot), copies + gzips the `.duckdb` file to `backups/weather_<date>.duckdb.gz`, prunes
  backup *files* older than the last 12 (a separate, smaller retention window than the
  10-year *data* retention — unbounded weekly full-file backups would otherwise grow disk
  usage indefinitely; only the underlying data must survive 10 years, not every historical
  backup snapshot). Verified live: backup written (2.04 MB compressed), and a full
  restore-from-backup was tested (`gunzip` + reopen with duckdb) confirming all tables/rows/
  views survive intact. Scheduled weekly via cron job `db5e4bc394a9`
  (`sw_backup_db.sh`, `0 5 * * 0` — Sunday 5am).

---

## Risks / Open Questions

1. **HRRR availability via Open-Meteo** — need to confirm the exact model slug and field
   coverage before Task 8; if missing, fall back to raw NOMADS GRIB (heavier — grib2 parsing via
   `pygrib`/`cfgrib`, no small task, would deserve its own sub-plan).
2. **Init-time resolution (Task 9)** is the crux of the whole forecast-accuracy feature — if we
   get this wrong, every "accuracy by lead time" number is silently mislabeled. Do this as a
   research spike before writing Task 8's final version, not after.
2b. **Rate limits** — Open-Meteo's free tier is generous but not unlimited; hourly polling of
   3 models × 1 location should be well within free-tier limits, but confirm before scaling to
   multiple locations.
3. **MIT pavilion source stays the noisiest** — pixel-extraction is ~1 mph resolution per the
   script's own docstring; when comparing "ground truth" across sources, treat pavilion readings
   as lower-confidence than NDBC/CO-OPS/METAR (all official instrumented feeds).
4. **Timezone handling** — station in EDT/EST awareness already noted in the existing script;
   standardize *everything* in the new DB as UTC (`ts_utc`, `valid_time_utc`, `init_time_utc`)
   and only convert to local time at display/report time, to avoid DST-transition bugs.
5. **CO-OPS is one-product-per-call** — Task 5 needs multiple API calls per poll (wind, air temp,
   water temp separately), so build in basic retry/backoff.

---

## Files Likely to Change / Create

- `~/weather_db/schema.sql`, `init_db.py`, `common.py` (new)
- `~/weather_db/ingest_ndbc.py`, `ingest_coops.py`, `ingest_metar.py`, `ingest_forecasts.py` (new)
- `~/weather_db/verify.py` (new)
- `~/mit_weather_archive/extract_wind.py` (modify to also write into the shared DB)

---

## Phase 6 — CBI dockhouse flag color + flag/wind correlation (added 2026-09-08)

**Goal (per user request):** Community Boating Inc. (CBI)'s dockhouse posts a color-coded
flag (Green/Yellow/Red/Black) reflecting the dockmaster's real-time judgment of sailing
conditions on the Charles River Basin. User wants this tracked over time so it can eventually
be correlated against observed wind speed/gust to build a probability table: "given this wind
speed/gust range, what's the historical distribution of flag colors" (and vice versa).

**Data source found:** `https://www.community-boating.org/about-us/weather-information/`
displays "Current Conditions: X Flag", backed by a clean public API discovered via browser
network inspection:

```
GET https://api.community-boating.org/api/flag
-> var FLAG_COLOR = "Y"   (a JS-snippet response, not JSON -- parsed via regex)
```

Confirmed live 2026-09-08. Codes observed: G/Y/R (green/yellow/red); Black (closed, heavy
weather/lightning) is documented on CBI's own flag-policy page
(`community-boating.org/timetable/flag-color/`) as a possible state, mapped defensively even
though not seen live. This is a clean HTTP GET + regex parse — no browser automation needed
for ongoing ingestion, despite discovering the endpoint via a browser session.

**Schema decision:** flag color is categorical (a human's judgment call), not a numeric sensor
reading like everything else in `observations` — kept in its own table rather than shoehorned
in as a fake "variable" with string values:

```sql
CREATE TABLE flags (
    location_id  TEXT NOT NULL REFERENCES locations(location_id),  -- 'cbi_dockhouse'
    ts_utc       TIMESTAMP NOT NULL,
    flag_color   TEXT NOT NULL,       -- 'green' | 'yellow' | 'red' | 'black'
    source       TEXT NOT NULL,       -- 'cbi_flag_api'
    PRIMARY KEY (location_id, ts_utc)
);
```

Added `cbi_dockhouse` to `locations` (21 David G Mugar Way, Boston — on the Charles River
Basin, near MIT Sailing Pavilion).

**Cadence decision:** flag changes are event-driven (a human updates it on demand), not on a
model-cycle schedule. User initially asked about 30-min polling, then explicitly changed this
to **6x/day (every 4 hours)** to avoid overloading the endpoint — `0 */4 * * *`. Every poll's
result is stored regardless of whether the color changed (not deduplicated to changes-only),
so gaps in the timeline can still be treated as "flag was still X as of this last-known
reading" for correlation purposes.

**Task: Flag ingester**
- Created: `scripts/ingest_flag.py` — fetches, regex-parses `FLAG_COLOR`, inserts into `flags`
- Verified live: correctly recorded `yellow`, matching the site's own displayed "Current
  Conditions: Yellow Flag" banner at time of check
- Scheduled: cron job `541a0489e566` (`sw_ingest_flag.sh`, `0 */4 * * *`)

**Task: Flag/wind correlation report**
- Created: `scripts/report_flag_wind_correlation.py` — nearest-observation match (±20 min
  tolerance) between each flag reading and `mit_pavilion`'s `wind_speed_kt`/`wind_gust_kt`
  observations (chosen over NDBC/CO-OPS/KBOS since MIT Pavilion is the closest wind sensor to
  CBI on the same stretch of the Charles River Basin — the other 3 sources are further away or
  a different body of water and would be less representative of what's actually driving the
  dockmaster's call)
- Buckets by wind-speed range (0-5, 5-10, 10-15, 15-20, 20-25, 25+ kt) and reports, per bucket:
  count/percent of each flag color observed, plus average gust — this is the "probability of
  flag color given wind speed" table the user asked for. A query-time construct (no cron job
  yet — could add a scheduled version like the Phase 4 accuracy report if useful once more
  data accumulates)
- Verified: join logic confirmed correct against a hand-checked example (yellow flag matched
  to 7.6kt sustained / 16.3kt gust at a widened tolerance for the test); at the real 20-min
  tolerance the report currently returns "no matched pairs" because MIT Pavilion's wind updates
  only 2x/day (11am/11pm) vs. the flag's 4-hour cadence — **this is expected data sparsity,
  not a bug**, and will fill in gradually. User was asked whether to also increase MIT
  Pavilion's polling frequency to reduce this sparsity and declined (chose not to add more load
  there) — accepted as a known limitation of the correlation's density, not something to fix
  by over-polling a scraped source.

**Open follow-up (not yet built):** the report currently reduces gust to a single "avg_gust_kt"
per bucket rather than a full 2D speed×gust breakdown. If the user wants the full joint
distribution (not just wind speed as the primary axis), that's a straightforward extension —
bucket both dimensions and cross-tabulate, flagged here for later rather than over-building
before real data volume justifies the extra complexity.
