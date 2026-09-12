-- Sailing Weather DB schema (DuckDB dialect)
-- See docs/plan.md for full design rationale.

-- One row per physical sensor/model-grid-point location we care about
CREATE TABLE IF NOT EXISTS locations (
    location_id   TEXT PRIMARY KEY,   -- e.g. 'mit_pavilion', 'ndbc_44013', 'coops_8443970', 'kbos'
    name          TEXT NOT NULL,
    lat           DOUBLE NOT NULL,
    lon           DOUBLE NOT NULL,
    kind          TEXT NOT NULL       -- 'coastal_station' | 'buoy' | 'airport'
);

-- Raw observations, tidy/long format so heterogeneous sources share one table
CREATE TABLE IF NOT EXISTS observations (
    location_id     TEXT NOT NULL REFERENCES locations(location_id),
    ts_utc          TIMESTAMP NOT NULL,  -- UTC timestamp
    variable        TEXT NOT NULL,       -- 'wind_speed_kt','wind_gust_kt','wind_dir_deg',
                                          -- 'pressure_hpa','air_temp_c','water_temp_c', ...
    value           DOUBLE,
    source          TEXT NOT NULL,       -- 'mit_pixel_scrape' | 'ndbc_realtime2' | 'coops_api' | 'metar'
    quality         TEXT,                -- NULL = normal; 'clipped_high' = value hit a
                                          -- known ceiling in its source (e.g. the MIT
                                          -- pavilion graph's axis max, whatever it is
                                          -- that day) and the true
                                          -- value may be HIGHER than recorded. Added
                                          -- 2026-09-09 after a 26.1kt gust reading traced
                                          -- to the pixel-scraped graph line clipping at its
                                          -- top pixel row -- see scripts/extract_wind.py.
                                          -- 'unverified_scale' = wind_speed_kt/wind_gust_kt
                                          -- readings recorded BEFORE 2026-09-10 that used a
                                          -- WRONG hardcoded 0-30mph axis assumption -- the
                                          -- graph's axis actually auto-scales per day (e.g.
                                          -- confirmed 0-16mph on a calm day), so these older
                                          -- readings may be systematically mis-scaled and
                                          -- cannot be retroactively corrected (the source
                                          -- graph image is a rolling window, not archived).
                                          -- Fixed going forward via OCR-based axis reading
                                          -- (mit_graph_ocr.py) -- see scripts/extract_wind.py.
    PRIMARY KEY (location_id, ts_utc, variable)
);
CREATE INDEX IF NOT EXISTS idx_obs_var_ts ON observations(variable, ts_utc);

-- One row per (model, init_time, location) forecast *run* we ingested.
-- DuckDB has no AUTOINCREMENT; use a SEQUENCE for the surrogate key.
CREATE SEQUENCE IF NOT EXISTS seq_forecast_run_id START 1;
CREATE TABLE IF NOT EXISTS forecast_runs (
    run_id        BIGINT PRIMARY KEY DEFAULT nextval('seq_forecast_run_id'),
    model         TEXT NOT NULL,      -- 'gfs' | 'ecmwf' | 'hrrr'
    location_id   TEXT NOT NULL REFERENCES locations(location_id),
    init_time_utc TIMESTAMP NOT NULL, -- model cycle time, e.g. 2026-09-05 12:00:00
    retrieved_at  TIMESTAMP NOT NULL, -- when WE pulled it (may lag init_time by hours)
    UNIQUE (model, location_id, init_time_utc)
);
-- Speeds up "all runs for this location/model" lookups -- needed now that
-- forecasts are pulled per-location (one location per physical/model grid
-- point) rather than a single shared point for everything.
CREATE INDEX IF NOT EXISTS idx_fr_location_model ON forecast_runs(location_id, model, init_time_utc);

-- The actual forecast values for a run, one row per valid_time x variable
CREATE TABLE IF NOT EXISTS forecast_values (
    run_id         BIGINT NOT NULL REFERENCES forecast_runs(run_id),
    valid_time_utc TIMESTAMP NOT NULL,  -- the time this forecast point is FOR
    variable       TEXT NOT NULL,
    value          DOUBLE,
    PRIMARY KEY (run_id, valid_time_utc, variable)
);
CREATE INDEX IF NOT EXISTS idx_fv_valid ON forecast_values(valid_time_utc, variable);
CREATE INDEX IF NOT EXISTS idx_fv_run ON forecast_values(run_id);

-- Community Boating Inc. (CBI) dockhouse flag color -- a categorical
-- go/no-go signal (Green/Yellow/Red[/Black-closed]) set by a human
-- dockmaster based on current conditions on the Charles River basin.
-- Kept as its own table (not folded into `observations`) because it is
-- categorical, not a numeric measurement, and changes on its own cadence
-- (checked every 30 min, per user 2026-09-08) independent of the
-- wind/weather observation ingesters. The eventual goal (per user) is a
-- probability table correlating observed wind_speed_kt/wind_gust_kt at a
-- given time with the flag color in force at that same time -- e.g. "what
-- fraction of Yellow-flag readings had gusts in the 15-20kt range".
CREATE TABLE IF NOT EXISTS flags (
    location_id  TEXT NOT NULL REFERENCES locations(location_id),  -- 'cbi_dockhouse'
    ts_utc       TIMESTAMP NOT NULL,  -- when WE observed/recorded this flag value
    flag_color   TEXT NOT NULL,       -- 'green' | 'yellow' | 'red' | 'closed' (raw source codes G/Y/R/C normalized; confirmed live 2026-09-10)
    source       TEXT NOT NULL,       -- 'cbi_flag_api'
    PRIMARY KEY (location_id, ts_utc)
);
CREATE INDEX IF NOT EXISTS idx_flags_ts ON flags(ts_utc);

-- Live view: MAE/RMSE per (location, model, variable, lead-hours bucket).
-- Fully computed on read from forecast_values/observations (the permanent
-- raw data) -- no separate snapshot table, no generated files. Query it
-- directly any time: SELECT * FROM v_accuracy_by_lead_time;
-- Nearest-observation matching mirrors db/verify.py's tolerance (7.5 min).
CREATE OR REPLACE VIEW v_accuracy_by_lead_time AS
WITH matched AS (
    WITH candidates AS (
        SELECT
            fv.run_id,
            fv.valid_time_utc,
            fv.variable,
            fv.value AS forecast_value,
            o.ts_utc AS obs_ts_utc,
            o.value AS observed_value,
            abs(epoch(fv.valid_time_utc) - epoch(o.ts_utc)) AS time_diff_seconds,
            row_number() OVER (
                PARTITION BY fv.run_id, fv.valid_time_utc, fv.variable
                ORDER BY abs(epoch(fv.valid_time_utc) - epoch(o.ts_utc))
            ) AS rn
        FROM forecast_values fv
        JOIN forecast_runs fr ON fr.run_id = fv.run_id
        JOIN observations o
            ON o.location_id = fr.location_id
           AND o.variable = fv.variable
           AND o.ts_utc BETWEEN fv.valid_time_utc - INTERVAL '7.5 minutes'
                             AND fv.valid_time_utc + INTERVAL '7.5 minutes'
    )
    SELECT run_id, valid_time_utc, variable, forecast_value, obs_ts_utc,
           observed_value, time_diff_seconds
    FROM candidates
    WHERE rn = 1
),
with_lead AS (
    SELECT
        m.*,
        fr.model,
        fr.location_id,
        date_diff('hour', fr.init_time_utc, m.valid_time_utc) AS lead_hours
    FROM matched m
    JOIN forecast_runs fr ON fr.run_id = m.run_id
),
bucketed AS (
    SELECT
        location_id,
        model,
        variable,
        CASE
            WHEN lead_hours >= 0 AND lead_hours < 6 THEN '0-6h'
            WHEN lead_hours >= 6 AND lead_hours < 12 THEN '6-12h'
            WHEN lead_hours >= 12 AND lead_hours < 24 THEN '12-24h'
            WHEN lead_hours >= 24 AND lead_hours < 48 THEN '24-48h'
            WHEN lead_hours >= 48 AND lead_hours < 72 THEN '48-72h'
            WHEN lead_hours >= 72 THEN '72h+'
            ELSE 'other'
        END AS lead_bucket,
        forecast_value - observed_value AS error
    FROM with_lead
)
SELECT
    location_id,
    model,
    variable,
    lead_bucket,
    count(*) AS n,
    round(avg(abs(error)), 3) AS mae,
    round(sqrt(avg(error * error)), 3) AS rmse
FROM bucketed
GROUP BY location_id, model, variable, lead_bucket;
