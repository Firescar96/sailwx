# Sailing Weather Dashboard

A small local dashboard that visualizes sailing weather data (observations,
model forecasts, forecast accuracy) from the read-only DuckDB database at
`../db/weather.duckdb`. Built with a stdlib-only Python backend and a plain
JS + D3.js frontend — no frontend framework, no extra Python packages.

## How to run it

```
/home/firescar96/.pyenv/versions/3.11.1/bin/python3 /home/firescar96/.hermes/projects/sailing-weather/website/app.py
```

Then open **http://localhost:8420** in a browser.

- Uses the pyenv interpreter that already has `duckdb` installed (no venv,
  no pip installs needed — Flask/FastAPI are NOT installed on this box, so
  the backend is a small `http.server`-based app instead).
- Override the port with `PORT=<port>` in the environment before running.
- The server opens all DuckDB connections with `read_only=True`, so it is
  always safe to run alongside the project's ingestion cron jobs — it
  never contends for the write lock and never writes to the database.
- Stop the server with Ctrl-C (or kill the process) — it holds no
  persistent state.

## What's in the dashboard

1. **Current Conditions** (top panel) — the most recent observation for
   every variable at every location, with a "how long ago" indicator, plus
   the latest CBI dockhouse flag color if available. Cheap to compute and
   gives context before diving into the charts.

2. **Model Accuracy by Lead Time** — a grouped bar chart of `v_accuracy_by_lead_time`
   (MAE per model, grouped by lead-time bucket: 0-6h, 6-12h, 12-24h,
   24-48h, 48-72h, 72h+) for the selected location + variable. Hover a bar
   for exact MAE/RMSE/n. This is the "which model do I trust, and at what
   lead time" view. Faded/zero bars mean no accuracy rows exist yet for
   that model+bucket combination (not necessarily zero error — just no
   data).

3. **Forecast Time Series** — line chart of each model's (GFS/ECMWF/HRRR)
   most recent forecast run for the selected location + variable, with an
   optional dashed grey overlay of recent actual observations for
   comparison. A window selector (24h/48h/72h/7d) controls how far forward
   the chart looks from each model's most recent init time. Click any
   point on a model's line to jump straight to the Forecast Convergence
   chart for that exact target time.

4. **Forecast Convergence** — for one fixed target (valid) time, shows how
   each model's prediction for that time changed across successive model
   runs as it got closer to being realized. Pick a target time manually
   (UTC) or click a point on the Forecast Time Series chart above. This
   mirrors the "forecast convergence" concept used elsewhere in the
   project (see `scripts/report_forecast_convergence.py`).

A dropdown at the top switches the location (all 5: `mit_pavilion`,
`cbi_dockhouse` on the Charles River Basin; `ndbc_44013`, `coops_8443970`,
`kbos` in/near Boston Inner Harbor) and the variable
(`wind_speed_kt`/`wind_gust_kt` always offered; other variables like
`pressure_hpa`/`air_temp_c`/`wind_dir_deg` appear automatically when the
selected location has accuracy data for them).

## API endpoints

All under the same server, JSON responses:

- `GET /api/locations` — the 5 rows of `locations`.
- `GET /api/current-conditions` — most recent observation per (location, variable).
- `GET /api/flags-latest` — most recent flag color per location (currently only cbi_dockhouse has rows).
- `GET /api/accuracy?location=<id>&variable=<var>` — rows from `v_accuracy_by_lead_time` for that location+variable.
- `GET /api/accuracy-variables` — every (location, variable) pair that actually has accuracy rows, used to populate the variable dropdown.
- `GET /api/forecast?location=<id>&variable=<var>&hours=<n>` — latest run per model, forecast values out to `hours` ahead of each run's init time.
- `GET /api/forecast-convergence?location=<id>&variable=<var>&target=<ISO timestamp>` — every model run's prediction for one fixed valid_time_utc.
- `GET /api/observations?location=<id>&variable=<var>&hours=<n>` — raw observations for the last `hours` before the most recent observation at that location (used for the forecast chart overlay).
- `GET /api/variables-for-location?location=<id>` — which observation variables actually exist at a location.

## Known limitations

- **Sparse data at some locations.** `cbi_dockhouse` has no wind
  observations at all (by design — it only has flag color history), so its
  Current Conditions card and both charts will show "no data" for wind
  variables at that location; this is real data absence, not a bug.
  `kbos` and `mit_pavilion` have far fewer historical rows than
  `ndbc_44013` (tens vs. thousands), simply because those ingestion
  sources were added more recently — accuracy bars for those locations
  will have low `n` and should be read with that in mind.
- **`v_accuracy_by_lead_time` only currently has non-trivial data for**
  `kbos`, `ndbc_44013`, and `mit_pavilion` (no forecast history yet for
  `coops_8443970`/`cbi_dockhouse` at time of writing) — the accuracy chart
  will show "no accuracy data" if you pick a location/variable combo
  without matched forecast+observation pairs.
- **Forecast chart shows only the single most recent run per model.** By
  design (per spec: "show the most recent run's prediction by default").
  Use the Forecast Convergence chart to see how earlier runs predicted the
  same target time.
- **No auth, no HTTPS, no production hardening** — this is a local-only
  tool per the original spec. Do not expose it beyond localhost as-is.
- Timestamps are all UTC throughout (both stored in the DB and displayed
  in the UI) to match the underlying data — there is no local-timezone
  conversion.

## Architecture

- `app.py` — single-file Python backend using only the standard library
  (`http.server`) plus the `duckdb` package. No pip installs required
  beyond what's already on the box. Serves static files from `static/`
  and JSON from `/api/*`.
- `static/index.html` / `static/style.css` / `static/app.js` — plain HTML +
  CSS + vanilla JS. D3 v7 is loaded from a CDN
  (`https://cdn.jsdelivr.net/npm/d3@7`) — requires internet access on
  first load to fetch the CDN script; no local vendoring was done since
  the box already has outbound access for CDN fetches in dev.

## Verifying it works

```
/home/firescar96/.pyenv/versions/3.11.1/bin/python3 app.py &
curl -s http://localhost:8420/api/locations
curl -s "http://localhost:8420/api/accuracy?location=ndbc_44013&variable=wind_speed_kt"
```

Both should return real JSON rows pulled live from `weather.duckdb`
(verified during development — see project git history / PR description
for example output).
