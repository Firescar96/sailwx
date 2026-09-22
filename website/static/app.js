// Sailing Weather Dashboard frontend. Plain JS + D3 v7, no framework.

const MODEL_COLORS = {
  gfs: getCssVar('--gfs'),
  ecmwf: getCssVar('--ecmwf'),
  hrrr: getCssVar('--hrrr'),
  icon: getCssVar('--icon'),
  nam: getCssVar('--nam'),
  hrdps: getCssVar('--hrdps'),
};

const FLAG_COLORS = {
  green: getCssVar('--flag-green'),
  yellow: getCssVar('--flag-yellow'),
  red: getCssVar('--flag-red'),
  closed: getCssVar('--flag-black'),
};

const VARIABLE_LABELS = {
  wind_speed_kt: 'Wind Speed (kt)',
  wind_gust_kt: 'Wind Gust (kt)',
  wind_dir_deg: 'Wind Direction (deg)',
  pressure_hpa: 'Pressure (hPa)',
  air_temp_f: 'Air Temp (°F)',
  water_temp_f: 'Water Temp (°F)',
  water_level_ft_mllw: 'Water Level (ft MLLW)',
};

// 'other' (negative lead-time / hindcast backfill rows -- past_days/
// past_hours data added so the forecast time-series chart can show
// historical model context, see ingest_forecasts.py) is intentionally
// excluded here. It's real, legitimate data, just not "forecast skill
// at a given lead time" (there's no lead time when valid_time predates
// init_time), so it doesn't belong in the Model Accuracy chart --
// user-requested 2026-09-11 to skip it there specifically.
const LEAD_BUCKET_ORDER = ['0-6h', '6-12h', '12-24h', '24-48h', '48-72h', '72h+'];

function getCssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function fmtVarLabel(v) {
  return VARIABLE_LABELS[v] || v;
}

// Every chart's tooltip div was previously created via a bare
// `d3.select('body').append('div').attr('class', '...')` inside that
// chart's own load function -- since these load functions re-run on
// every dropdown change (model/hours/window/etc), each re-render
// appended ANOTHER tooltip div to <body> without ever removing the
// previous one. Normally invisible (opacity:0 at rest, and whichever
// one you're currently hovering looks fine on its own), but changing a
// dropdown WHILE a tooltip happened to be visible left the stale one
// stuck on-screen permanently, on top of a newly-created one for the
// same chart -- exactly the "tooltips get stuck when changing time
// period" bug reported 2026-09-22 (visible in a screenshot showing two
// overlapping CBI Flag Prediction tooltips at once). Fix: a single
// shared helper that removes ANY pre-existing tooltip(s) with the given
// class before creating a fresh one, so there is always at most one
// live tooltip div per chart at a time, no matter how many times its
// load function has re-run.
function getOrCreateTooltip(className) {
  d3.selectAll(`div.${className}`).remove();
  return d3.select('body').append('div').attr('class', className);
}

const state = {
  locations: [],
  location: null,
  variable: 'wind_speed_kt',
  accuracyVarsByLocation: {}, // location_id -> Set(variable)
  hours: 336, // matches the <select> default in index.html (Next 14 days) -- 14d is roughly GFS/ECMWF's own forecast horizon (the longest of the 6 models)
  pastHours: 168, // matches the <select> default in index.html (Last 7 days)
  hiddenModels: new Set(), // model keys (or 'observed') toggled off via legend click, Forecast Time Series chart
  hiddenStabilityModels: new Set(), // same idea, separate set, Forecast Stability chart (user 2026-09-21: "hover to brighten works well, but only one chart has the click to temporarily hide")
};

// ---------------------------------------------------------------------------
// Control-state persistence: remembers every dropdown/selector's value
// across a page refresh (user 2026-09-19: "I need to be able to save all
// state of drop-downs and selectors in an organized way so it comes back on
// refresh"). Deliberately a small hand-written localStorage wrapper, not a
// state-management library (Vue/Redux/Zustand/etc, discussed and explicitly
// decided against with the user) -- persisting ~14 <select> values is a
// "read on load, write on change" problem, not something that needs
// reactivity or component composition, and this dashboard's existing
// plain-JS-plus-D3 approach (no build step, no framework) would fight with
// any framework's own DOM ownership rather than benefit from it.
// ---------------------------------------------------------------------------

const CONTROL_STORAGE_KEY = 'sailwx.controls.v1';

const ControlStorage = {
  _cache: null,
  _load() {
    if (this._cache) return this._cache;
    try {
      this._cache = JSON.parse(localStorage.getItem(CONTROL_STORAGE_KEY) || '{}');
    } catch (e) {
      console.warn('ControlStorage: saved state was corrupt, resetting', e);
      this._cache = {};
    }
    return this._cache;
  },
  get(id) {
    return this._load()[id];
  },
  set(id, value) {
    const data = this._load();
    data[id] = value;
    try {
      localStorage.setItem(CONTROL_STORAGE_KEY, JSON.stringify(data));
    } catch (e) {
      // localStorage can throw (private-browsing quota, disabled storage,
      // etc.) -- persistence is a nice-to-have, never let it break the
      // dashboard itself.
      console.warn('ControlStorage: failed to save', id, e);
    }
  },
};

// Restores a <select>'s value from storage IF a saved value exists AND is
// still a valid option on that element right now (the option list can
// legitimately differ between sessions -- e.g. a location added/removed,
// or the Wind Rose/Gustiness model lists depend on MODEL_COLORS staying
// the same 6 keys, which it always does today, but this stays defensive
// regardless). Returns the resolved value (so callers can sync any
// corresponding `state.*` field too), or null if nothing was restored --
// in which case the element is left exactly as it already was (its HTML
// `selected` default, or whatever earlier init() code already set).
function restoreSelectValue(id) {
  const saved = ControlStorage.get(id);
  if (saved == null) return null;
  const el = document.getElementById(id);
  if (!el) return null;
  const validOptions = Array.from(el.options).map(o => o.value);
  if (!validOptions.includes(saved)) return null;
  el.value = saved;
  return saved;
}

// Adds a NAMESPACED 'change.persist' listener that saves the control's new
// value on every change. Namespaced specifically so it coexists with each
// panel's own '.on("change", ...)' chart-reload handler instead of
// clobbering it -- d3 (like native addEventListener under the hood) only
// replaces a listener when both the event name AND namespace match, so
// 'change' and 'change.persist' are independent and both fire.
function persistSelectOnChange(id) {
  d3.select(`#${id}`).on('change.persist', function () {
    ControlStorage.set(id, this.value);
  });
}

// Base path this page was actually loaded under (e.g. "" at the domain
// root, "/sailwx" when reverse-proxied under a subpath -- see Tailscale
// Serve's --set-path, used to publish this dashboard at
// https://<tailnet>.ts.net/sailwx). Serve's path-based routing STRIPS the
// prefix before proxying to the backend (confirmed empirically 2026-09-19:
// a request for /sailwx/style.css reaches this Flask-ish app as a request
// for /style.css), so the page's own static assets (href="/style.css" etc
// in index.html) work unmodified -- but a client-side `fetch('/api/...')`
// call is a BROWSER-side absolute-path request that has no knowledge of
// the server-side prefix stripping and resolves against the domain root,
// not wherever the page itself was loaded from. That silently 404'd every
// API call once this dashboard moved off the domain root. Fix: derive the
// real mount prefix from the page's own URL once (this app is a single
// page with no client-side routing, so window.location.pathname IS the
// mount path, not a sub-route) and prepend it to '/api/...' fetches.
const APP_BASE_PATH = (() => {
  const p = window.location.pathname.replace(/\/index\.html$/, '').replace(/\/$/, '');
  return p; // '' at the domain root, e.g. '/sailwx' when mounted under a subpath
})();

async function fetchJSON(url) {
  const finalUrl = url.startsWith('/api/') ? APP_BASE_PATH + url : url;
  const res = await fetch(finalUrl);
  if (!res.ok) throw new Error(`${finalUrl} -> HTTP ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  const [locations, accuracyVars] = await Promise.all([
    fetchJSON('/api/locations'),
    fetchJSON('/api/accuracy-variables'),
  ]);

  state.locations = locations;
  accuracyVars.forEach(row => {
    if (!state.accuracyVarsByLocation[row.location_id]) {
      state.accuracyVarsByLocation[row.location_id] = new Set();
    }
    state.accuracyVarsByLocation[row.location_id].add(row.variable);
  });

  const locSelect = d3.select('#location-select');
  locSelect.selectAll('option')
    .data(locations)
    .join('option')
    .attr('value', d => d.location_id)
    .text(d => d.name);

  state.location = restoreSelectValue('location-select') || locations[0].location_id;
  locSelect.property('value', state.location);
  locSelect.on('change', function () {
    state.location = this.value;
    state.hiddenModels.clear(); // don't carry a stale hidden-set across to a different location's chart
    state.hiddenStabilityModels.clear();
    refreshVariableOptions();
    refreshAll();
    updatePredictionPanelVisibility();
    if (state.location === 'cbi_dockhouse') {
      // BUG FIXED 2026-09-22 (user: "CBI Flag Prediction is messed up,
      // date dropdown broken" -- reproduced live: switching TO cbi_dockhouse
      // via the location dropdown left the Flag Prediction panel's chart
      // area completely empty, no chart AND no "no data" message, even
      // though the model/hours dropdowns rendered fine and manually
      // calling loadFlagPredictionChart() from the console worked
      // perfectly). Root cause: this handler called
      // loadWindPredictionChart/loadWindRoseChart/loadGustFactorChart
      // when switching to any OTHER location, but never called
      // loadFlagPredictionChart when switching TO cbi_dockhouse -- it
      // only ever ran via init()'s initial-page-load path, which is why
      // this looked fine as long as CBI happened to be the default
      // location every time it was tested.
      loadFlagPredictionChart();
    } else {
      loadWindPredictionChart();
      loadWindRoseChart();
      loadGustFactorChart();
    }
  });
  persistSelectOnChange('location-select');

  // Model Accuracy panel: time-window selector (user 2026-09-21: "add a
  // dropdown that let's me pick a time window, with a default of 7
  // days" -- following on from confirming the chart previously had NO
  // time filter at all, i.e. was genuinely all-time/all-history).
  restoreSelectValue('accuracy-hours-select');
  d3.select('#accuracy-hours-select').on('change', loadAccuracyChart);
  persistSelectOnChange('accuracy-hours-select');

  d3.select('#hours-select').on('change', function () {
    state.hours = +this.value;
    loadForecastChart();
  });
  if (restoreSelectValue('hours-select')) state.hours = +document.getElementById('hours-select').value;
  persistSelectOnChange('hours-select');

  d3.select('#past-hours-select').on('change', function () {
    state.pastHours = +this.value;
    loadForecastChart();
  });
  if (restoreSelectValue('past-hours-select')) state.pastHours = +document.getElementById('past-hours-select').value;
  persistSelectOnChange('past-hours-select');

  // Forecast Stability panel (replaces the old Forecast Convergence panel
  // entirely, per explicit user instruction 2026-09-14: "i want the
  // convergence chart to be a day over day forecast stability instead" /
  // "Replace the Forecast Convergence PANEL entirely... remove the old
  // click-a-point/target-time picker UI"). Answers a different question
  // than convergence did: not "how did predictions for ONE fixed target
  // time evolve," but "right now, how much is each model still changing
  // its own mind hour-to-hour across its last few runs" -- see
  // api_forecast_stability in app.py.
  restoreSelectValue('stability-num-runs-select');
  restoreSelectValue('stability-hours-select');
  d3.select('#stability-num-runs-select').on('change', loadStabilityChart);
  d3.select('#stability-hours-select').on('change', loadStabilityChart);
  persistSelectOnChange('stability-num-runs-select');
  persistSelectOnChange('stability-hours-select');

  // Flag Prediction panel (CBI only) and Wind Prediction panel (all other
  // locations with real wind observations) -- separate from the existing
  // Forecast/Accuracy charts (user explicit direction 2026-09-11: "do not
  // ruin or mess up the existing charts... I want additional overlay or a
  // separate chart"; then 2026-09-11 again: "only show CBI Flag Prediction
  // on the CBI page, on the other pages... show a very similar graph, but
  // do wind prediction instead"). Model lists reuse MODEL_COLORS' keys
  // since that's the same 6-model set the forecast ingester populates
  // everywhere.
  const flagModelSelect = d3.select('#flag-prediction-model-select');
  flagModelSelect.selectAll('option')
    .data(Object.keys(MODEL_COLORS))
    .join('option')
    .attr('value', d => d)
    .text(d => d.toUpperCase());
  flagModelSelect.property('value', restoreSelectValue('flag-prediction-model-select') || 'gfs');
  flagModelSelect.on('change', loadFlagPredictionChart);
  persistSelectOnChange('flag-prediction-model-select');
  restoreSelectValue('flag-prediction-hours-select');
  d3.select('#flag-prediction-hours-select').on('change', loadFlagPredictionChart);
  persistSelectOnChange('flag-prediction-hours-select');

  const windModelSelect = d3.select('#wind-prediction-model-select');
  windModelSelect.selectAll('option')
    .data(Object.keys(MODEL_COLORS))
    .join('option')
    .attr('value', d => d)
    .text(d => d.toUpperCase());
  windModelSelect.property('value', restoreSelectValue('wind-prediction-model-select') || 'gfs');
  windModelSelect.on('change', loadWindPredictionChart);
  persistSelectOnChange('wind-prediction-model-select');
  restoreSelectValue('wind-prediction-hours-select');
  d3.select('#wind-prediction-hours-select').on('change', loadWindPredictionChart);
  persistSelectOnChange('wind-prediction-hours-select');

  // Wind Rose panel: model select has an extra "Observed only" option
  // (empty value) since the comparison model is optional.
  const roseModelSelect = d3.select('#wind-rose-model-select');
  roseModelSelect.selectAll('option.model-option')
    .data(Object.keys(MODEL_COLORS))
    .join('option')
    .attr('class', 'model-option')
    .attr('value', d => d)
    .text(d => d.toUpperCase());
  restoreSelectValue('wind-rose-model-select');
  roseModelSelect.on('change', loadWindRoseChart);
  persistSelectOnChange('wind-rose-model-select');
  restoreSelectValue('wind-rose-hours-select');
  d3.select('#wind-rose-hours-select').on('change', loadWindRoseChart);
  persistSelectOnChange('wind-rose-hours-select');

  // Gustiness panel: 1-day-forward extension (user 2026-09-19: "I need
  // to be able to see a one day forward looking addition to the
  // charts... future data using a model of my choice"). Model select
  // has the same "None (observed only)" empty-value option as Wind
  // Rose's comparison model, since the forecast extension is optional.
  const gustModelSelect = d3.select('#gust-factor-model-select');
  gustModelSelect.selectAll('option.model-option')
    .data(Object.keys(MODEL_COLORS))
    .join('option')
    .attr('class', 'model-option')
    .attr('value', d => d)
    .text(d => d.toUpperCase());
  restoreSelectValue('gust-factor-model-select');
  gustModelSelect.on('change', loadGustFactorChart);
  persistSelectOnChange('gust-factor-model-select');
  restoreSelectValue('gust-factor-hours-select');
  d3.select('#gust-factor-hours-select').on('change', loadGustFactorChart);
  persistSelectOnChange('gust-factor-hours-select');

  updatePredictionPanelVisibility();

  refreshVariableOptions();
  await refreshAll();
  // Independent panels, not part of refreshAll's location/variable-driven
  // set -- only load whichever one is actually visible for this location.
  if (state.location === 'cbi_dockhouse') {
    await loadFlagPredictionChart();
  } else {
    await Promise.all([
      loadWindPredictionChart(),
      loadWindRoseChart(),
      loadGustFactorChart(),
    ]);
  }
}

// CBI is the only location with flag data (no wind sensor of its own);
// every other location has real wind observations instead. Exactly one
// of these two panels is ever shown at a time. Wind Rose + Gust Factor
// are ALSO only meaningful at locations with real wind sensors (same
// condition as Wind Prediction) -- per user 2026-09-11 ("only on places
// with observed wind data").
function updatePredictionPanelVisibility() {
  const isCbi = state.location === 'cbi_dockhouse';
  d3.select('#flag-prediction-panel').attr('hidden', isCbi ? null : true);
  d3.select('#wind-prediction-panel').attr('hidden', isCbi ? true : null);
  d3.select('#wind-rose-panel').attr('hidden', isCbi ? true : null);
  d3.select('#gust-factor-panel').attr('hidden', isCbi ? true : null);
}

function allVariablesUnion() {
  const s = new Set();
  Object.values(state.accuracyVarsByLocation).forEach(set => set.forEach(v => s.add(v)));
  // Always ensure the two required variables are present as options even
  // if a given location has no accuracy rows yet for them.
  s.add('wind_speed_kt');
  s.add('wind_gust_kt');
  return Array.from(s);
}

function refreshVariableOptions() {
  const varSelect = d3.select('#variable-select');
  const vars = allVariablesUnion().sort();
  // Only pull the persisted value in on the FIRST call (page load) --
  // refreshVariableOptions also runs on every location change, and by
  // then state.variable already reflects the user's current in-session
  // choice, which should win over a stale saved value from a previous
  // session/location.
  let current = state.variable;
  if (!refreshVariableOptions._restored) {
    refreshVariableOptions._restored = true;
    const saved = ControlStorage.get('variable-select');
    if (saved != null && vars.includes(saved)) current = saved;
  }

  varSelect.selectAll('option')
    .data(vars)
    .join('option')
    .attr('value', d => d)
    .text(d => fmtVarLabel(d));

  if (vars.includes(current)) {
    state.variable = current;
    varSelect.property('value', current);
  } else {
    state.variable = vars[0];
    varSelect.property('value', state.variable);
  }

  varSelect.on('change', function () {
    state.variable = this.value;
    refreshAll();
  });
  persistSelectOnChange('variable-select');
}

async function refreshAll() {
  await Promise.all([
    loadCurrentConditions(),
    loadAccuracyChart(),
    loadForecastChart(),
    loadStabilityChart(),
  ]);
}

// ---------------------------------------------------------------------------
// Current conditions panel
// ---------------------------------------------------------------------------

function ageString(isoTs) {
  const then = new Date(isoTs + 'Z'); // ts_utc has no offset suffix from the API
  const diffMs = Date.now() - then.getTime();
  const mins = Math.round(diffMs / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 48) return `${hrs}h ago`;
  return `${Math.round(hrs / 24)}d ago`;
}

async function loadCurrentConditions() {
  const [conditions, flags] = await Promise.all([
    fetchJSON('/api/current-conditions'),
    fetchJSON('/api/flags-latest'),
  ]);

  const byLocation = d3.group(conditions, d => d.location_id);
  const grid = d3.select('#conditions-grid');

  const cards = grid.selectAll('.condition-card')
    .data(state.locations, d => d.location_id)
    .join('div')
    .attr('class', 'condition-card');

  cards.each(function (loc) {
    const rows = (byLocation.get(loc.location_id) || [])
      .slice()
      .sort((a, b) => a.variable.localeCompare(b.variable));
    const card = d3.select(this);
    card.selectAll('*').remove();
    card.append('h3').text(loc.name);
    if (rows.length === 0) {
      card.append('div').attr('class', 'condition-age').text('No observations yet.');
      return;
    }
    let latestTs = rows[0].ts_utc;
    rows.forEach(r => {
      if (r.ts_utc > latestTs) latestTs = r.ts_utc;
      const row = card.append('div').attr('class', 'condition-row');
      row.append('span').text(fmtVarLabel(r.variable));
      row.append('span').attr('class', 'val').text(
        typeof r.value === 'number' ? r.value.toFixed(2) : r.value
      );
    });
    card.append('div').attr('class', 'condition-age').text(`updated ${ageString(latestTs)}`);
  });

  const flagsRow = d3.select('#flags-row');
  flagsRow.selectAll('*').remove();
  flags.forEach(f => {
    const loc = state.locations.find(l => l.location_id === f.location_id);
    const badge = flagsRow.append('span')
      .attr('class', 'flag-badge')
      .style('background', FLAG_COLORS[f.flag_color] || '#555')
      .text(`${loc ? loc.name : f.location_id}: ${f.flag_color.toUpperCase()} flag (${ageString(f.ts_utc)})`);
  });
}

// ---------------------------------------------------------------------------
// Accuracy chart: grouped bar chart, one group per lead bucket, one bar per model
// ---------------------------------------------------------------------------

async function loadAccuracyChart() {
  const hours = d3.select('#accuracy-hours-select').property('value');
  const url = hours
    ? `/api/accuracy?location=${state.location}&variable=${state.variable}&hours=${hours}`
    : `/api/accuracy?location=${state.location}&variable=${state.variable}&hours=`;
  const data = await fetchJSON(url);
  const container = d3.select('#accuracy-chart');
  container.selectAll('*').remove();
  d3.select('#accuracy-empty').attr('hidden', data.length ? true : null);
  if (!data.length) return;

  const buckets = LEAD_BUCKET_ORDER.filter(b => data.some(d => d.lead_bucket === b));
  const models = Array.from(new Set(data.map(d => d.model))).sort();

  const width = Math.min(900, container.node().clientWidth || 900);
  const height = 320;
  const margin = { top: 20, right: 20, bottom: 55, left: 55 };

  const svg = container.append('svg')
    .attr('width', width)
    .attr('height', height);

  const x0 = d3.scaleBand().domain(buckets).range([margin.left, width - margin.right]).padding(0.25);
  const x1 = d3.scaleBand().domain(models).range([0, x0.bandwidth()]).padding(0.1);
  const y = d3.scaleLinear()
    .domain([0, d3.max(data, d => d.mae) * 1.15 || 1])
    .nice()
    .range([height - margin.bottom, margin.top]);

  // gridlines
  svg.append('g')
    .selectAll('line')
    .data(y.ticks(5))
    .join('line')
    .attr('class', 'grid-line')
    .attr('x1', margin.left).attr('x2', width - margin.right)
    .attr('y1', d => y(d)).attr('y2', d => y(d));

  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x0))
    .selectAll('text')
    .attr('transform', 'rotate(-25)')
    .style('text-anchor', 'end');

  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(5));

  svg.append('text')
    .attr('x', -height / 2).attr('y', 16)
    .attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle')
    .attr('fill', 'var(--muted)')
    .attr('font-size', '0.75rem')
    .text('MAE');

  const tooltip = getOrCreateTooltip('bar-tooltip');

  const byBucket = d3.group(data, d => d.lead_bucket);

  const bucketGroups = svg.append('g')
    .selectAll('g')
    .data(buckets)
    .join('g')
    .attr('transform', b => `translate(${x0(b)},0)`);

  bucketGroups.selectAll('rect')
    .data(bucket => models.map(m => {
      const row = (byBucket.get(bucket) || []).find(d => d.model === m);
      return row ? { ...row } : { model: m, lead_bucket: bucket, mae: 0, rmse: 0, n: 0, missing: true };
    }))
    .join('rect')
    .attr('x', d => x1(d.model))
    .attr('width', x1.bandwidth())
    .attr('y', d => d.missing ? y(0) : y(d.mae))
    .attr('height', d => d.missing ? 0 : y(0) - y(d.mae))
    .attr('fill', d => MODEL_COLORS[d.model] || '#888')
    .attr('opacity', d => d.missing ? 0.15 : 0.9)
    .on('mousemove', (event, d) => {
      if (d.missing) return;
      tooltip.style('opacity', 1)
        .html(`<b>${d.model.toUpperCase()}</b> · ${d.lead_bucket}<br>MAE: ${d.mae}<br>RMSE: ${d.rmse}<br>n=${d.n}`)
        .style('left', (event.pageX + 12) + 'px')
        .style('top', (event.pageY - 10) + 'px');
    })
    .on('mouseleave', () => tooltip.style('opacity', 0));

  // legend
  const legend = container.insert('div', 'svg').attr('class', 'legend');
  models.forEach(m => {
    const item = legend.append('div').attr('class', 'legend-item');
    item.append('span').attr('class', 'legend-swatch').style('background', MODEL_COLORS[m] || '#888');
    item.append('span').text(m.toUpperCase());
  });
}

// ---------------------------------------------------------------------------
// Forecast time-series chart: one line per model, latest run, optional obs overlay
// ---------------------------------------------------------------------------

let forecastXScale = null;

async function loadForecastChart() {
  const hours = state.hours;
  const pastHours = state.pastHours;
  // Observation overlay window: strictly follows "Also show past". If
  // it's OFF (0), show ZERO trailing observation history -- previously
  // this fell back to a 6h "recent context" default even when Off,
  // which visibly contradicted what "Off" should mean (user-reported
  // 2026-09-10/11: "forecast show past Off doesnt work, i can always
  // see some history"). Off now genuinely means no observed overlay at
  // all; only current-conditions data (shown elsewhere on the page)
  // reflects "right now" when past-hours is Off.
  const obsHours = pastHours;
  const showFlagBands = state.location === 'cbi_dockhouse'; // only CBI has flag data
  // Flag history window follows ONLY "Also show past" (obsHours), not
  // the forward-looking forecast window (`hours`) at all. Previously
  // used Math.max(hours, obsHours), which meant a large forward window
  // (e.g. "Next 14 days" = 336h) forced the flag lookback to 336h too,
  // showing flag data from "since the beginning," far past what the
  // past-hours selector actually asked for (user-reported 2026-09-11:
  // "the flag is still showing since beginning of data, not for the
  // time period I want"). Flags have no future data anyway, so there
  // was never a reason to couple this to the forward-looking `hours`.
  let [forecast, obs, flagHistory] = await Promise.all([
    fetchJSON(`/api/forecast?location=${state.location}&variable=${state.variable}&hours=${hours}&past_hours=${pastHours}`),
    obsHours > 0
      ? fetchJSON(`/api/observations?location=${state.location}&variable=${state.variable}&hours=${obsHours}`)
      : Promise.resolve([]), // Off (0) means genuinely no observed overlay, not "hours=0" API edge case
    showFlagBands
      ? fetchJSON(`/api/flags-history?location=${state.location}&hours=${obsHours}`)
      : Promise.resolve([]),
  ]);

  const container = d3.select('#forecast-chart');
  container.selectAll('*').remove();
  d3.select('#forecast-empty').attr('hidden', forecast.length ? true : null);
  if (!forecast.length) { forecastXScale = null; return; }

  forecast.forEach(d => { d.valid_time_utc_date = new Date(d.valid_time_utc + 'Z'); });
  obs.forEach(d => { d.ts_utc_date = new Date(d.ts_utc + 'Z'); });
  flagHistory.forEach(d => { d.ts_utc_date = new Date(d.ts_utc + 'Z'); });

  // X-axis domain: computed EXPLICITLY from the selected forward/backward
  // windows (now +/- hours), not derived from d3.extent() of whatever
  // dates happen to be present in forecast+obs+flagHistory combined.
  // Previously combining all three data sources' actual dates into one
  // extent() meant the domain's min/max silently depended on which
  // series happened to have the widest real data, decoupled from what
  // the user actually selected in the Window/Also-show-past controls --
  // this is also what let the flag strip's window drift out of sync
  // (user-reported 2026-09-11: "separate in code the minimum date and
  // maximum date and don't combine the forward and backward windows").
  // minDate is exactly "now - pastHours" (or "now" if Off); maxDate is
  // exactly "now + hours" -- both independent, neither derived from the
  // other or from any dataset's actual contents.
  const nowForDomain = new Date();
  const minDate = new Date(nowForDomain.getTime() - pastHours * 3600000);
  const maxDate = new Date(nowForDomain.getTime() + hours * 3600000);

  // Clip observed/forecast/flag points to exactly [minDate, maxDate]
  // BEFORE grouping by model -- this must happen before `byModel`/
  // `models` are built below (a model whose latest run's own past-data
  // window only reaches back e.g. 48h, while the user selected "Last 7
  // days," has no reason to have points outside its own real data, but
  // when it DOES have a point just past minDate due to how each model's
  // backfill window is computed server-side, that stray point was still
  // being drawn -- filtering only happened later on a variable that
  // `byModel`/the actual <path>/<circle> rendering never used, since it
  // read from the ORIGINAL unfiltered array. This caused model lines to
  // visibly extend past the chart's own plotted x-range (user-reported
  // 2026-09-11: "models breaking chart bounds because they are limited
  // to the min backwards window"). Filtering here, before any grouping,
  // guarantees every rendered point is truly within the chart's exact
  // window.
  obs = obs.filter(d => d.ts_utc_date >= minDate && d.ts_utc_date <= maxDate);
  forecast = forecast.filter(d => d.valid_time_utc_date >= minDate && d.valid_time_utc_date <= maxDate);
  flagHistory = flagHistory.filter(d => d.ts_utc_date >= minDate && d.ts_utc_date <= maxDate);

  const models = Array.from(new Set(forecast.map(d => d.model))).sort();
  const byModel = d3.group(forecast, d => d.model);

  const width = Math.min(900, container.node().clientWidth || 900);
  const flagStripSpace = showFlagBands ? 22 : 0; // extra room below the axis for the flag strip + its own label
  const height = 340 + flagStripSpace;
  const margin = { top: 20, right: 20, bottom: 95 + flagStripSpace, left: 55 };

  const x = d3.scaleTime()
    .domain([minDate, maxDate])
    .range([margin.left, width - margin.right]);
  forecastXScale = x;

  // Y-axis domain only considers currently-VISIBLE series (respecting
  // hiddenModels) so toggling a model off via the legend rescales the
  // chart sensibly around what's actually shown, instead of leaving
  // dead space sized for a hidden series' range.
  const visibleForecast = forecast.filter(d => !state.hiddenModels.has(d.model));
  const visibleObs = state.hiddenModels.has('observed') ? [] : obs;
  const allValues = visibleForecast.map(d => d.value).concat(visibleObs.map(d => d.value)).filter(v => v != null);
  const y = d3.scaleLinear()
    .domain([d3.min(allValues) * (d3.min(allValues) < 0 ? 1.1 : 0.9), d3.max(allValues) * 1.1])
    .nice()
    .range([height - margin.bottom, margin.top]);

  const svg = container.append('svg').attr('width', width).attr('height', height);

  // Single shared tooltip div for this whole chart render, reused by
  // the flag-color strip, every model's line/dots, and the observed
  // overlay -- BUG FIXED 2026-09-22 (user: "tooltips get stuck when
  // changing time period", reproduced via a screenshot showing two
  // overlapping stuck tooltips on the CBI Flag Prediction chart).
  // Previously each of those three usages independently created its
  // OWN "shared" tooltip via `d3.select('body').selectAll(some-class)
  // .data([0]).join('div').attr('class','point-tooltip')` -- but each
  // used a DIFFERENT selector class (.flag-band-tooltip / .point-tooltip
  // / .obs-tooltip) while all three actually SET class="point-tooltip",
  // so the selectAll() each one used never matched anything (wrong
  // class name), meaning .join() created a fresh div every single time
  // regardless of what already existed. Worse, the per-model-dots one
  // was INSIDE a forEach loop, so it alone created 6 new tooltip divs
  // per chart re-render (one per model) that were then never removed.
  // A dropdown change that fired mid-hover left whichever tooltip was
  // visible at that moment permanently stuck on screen, with newer
  // renders piling up their own on top. Fix: getOrCreateTooltip()
  // removes any existing element(s) with the given class before
  // creating exactly one fresh one, and this single call/variable is
  // now shared across all three usages in this function instead of
  // each site creating its own.
  const tooltip = getOrCreateTooltip('point-tooltip');

  // Flag-color strip (CBI only): each flag reading is a point in time,
  // extended as a colored segment until the next reading (or the
  // chart's right edge for the last one). Rendered as a thick line
  // hugging the x-axis rather than a full-height background wash --
  // reads more like a discrete status timeline than a shaded region.
  if (flagHistory.length) {
    // Simple "carry forward last known value" band: each reading's color
    // extends until the next reading (or the chart's right edge for the
    // last one). No uncertainty/gap-dimming concept -- per explicit user
    // direction 2026-09-11 ("remove the uncertainty concept and just
    // fill in with the last data known for past data"), a missed poll
    // just means the last known flag color is assumed to have persisted,
    // same as any other "last observation carried forward" series on
    // this dashboard.
    const bands = flagHistory.map((d, i) => {
      const next = flagHistory[i + 1];
      const end = next ? next.ts_utc_date : new Date(x.domain()[1]);
      return { color: d.flag_color, start: d.ts_utc_date, end };
    });
    const flagStripY = height - 10; // in the extra bottom space, below the rotated axis tick labels
    svg.append('text')
      .attr('x', margin.left - 6).attr('y', flagStripY + 3)
      .attr('text-anchor', 'end')
      .attr('fill', 'var(--muted)')
      .attr('font-size', '0.68rem')
      .text('Flag');
    svg.append('g')
      .attr('class', 'flag-strip')
      .selectAll('line')
      .data(bands)
      .join('line')
      .attr('x1', d => x(d.start))
      .attr('x2', d => Math.max(x(d.start), x(d.end)))
      .attr('y1', flagStripY)
      .attr('y2', flagStripY)
      .attr('stroke', d => FLAG_COLORS[d.color] || '#888')
      .attr('stroke-width', 8)
      .attr('stroke-linecap', 'butt')
      .style('cursor', 'default')
      .on('mousemove', (event, d) => {
        tooltip.style('opacity', 1)
          .html(`<b>${d.color.toUpperCase()} flag</b><br>${d.start.toISOString().slice(0, 16).replace('T', ' ')} UTC onward`)
          .style('left', (event.pageX + 12) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseleave', () => tooltip.style('opacity', 0));
  }

  svg.append('g')
    .selectAll('line')
    .data(y.ticks(5))
    .join('line')
    .attr('class', 'grid-line')
    .attr('x1', margin.left).attr('x2', width - margin.right)
    .attr('y1', d => y(d)).attr('y2', d => y(d));

  // "Right now" vertical marker -- a static reference line so it's easy
  // to see at a glance which points are past/observed vs. future/
  // forecast, without having to read the axis labels' relative offsets.
  // Drawn early (in the background, before the data lines) so it never
  // visually competes with them.
  const nowDate = new Date();
  if (nowDate >= x.domain()[0] && nowDate <= x.domain()[1]) {
    svg.append('line')
      .attr('class', 'now-line')
      .attr('x1', x(nowDate)).attr('x2', x(nowDate))
      .attr('y1', margin.top).attr('y2', height - margin.bottom)
      .attr('stroke', 'var(--muted)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '2,2')
      .attr('opacity', 0.5);
    svg.append('text')
      .attr('x', x(nowDate) + 4).attr('y', margin.top + 10)
      .attr('fill', 'var(--muted)')
      .attr('font-size', '0.68rem')
      .text('now');
  }


  // X-axis tick format: previously "%m/%d %Hh" (e.g. "09/09 06h") was
  // ambiguous about which day/year and gave no sense of "how far from
  // now" at a glance -- replaced with an absolute timestamp plus a
  // relative offset from the current moment, e.g.
  // "2026-09-09 15:30 (-3h)" / "2026-09-11 09:00 (+41h)".
  function formatAxisTick(d) {
    const pad = n => String(n).padStart(2, '0');
    const abs = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
    const diffHours = Math.round((d.getTime() - Date.now()) / 3600000);
    const sign = diffHours >= 0 ? '+' : '';
    return `${abs} (${sign}${diffHours}h)`;
  }

  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(Math.min(8, (hours + pastHours) / 12)).tickFormat(formatAxisTick))
    .selectAll('text')
    .attr('transform', 'rotate(-45)')
    .style('text-anchor', 'end');

  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6));

  const line = d3.line()
    .x(d => x(d.valid_time_utc_date))
    .y(d => y(d.value))
    .defined(d => d.value != null);

  // Hover-to-highlight: when overlapping lines make it hard to tell which
  // model is which, hovering a line (or its legend swatch) bolds that
  // model's line/points and dims the rest instead of leaving everything
  // at equal visual weight.
  function setHighlight(hoveredModel) {
    svg.selectAll('.model-line')
      .attr('stroke-width', d => d === hoveredModel ? 4 : 2)
      .attr('opacity', d => !hoveredModel || d === hoveredModel ? 1 : 0.25);
    svg.selectAll('.model-dot')
      .attr('opacity', d => !hoveredModel || d.__model === hoveredModel ? 1 : 0.2);
  }

  models.forEach(m => {
    if (state.hiddenModels.has(m)) return; // toggled off via legend click
    const series = (byModel.get(m) || []).slice().sort((a, b) => a.valid_time_utc_date - b.valid_time_utc_date);
    svg.append('path')
      .datum(m)
      .attr('class', 'model-line')
      .attr('fill', 'none')
      .attr('stroke', MODEL_COLORS[m] || '#888')
      .attr('stroke-width', 2)
      .attr('d', () => line(series))
      .style('cursor', 'pointer')
      .on('mouseenter', () => setHighlight(m))
      .on('mouseleave', () => setHighlight(null));

    series.forEach(d => { d.__model = m; });
    svg.append('g')
      .selectAll('circle')
      .data(series)
      .join('circle')
      .attr('class', 'model-dot')
      .attr('cx', d => x(d.valid_time_utc_date))
      .attr('cy', d => y(d.value))
      .attr('r', 3)
      .attr('fill', MODEL_COLORS[m] || '#888')
      .style('cursor', 'default')
      .on('mouseenter', () => setHighlight(m))
      .on('mousemove', (event, d) => {
        tooltip.style('opacity', 1)
          .html(`<b>${m.toUpperCase()}</b><br>${d.valid_time_utc}<br>value: ${d.value}`)
          .style('left', (event.pageX + 12) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseleave', () => { tooltip.style('opacity', 0); setHighlight(null); });
  });

  // observation overlay (dashed grey line + dots), participates in the
  // same hover-to-highlight system as the model lines -- previously had
  // no interactive elements at all, so hovering it (or its legend entry)
  // did nothing.
  const OBS_KEY = 'observed';
  if (obs.length && !state.hiddenModels.has(OBS_KEY)) {
    const obsLine = d3.line()
      .x(d => x(d.ts_utc_date))
      .y(d => y(d.value))
      .defined(d => d.value != null);
    svg.append('path')
      .datum(OBS_KEY)
      .attr('class', 'model-line')
      .attr('fill', 'none')
      .attr('stroke', '#9aa5ab')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3')
      .attr('d', () => obsLine(obs))
      .style('cursor', 'pointer')
      .on('mouseenter', () => setHighlight(OBS_KEY))
      .on('mouseleave', () => setHighlight(null));

    obs.forEach(d => { d.__model = OBS_KEY; });
    svg.append('g')
      .selectAll('circle')
      .data(obs.filter(d => d.value != null))
      .join('circle')
      .attr('class', 'model-dot')
      .attr('cx', d => x(d.ts_utc_date))
      .attr('cy', d => y(d.value))
      .attr('r', 2.5)
      .attr('fill', '#9aa5ab')
      .style('cursor', 'pointer')
      .on('mouseenter', () => setHighlight(OBS_KEY))
      .on('mousemove', (event, d) => {
        tooltip.style('opacity', 1)
          .html(`<b>Observed</b><br>${d.ts_utc}<br>value: ${d.value}`)
          .style('left', (event.pageX + 12) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseleave', () => { tooltip.style('opacity', 0); setHighlight(null); });
  }

  // Legend: hover-to-highlight (existing behavior) PLUS click-to-toggle
  // visibility of that model's line/dots entirely. A toggled-off item
  // gets a dimmed/struck-through look so it's clear it's disabled, not
  // just unhighlighted.
  function toggleModel(key) {
    if (state.hiddenModels.has(key)) {
      state.hiddenModels.delete(key);
    } else {
      state.hiddenModels.add(key);
    }
    loadForecastChart(); // re-render with the updated visibility set
  }

  const legend = container.insert('div', 'svg').attr('class', 'legend');
  models.forEach(m => {
    const item = legend.append('div').attr('class', 'legend-item')
      .classed('legend-disabled', state.hiddenModels.has(m))
      .style('cursor', 'pointer')
      .on('mouseenter', () => setHighlight(m))
      .on('mouseleave', () => setHighlight(null))
      .on('click', () => toggleModel(m));
    item.append('span').attr('class', 'legend-swatch').style('background', MODEL_COLORS[m] || '#888');
    item.append('span').text(m.toUpperCase());
  });
  if (obs.length) {
    const item = legend.append('div').attr('class', 'legend-item')
      .classed('legend-disabled', state.hiddenModels.has(OBS_KEY))
      .style('cursor', 'pointer')
      .on('mouseenter', () => setHighlight(OBS_KEY))
      .on('mouseleave', () => setHighlight(null))
      .on('click', () => toggleModel(OBS_KEY));
    item.append('span').attr('class', 'legend-swatch').style('background', '#9aa5ab');
    item.append('span').text('Observed');
  }

  if (flagHistory.length) {
    const seen = new Set(flagHistory.map(d => d.flag_color));
    Array.from(seen).sort().forEach(color => {
      const item = legend.append('div').attr('class', 'legend-item');
      item.append('span').attr('class', 'legend-swatch')
        .style('background', FLAG_COLORS[color] || '#888');
      item.append('span').text(`${color.toUpperCase()} flag`);
    });
  }
}

// ---------------------------------------------------------------------------
// Forecast Stability chart: for the upcoming forecast window, how much has
// each model's OWN prediction for each hour changed across its last few
// runs? REPLACES the old Forecast Convergence chart entirely (user explicit
// direction 2026-09-14: "i want the convergence chart to be a day over day
// forecast stability instead" / "Replace the Forecast Convergence PANEL
// entirely... remove the old click-a-point/target-time picker UI").
// Convergence answered "how did predictions for ONE fixed target time
// evolve over the runs leading up to it"; this answers "right now, across
// the WHOLE upcoming forecast, which hours/models are still unstable
// run-to-run" -- backed by /api/forecast-stability, a brand new endpoint
// that fully replaced the old convergence endpoint (removed 2026-09-14).
// ---------------------------------------------------------------------------

// Same overlapping-request race guard pattern used elsewhere on this
// dashboard (see the convergence-button race-condition fix history) --
// changing "Compare last N runs" or "Forecast window" fires a new fetch
// before an older one may have resolved.
let stabilityRequestToken = 0;

async function loadStabilityChart() {
  const myToken = ++stabilityRequestToken;
  const numRuns = d3.select('#stability-num-runs-select').property('value');
  const hours = d3.select('#stability-hours-select').property('value');
  if (!state.location || !state.variable) return;

  const data = await fetchJSON(
    `/api/forecast-stability?location=${state.location}&variable=${state.variable}&num_runs=${numRuns}&hours=${hours}`
  );
  if (myToken !== stabilityRequestToken) return; // a newer request has since started/finished

  const container = d3.select('#stability-chart');
  container.selectAll('*').remove();
  const empty = d3.select('#stability-empty');

  if (data.error) {
    empty.attr('hidden', null).text(data.error);
    return;
  }
  if (!data.rows || !data.rows.length) {
    empty.attr('hidden', null).text(
      `Not enough forecast run history yet to compare the last ${numRuns} runs for this location/variable.`
    );
    return;
  }
  empty.attr('hidden', true);

  const rows = data.rows;
  rows.forEach(d => { d.valid_time_utc_date = new Date(d.valid_time_utc + 'Z'); });
  const models = Array.from(new Set(rows.map(d => d.model))).sort();
  const byModel = d3.group(rows, d => d.model);
  // Recompute the Y-axis domain from only VISIBLE models' rows (matches
  // the Forecast Time Series chart's behavior: hiding a model rescales
  // the chart to fit what's actually still shown, instead of leaving
  // dead space sized for a hidden series).
  const visibleRows = rows.filter(d => !state.hiddenStabilityModels.has(d.model));
  const yDomainRows = visibleRows.length ? visibleRows : rows;

  const width = Math.min(900, container.node().clientWidth || 900);
  const height = 320;
  const margin = { top: 20, right: 20, bottom: 70, left: 60 };

  const x = d3.scaleTime()
    .domain(d3.extent(rows, d => d.valid_time_utc_date))
    .range([margin.left, width - margin.right]);
  const y = d3.scaleLinear()
    .domain([d3.min(yDomainRows, d => d.min_value), d3.max(yDomainRows, d => d.max_value)])
    .nice()
    .range([height - margin.bottom, margin.top]);

  const svg = container.append('svg').attr('width', width).attr('height', height);

  svg.append('g')
    .selectAll('line')
    .data(y.ticks(6))
    .join('line')
    .attr('class', 'grid-line')
    .attr('x1', margin.left).attr('x2', width - margin.right)
    .attr('y1', d => y(d)).attr('y2', d => y(d));

  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(Math.min(10, rows.length)))
    .selectAll('text')
    .attr('transform', 'rotate(-35)')
    .style('text-anchor', 'end');

  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6));

  svg.append('text')
    .attr('x', -height / 2).attr('y', 14)
    .attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle')
    .attr('fill', 'var(--muted)')
    .attr('font-size', '0.75rem')
    .text(fmtVarLabel(state.variable));

  svg.append('text')
    .attr('x', width / 2).attr('y', height - 6)
    .attr('text-anchor', 'middle')
    .attr('fill', 'var(--muted)').attr('font-size', '0.72rem')
    .text(`Upcoming forecast hours -- shaded band = spread across each model's last ${numRuns} runs`);

  const tooltip = getOrCreateTooltip('point-tooltip');

  function setStabilityHighlight(hoveredModel) {
    svg.selectAll('.stability-band')
      .attr('opacity', d => !hoveredModel || d === hoveredModel ? 0.28 : 0.08);
    svg.selectAll('.stability-line')
      .attr('stroke-width', d => d === hoveredModel ? 3 : 1.75)
      .attr('opacity', d => !hoveredModel || d === hoveredModel ? 1 : 0.25);
  }

  const areaGen = d3.area()
    .x(d => x(d.valid_time_utc_date))
    .y0(d => y(d.min_value))
    .y1(d => y(d.max_value));
  const lineGen = d3.line()
    .x(d => x(d.valid_time_utc_date))
    .y(d => y(d.avg_value));

  models.forEach(m => {
    if (state.hiddenStabilityModels.has(m)) return; // toggled off via legend click
    const series = (byModel.get(m) || []).slice().sort((a, b) => a.valid_time_utc_date - b.valid_time_utc_date);
    const color = MODEL_COLORS[m] || '#888';

    // shaded min-max spread band -- the actual "stability" signal: wide =
    // this model's own predictions for that hour disagreed across its
    // last few runs, tight = it's been consistent.
    svg.append('path')
      .datum(m)
      .attr('class', 'stability-band')
      .attr('fill', color)
      .attr('opacity', 0.28)
      .attr('d', () => areaGen(series))
      .style('pointer-events', 'none');

    // average-value line on top, for a clean reference of "where the
    // model currently sits" independent of the band's width.
    svg.append('path')
      .datum(m)
      .attr('class', 'stability-line')
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 1.75)
      .attr('d', () => lineGen(series))
      .style('cursor', 'pointer')
      .on('mouseenter', () => setStabilityHighlight(m))
      .on('mouseleave', () => setStabilityHighlight(null));

    svg.append('g')
      .selectAll('circle')
      .data(series)
      .join('circle')
      .attr('cx', d => x(d.valid_time_utc_date))
      .attr('cy', d => y(d.avg_value))
      .attr('r', 3)
      .attr('fill', color)
      .style('cursor', 'pointer')
      .on('mouseenter', () => setStabilityHighlight(m))
      .on('mousemove', (event, d) => {
        tooltip.style('opacity', 1)
          .html(`<b>${m.toUpperCase()}</b><br>${d.valid_time_utc}<br>` +
                `avg: ${d.avg_value} ${fmtVarLabel(state.variable)}<br>` +
                `range across last ${numRuns} runs: ${d.min_value} - ${d.max_value} (spread ${d.spread})<br>` +
                `stddev: ${d.stddev_value}`)
          .style('left', (event.pageX + 12) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseleave', () => { tooltip.style('opacity', 0); setStabilityHighlight(null); });
  });

  // Legend: hover-to-highlight (existing behavior) PLUS click-to-toggle
  // visibility, mirroring the Forecast Time Series chart's legend
  // (user 2026-09-21: "hover to brighten works well, but only one
  // chart has the click to temporarily hide").
  function toggleStabilityModel(key) {
    if (state.hiddenStabilityModels.has(key)) {
      state.hiddenStabilityModels.delete(key);
    } else {
      state.hiddenStabilityModels.add(key);
    }
    loadStabilityChart(); // re-render with the updated visibility set
  }

  const legend = container.insert('div', 'svg').attr('class', 'legend');
  models.forEach(m => {
    const item = legend.append('div').attr('class', 'legend-item')
      .classed('legend-disabled', state.hiddenStabilityModels.has(m))
      .style('cursor', 'pointer')
      .on('mouseenter', () => setStabilityHighlight(m))
      .on('mouseleave', () => setStabilityHighlight(null))
      .on('click', () => toggleStabilityModel(m));
    item.append('span').attr('class', 'legend-swatch').style('background', MODEL_COLORS[m] || '#888');
    const runsForModel = (data.runs_used_by_model && data.runs_used_by_model[m]) || [];
    item.append('span').text(`${m.toUpperCase()}${runsForModel.length ? ` (${runsForModel.length} runs)` : ''}`);
  });
}

// ---------------------------------------------------------------------------
// CBI Flag Prediction panel: Bayesian P(flag color | forecast wind) from one
// selected model. SEPARATE from the Forecast/Accuracy/Stability charts
// above -- user explicit direction 2026-09-11: "do not ruin or mess up the
// existing charts... I want additional overlay or a separate chart". This
// panel answers "what flag color is likely at CBI at some future hour",
// NOT a wind forecast itself -- backed by /api/flag-prediction, which pairs
// the selected model's forecast wind with CBI's own historical wind-vs-flag
// distribution (Bayesian posterior with Dirichlet smoothing, since the
// training set of ~40 flag readings is small).
// ---------------------------------------------------------------------------

async function loadFlagPredictionChart() {
  const model = d3.select('#flag-prediction-model-select').property('value');
  const hours = +d3.select('#flag-prediction-hours-select').property('value');
  if (!model) return;

  const data = await fetchJSON(`/api/flag-prediction?model=${model}&hours=${hours}`);
  const container = d3.select('#flag-prediction-chart');
  container.selectAll('*').remove();
  const empty = d3.select('#flag-prediction-empty');

  if (data.error) {
    empty.attr('hidden', null).text(data.error);
    return;
  }
  if (!data.predictions || !data.predictions.length) {
    empty.attr('hidden', null).text(`No forecast data yet for ${model.toUpperCase()} at CBI.`);
    return;
  }
  empty.attr('hidden', true);

  const predictions = data.predictions.filter(p => p.flag_probabilities);
  predictions.forEach(p => { p.valid_time_utc_date = new Date(p.valid_time_utc + 'Z'); });

  const width = Math.min(900, container.node().clientWidth || 900);
  const height = 320;
  const margin = { top: 20, right: 20, bottom: 70, left: 55 };

  const x = d3.scaleBand()
    .domain(predictions.map(p => p.valid_time_utc))
    .range([margin.left, width - margin.right])
    .padding(0.15);
  const y = d3.scaleLinear().domain([0, 1]).range([height - margin.bottom, margin.top]);
  const stackOrder = ['closed', 'red', 'yellow', 'green']; // most-restrictive at bottom
  const stacked = d3.stack().keys(stackOrder).value((p, key) => p.flag_probabilities[key])(predictions);

  const svg = container.append('svg').attr('width', width).attr('height', height);

  svg.append('g')
    .selectAll('line')
    .data(y.ticks(5))
    .join('line')
    .attr('class', 'grid-line')
    .attr('x1', margin.left).attr('x2', width - margin.right)
    .attr('y1', d => y(d)).attr('y2', d => y(d));

  const tooltip = getOrCreateTooltip('bar-tooltip');

  svg.append('g')
    .selectAll('g')
    .data(stacked)
    .join('g')
    .attr('fill', d => FLAG_COLORS[d.key] || '#888')
    .selectAll('rect')
    .data(d => d.map(seg => ({ ...seg, key: d.key })))
    .join('rect')
    .attr('x', (d, i) => x(predictions[i].valid_time_utc))
    .attr('width', x.bandwidth())
    .attr('y', d => y(d[1]))
    .attr('height', d => y(d[0]) - y(d[1]))
    .on('mousemove', (event, d) => {
      const p = predictions.find(pr => pr.valid_time_utc === d.data.valid_time_utc);
      tooltip.style('opacity', 1)
        .html(`<b>${d.key.toUpperCase()}</b>: ${(p.flag_probabilities[d.key] * 100).toFixed(1)}%<br>` +
              `${p.valid_time_utc.replace('T', ' ')} UTC<br>` +
              `forecast wind: ${p.wind_kt != null ? p.wind_kt.toFixed(1) + ' kt' : 'n/a'}, gust: ${p.gust_kt != null ? p.gust_kt.toFixed(1) + ' kt' : 'n/a'} (effective bucket: ${p.wind_bucket || 'n/a'})<br>` +
              `<i>most likely: ${p.most_likely_flag ? p.most_likely_flag.toUpperCase() : 'n/a'}</i>`)
        .style('left', (event.pageX + 12) + 'px')
        .style('top', (event.pageY - 10) + 'px');
    })
    .on('mouseleave', () => tooltip.style('opacity', 0));

  const tickEvery = Math.max(1, Math.ceil(predictions.length / 10));
  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).tickValues(x.domain().filter((d, i) => i % tickEvery === 0))
      .tickFormat(d => new Date(d + 'Z').toISOString().slice(5, 16).replace('T', ' ')))
    .selectAll('text')
    .attr('transform', 'rotate(-35)')
    .style('text-anchor', 'end');

  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.0%')));

  svg.append('text')
    .attr('x', -height / 2).attr('y', 16)
    .attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle')
    .attr('fill', 'var(--muted)')
    .attr('font-size', '0.75rem')
    .text('P(flag color)');

  const legend = container.insert('div', 'svg').attr('class', 'legend');
  stackOrder.forEach(c => {
    const item = legend.append('div').attr('class', 'legend-item');
    item.append('span').attr('class', 'legend-swatch').style('background', FLAG_COLORS[c] || '#888');
    item.append('span').text(c.toUpperCase());
  });
  legend.append('div').attr('class', 'legend-item')
    .attr('style', 'color:var(--muted); font-size:0.75rem;')
    .text(`(trained on ${data.training_sample_size} historical flag readings)`);
}

// ---------------------------------------------------------------------------
// Wind Prediction panel (all locations EXCEPT CBI -- CBI has no wind sensor
// of its own, hence the separate Flag Prediction panel above): Bayesian
// P(actual wind bucket | forecast wind), from one selected model. Mirrors
// the Flag Prediction panel's stacked-probability-by-hour design, per user
// direction 2026-09-11 ("show a very similar graph, but do wind prediction
// instead"), backed by /api/wind-prediction.
// ---------------------------------------------------------------------------

function windBucketColorScale(buckets) {
  // Ordered numeric buckets get a sequential color scale (unlike flag
  // colors, which are fixed categorical values) -- low wind = cool color,
  // high wind = warm color, consistent regardless of how many buckets a
  // given location's training data happens to produce.
  const n = Math.max(1, buckets.length - 1);
  const scale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, n]);
  const map = {};
  buckets.forEach((b, i) => { map[b] = scale(i); });
  return map;
}

async function loadWindPredictionChart() {
  const model = d3.select('#wind-prediction-model-select').property('value');
  const hours = +d3.select('#wind-prediction-hours-select').property('value');
  if (!model || !state.location) return;

  const data = await fetchJSON(`/api/wind-prediction?location=${state.location}&model=${model}&hours=${hours}`);
  const container = d3.select('#wind-prediction-chart');
  container.selectAll('*').remove();
  const empty = d3.select('#wind-prediction-empty');

  if (data.error) {
    empty.attr('hidden', null).text(data.error);
    return;
  }
  if (!data.predictions || !data.predictions.length || !data.actual_wind_buckets || !data.actual_wind_buckets.length) {
    empty.attr('hidden', null).text(`No forecast/training data yet for ${model.toUpperCase()} at this location.`);
    return;
  }
  empty.attr('hidden', true);

  const predictions = data.predictions.filter(p => p.actual_wind_probabilities);
  if (!predictions.length) {
    empty.attr('hidden', null).text(`No forecast data yet for ${model.toUpperCase()} at this location.`);
    return;
  }

  const buckets = data.actual_wind_buckets; // already sorted low->high by the backend
  const bucketColors = windBucketColorScale(buckets);

  const width = Math.min(900, container.node().clientWidth || 900);
  const height = 320;
  const margin = { top: 20, right: 20, bottom: 70, left: 55 };

  const x = d3.scaleBand()
    .domain(predictions.map(p => p.valid_time_utc))
    .range([margin.left, width - margin.right])
    .padding(0.15);
  const y = d3.scaleLinear().domain([0, 1]).range([height - margin.bottom, margin.top]);
  const stacked = d3.stack().keys(buckets).value((p, key) => p.actual_wind_probabilities[key] || 0)(predictions);

  const svg = container.append('svg').attr('width', width).attr('height', height);

  svg.append('g')
    .selectAll('line')
    .data(y.ticks(5))
    .join('line')
    .attr('class', 'grid-line')
    .attr('x1', margin.left).attr('x2', width - margin.right)
    .attr('y1', d => y(d)).attr('y2', d => y(d));

  const tooltip = getOrCreateTooltip('bar-tooltip');

  svg.append('g')
    .selectAll('g')
    .data(stacked)
    .join('g')
    .attr('fill', d => bucketColors[d.key] || '#888')
    .selectAll('rect')
    .data(d => d.map(seg => ({ ...seg, key: d.key })))
    .join('rect')
    .attr('x', (d, i) => x(predictions[i].valid_time_utc))
    .attr('width', x.bandwidth())
    .attr('y', d => y(d[1]))
    .attr('height', d => y(d[0]) - y(d[1]))
    .on('mousemove', (event, d) => {
      const p = predictions.find(pr => pr.valid_time_utc === d.data.valid_time_utc);
      const prob = p.actual_wind_probabilities[d.key] || 0;
      tooltip.style('opacity', 1)
        .html(`<b>${d.key}</b>: ${(prob * 100).toFixed(1)}%<br>` +
              `${p.valid_time_utc.replace('T', ' ')} UTC<br>` +
              `model forecast: ${p.forecast_wind_kt != null ? p.forecast_wind_kt.toFixed(1) + ' kt' : 'n/a'} (${p.forecast_wind_bucket || 'n/a'})<br>` +
              `<i>most likely actual: ${p.most_likely_actual_bucket || 'n/a'}</i>`)
        .style('left', (event.pageX + 12) + 'px')
        .style('top', (event.pageY - 10) + 'px');
    })
    .on('mouseleave', () => tooltip.style('opacity', 0));

  const tickEvery = Math.max(1, Math.ceil(predictions.length / 10));
  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).tickValues(x.domain().filter((d, i) => i % tickEvery === 0))
      .tickFormat(d => new Date(d + 'Z').toISOString().slice(5, 16).replace('T', ' ')))
    .selectAll('text')
    .attr('transform', 'rotate(-35)')
    .style('text-anchor', 'end');

  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.0%')));

  svg.append('text')
    .attr('x', -height / 2).attr('y', 16)
    .attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle')
    .attr('fill', 'var(--muted)')
    .attr('font-size', '0.75rem')
    .text('P(actual wind bucket)');

  const legend = container.insert('div', 'svg').attr('class', 'legend');
  buckets.forEach(b => {
    const item = legend.append('div').attr('class', 'legend-item');
    item.append('span').attr('class', 'legend-swatch').style('background', bucketColors[b] || '#888');
    item.append('span').text(b);
  });
  legend.append('div').attr('class', 'legend-item')
    .attr('style', 'color:var(--muted); font-size:0.75rem;')
    .text(`(trained on ${data.training_sample_size} historical forecast-vs-observed pairs)`);
}

// ---------------------------------------------------------------------------
// Wind Rose panel: direction + speed frequency, observed vs. (optional)
// forecast, for locations with real wind sensors. New chart category --
// unlike everything else on this dashboard, this uses a polar/radial
// layout instead of a time-series x-axis, since direction is inherently
// circular. Renders TWO side-by-side radial charts when a comparison
// model is selected (one arc-length ring per direction sector, color per
// speed bin, stacked radially like a proper wind rose).
// ---------------------------------------------------------------------------

function drawWindRose(svg, cx, cy, radius, rows, speedBins, title) {
  const bySector = d3.group(rows, d => d.sector);
  const maxTotal = d3.max(Array.from(bySector.values()), sectorRows => d3.sum(sectorRows, r => r.frequency)) || 0.0001;
  const rScale = d3.scaleLinear().domain([0, maxTotal]).range([0, radius]);
  const speedColor = d3.scaleOrdinal().domain(speedBins).range(d3.quantize(d3.interpolateYlOrRd, speedBins.length + 1).slice(1));
  const angleStep = (2 * Math.PI) / 16;

  const tooltip = getOrCreateTooltip('point-tooltip');

  const g = svg.append('g').attr('transform', `translate(${cx},${cy})`);

  // gridlines: concentric circles at 25/50/75/100% of the max sector total
  [0.25, 0.5, 0.75, 1.0].forEach(frac => {
    g.append('circle')
      .attr('r', rScale(maxTotal * frac))
      .attr('fill', 'none')
      .attr('class', 'grid-line');
    g.append('text')
      .attr('x', 4).attr('y', -rScale(maxTotal * frac) - 2)
      .attr('fill', 'var(--muted)').attr('font-size', '0.6rem')
      .text(d3.format('.0%')(frac * maxTotal));
  });

  // compass labels (N/E/S/W)
  const compassLabels = [[0, 'N'], [4, 'E'], [8, 'S'], [12, 'W']];
  compassLabels.forEach(([sector, label]) => {
    const angle = sector * angleStep - Math.PI / 2;
    g.append('text')
      .attr('x', (radius + 14) * Math.cos(angle))
      .attr('y', (radius + 14) * Math.sin(angle))
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
      .attr('fill', 'var(--text)').attr('font-size', '0.75rem').attr('font-weight', 'bold')
      .text(label);
  });

  for (const [sector, sectorRows] of bySector.entries()) {
    let cumFreq = 0;
    const angle0 = sector * angleStep - Math.PI / 2 - angleStep / 2 + angleStep * 0.08;
    const angle1 = sector * angleStep - Math.PI / 2 + angleStep / 2 - angleStep * 0.08;
    const arcGen = d3.arc().innerRadius(d => rScale(d.r0)).outerRadius(d => rScale(d.r1))
      .startAngle(angle0 + Math.PI / 2).endAngle(angle1 + Math.PI / 2);
    speedBins.forEach(bin => {
      const row = sectorRows.find(r => r.speed_bin === bin);
      const freq = row ? row.frequency : 0;
      if (freq <= 0) return;
      const d = { r0: cumFreq, r1: cumFreq + freq };
      cumFreq += freq;
      g.append('path')
        .attr('d', arcGen(d))
        .attr('fill', speedColor(bin))
        .on('mousemove', (event) => {
          tooltip.style('opacity', 1)
            .html(`<b>${bin}</b><br>freq: ${(freq * 100).toFixed(1)}%<br>count: ${row.count}`)
            .style('left', (event.pageX + 12) + 'px')
            .style('top', (event.pageY - 10) + 'px');
        })
        .on('mouseleave', () => tooltip.style('opacity', 0));
    });
  }

  svg.append('text')
    .attr('x', cx).attr('y', cy - radius - 30)
    .attr('text-anchor', 'middle')
    .attr('fill', 'var(--text)').attr('font-size', '0.85rem').attr('font-weight', 'bold')
    .text(title);

  return speedColor;
}

async function loadWindRoseChart() {
  const model = d3.select('#wind-rose-model-select').property('value');
  const hours = +d3.select('#wind-rose-hours-select').property('value');
  if (!state.location) return;

  const url = `/api/wind-rose?location=${state.location}&hours=${hours}` + (model ? `&model=${model}` : '');
  const data = await fetchJSON(url);
  const container = d3.select('#wind-rose-chart');
  container.selectAll('*').remove();
  const empty = d3.select('#wind-rose-empty');

  if (data.error) {
    empty.attr('hidden', null).text(data.error);
    return;
  }
  if (!data.observed || !data.observed.total_samples) {
    empty.attr('hidden', null).text('No observed wind direction/speed data yet for this location.');
    return;
  }
  empty.attr('hidden', true);

  const showComparison = !!(data.forecast && data.forecast.total_samples);
  const width = Math.min(900, container.node().clientWidth || 900);
  const height = 420;
  const radius = 130;
  const svg = container.append('svg').attr('width', width).attr('height', height);

  let speedColor;
  if (showComparison) {
    speedColor = drawWindRose(svg, width * 0.27, height / 2 + 20, radius, data.observed.rows, data.speed_bins,
      `Observed (n=${data.observed.total_samples})`);
    drawWindRose(svg, width * 0.73, height / 2 + 20, radius, data.forecast.rows, data.speed_bins,
      `${model.toUpperCase()} Forecast (n=${data.forecast.total_samples})`);
  } else {
    speedColor = drawWindRose(svg, width / 2, height / 2 + 20, radius, data.observed.rows, data.speed_bins,
      `Observed (n=${data.observed.total_samples})`);
  }

  const legend = container.insert('div', 'svg').attr('class', 'legend');
  data.speed_bins.forEach(b => {
    const item = legend.append('div').attr('class', 'legend-item');
    item.append('span').attr('class', 'legend-swatch').style('background', speedColor(b));
    item.append('span').text(b);
  });
}

// ---------------------------------------------------------------------------
// Gust Factor panel: gust_kt / sustained_kt time series, for locations with
// real gust sensor data. A ratio near 1.0 = steady wind; higher = gusty and
// potentially more hazardous than steady wind of the same average speed.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Gustiness panel: primary chart is the absolute gust delta (gust minus
// sustained, in kt) over time, colored by gust/sustained ratio; secondary
// is a sustained-vs-gust scatter, same color scheme. REDESIGNED 2026-09-19
// per user: "let's talk about some better graphs to represent the
// gustiness factor because 10x gust is different if wind is 10 knots vs 1
// knot base." The old chart plotted the RATIO alone, which is misleading
// at low wind (dividing by a small base wind inflates the ratio even for
// a mild absolute gust) -- see api_gust_factor's docstring in app.py for
// the empirical numbers that motivated this. Delta is now the primary
// Y-axis/line metric because "+5kt of gust" means the same real thing
// regardless of the base wind, unlike a ratio -- but per user follow-up
// ("I like colors on the gust factor, do not null it at all at low wind"
// then "grey out 3knots and below... add similar color scale... to 1d
// chart"), the ratio is still shown as color on both charts, with points
// at GUST_COLOR_GREY_BELOW_KT sustained wind or below rendered a flat
// grey instead of colored -- not because the ratio is hidden/unavailable
// (the backend always computes it now), but because a few near-zero-wind
// points have such extreme ratios (up to ~20x) that including them in the
// color domain washed out all the real variation in the rest of the plot.
// ---------------------------------------------------------------------------

const GUST_COLOR_GREY_BELOW_KT = 1.5;

function gustColorScaleFor(data) {
  const colorable = data.filter(d => d.gust_factor != null && d.sustained_kt > GUST_COLOR_GREY_BELOW_KT);
  const colorDomain = colorable.length ? d3.extent(colorable, d => d.gust_factor) : [1, 2];
  return d3.scaleSequential(d3.interpolateYlOrRd).domain(colorDomain);
}

function gustColorFor(d, color) {
  if (d.gust_factor == null || d.sustained_kt <= GUST_COLOR_GREY_BELOW_KT) return 'var(--muted)';
  return color(d.gust_factor);
}

async function loadGustFactorChart() {
  const hours = +d3.select('#gust-factor-hours-select').property('value');
  const forecastModel = d3.select('#gust-factor-model-select').property('value');
  if (!state.location) return;

  const url = forecastModel
    ? `/api/gust-factor?location=${state.location}&hours=${hours}&forecast_model=${forecastModel}&forecast_hours=24`
    : `/api/gust-factor?location=${state.location}&hours=${hours}`;
  const resp = await fetchJSON(url);
  const container = d3.select('#gust-factor-chart');
  container.selectAll('*').remove();
  const empty = d3.select('#gust-factor-empty');

  if (resp.error) {
    empty.attr('hidden', null).text(resp.error);
    return;
  }
  const observed = resp.observed || [];
  const forecast = resp.forecast || [];
  if (!observed.length && !forecast.length) {
    empty.attr('hidden', null).text('No gust data yet for this location/window.');
    return;
  }
  empty.attr('hidden', true);

  observed.forEach(d => { d.ts_utc_date = new Date(d.ts_utc + 'Z'); d.is_forecast = false; });
  forecast.forEach(d => { d.ts_utc_date = new Date(d.ts_utc + 'Z'); d.is_forecast = true; });
  const allPoints = observed.concat(forecast);
  const now = new Date();

  const width = Math.min(900, container.node().clientWidth || 900);
  const height = 280;
  const margin = { top: 20, right: 70, bottom: 60, left: 55 };

  const x = d3.scaleTime().domain(d3.extent(allPoints, d => d.ts_utc_date)).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear()
    .domain([0, Math.max(2, d3.max(allPoints, d => d.gust_delta_kt) * 1.1)])
    .nice()
    .range([height - margin.bottom, margin.top]);
  const color = gustColorScaleFor(allPoints);

  const svg = container.append('svg').attr('width', width).attr('height', height);

  svg.append('g')
    .selectAll('line')
    .data(y.ticks(5))
    .join('line')
    .attr('class', 'grid-line')
    .attr('x1', margin.left).attr('x2', width - margin.right)
    .attr('y1', d => y(d)).attr('y2', d => y(d));

  // "now" divider -- everything left of this is real observed history,
  // everything right is the selected model's own forecast (user
  // 2026-09-19: "some indicator in the chart, a different color and/or
  // dotted line for today which data is predicted from selected model").
  if (forecast.length) {
    const nowX = x(now);
    svg.append('line')
      .attr('x1', nowX).attr('x2', nowX)
      .attr('y1', margin.top).attr('y2', height - margin.bottom)
      .attr('stroke', 'var(--accent, #4fc3f7)').attr('stroke-width', 1.5).attr('stroke-dasharray', '2,2').attr('opacity', 0.7);
    svg.append('text')
      .attr('x', nowX + 4).attr('y', margin.top + 10)
      .attr('fill', 'var(--accent, #4fc3f7)').attr('font-size', '0.65rem')
      .text('now \u2192 forecast (' + forecastModel.toUpperCase() + ')');
  }

  // Connecting line stays a single neutral color (a multi-color line is
  // hard to read); the colored DOTS carry the ratio signal instead.
  // Observed and forecast are drawn as two SEPARATE paths (not one
  // continuous line) with different dash patterns, so the observed/
  // forecast boundary is visually unambiguous even before the "now" line.
  const line = d3.line().x(d => x(d.ts_utc_date)).y(d => y(d.gust_delta_kt));
  if (observed.length) {
    svg.append('path')
      .datum(observed)
      .attr('fill', 'none')
      .attr('stroke', 'var(--muted)')
      .attr('stroke-width', 1)
      .attr('opacity', 0.5)
      .attr('d', line);
  }
  if (forecast.length) {
    // connect the last observed point to the first forecast point so
    // there's no visual gap at the "now" boundary
    const bridge = observed.length ? [observed[observed.length - 1], ...forecast] : forecast;
    svg.append('path')
      .datum(bridge)
      .attr('fill', 'none')
      .attr('stroke', 'var(--muted)')
      .attr('stroke-width', 1.3)
      .attr('stroke-dasharray', '5,3')
      .attr('opacity', 0.7)
      .attr('d', line);
  }

  const tooltip = getOrCreateTooltip('point-tooltip');
  svg.selectAll('circle.gust-dot')
    .data(allPoints)
    .join('circle')
    .attr('class', 'gust-dot')
    .attr('cx', d => x(d.ts_utc_date))
    .attr('cy', d => y(d.gust_delta_kt))
    .attr('r', d => d.is_forecast ? 3.2 : 2.8)
    .attr('fill', d => d.is_forecast ? 'none' : gustColorFor(d, color))
    .attr('stroke', d => d.is_forecast ? gustColorFor(d, color) : 'none')
    .attr('stroke-width', d => d.is_forecast ? 1.6 : 0)
    .on('mousemove', (event, d) => {
      const factorText = d.gust_factor != null ? `${d.gust_factor}x` : 'n/a (0kt sustained)';
      const label = d.is_forecast ? `<b>+${d.gust_delta_kt}kt (${forecastModel.toUpperCase()} forecast)</b>` : `<b>+${d.gust_delta_kt}kt</b>`;
      tooltip.style('opacity', 1)
        .html(`${label}<br>${d.ts_utc}<br>sustained: ${d.sustained_kt}kt, gust: ${d.gust_kt}kt<br>ratio: ${factorText}`)
        .style('left', (event.pageX + 12) + 'px')
        .style('top', (event.pageY - 10) + 'px');
    })
    .on('mouseleave', () => tooltip.style('opacity', 0));

  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(6))
    .selectAll('text')
    .attr('transform', 'rotate(-25)')
    .style('text-anchor', 'end');

  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(5).tickFormat(d => d + 'kt'));

  svg.append('text')
    .attr('x', -height / 2).attr('y', 16)
    .attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle')
    .attr('fill', 'var(--muted)')
    .attr('font-size', '0.75rem')
    .text('Gust delta (gust - sustained, kt)');

  drawGustColorLegend(svg, color, width, height, margin);

  loadGustScatterChart(allPoints, forecastModel);
}

// Shared color-legend renderer for both the 1D and 2D gustiness charts --
// factored out so the two stay visually consistent (same gradient, same
// "grey = <=3kt" caption) rather than drifting apart over time.
function drawGustColorLegend(svg, color, width, height, margin) {
  const legendHeight = Math.min(140, height - margin.top - margin.bottom);
  const legendX = width - margin.right + 20;
  const legendScale = d3.scaleLinear().domain(color.domain()).range([legendHeight, 0]);
  const gradientId = `gust-ratio-gradient-${Math.random().toString(36).slice(2)}`;
  const defs = svg.append('defs');
  const gradient = defs.append('linearGradient')
    .attr('id', gradientId).attr('x1', '0%').attr('x2', '0%').attr('y1', '100%').attr('y2', '0%');
  const stops = 10;
  const [lo, hi] = color.domain();
  for (let i = 0; i <= stops; i++) {
    const t = i / stops;
    gradient.append('stop').attr('offset', `${t * 100}%`).attr('stop-color', color(lo + t * (hi - lo)));
  }
  svg.append('rect')
    .attr('x', legendX).attr('y', margin.top).attr('width', 14).attr('height', legendHeight)
    .attr('fill', `url(#${gradientId})`);
  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(${legendX + 14},${margin.top})`)
    .call(d3.axisRight(legendScale).ticks(4).tickFormat(d => d.toFixed(1) + 'x'));
  svg.append('text')
    .attr('x', legendX + 7).attr('y', margin.top - 6)
    .attr('text-anchor', 'middle').attr('fill', 'var(--muted)').attr('font-size', '0.65rem')
    .text('ratio');
  svg.append('rect')
    .attr('x', legendX).attr('y', margin.top + legendHeight + 10).attr('width', 10).attr('height', 10)
    .attr('fill', 'var(--muted)');
  svg.append('text')
    .attr('x', legendX + 14).attr('y', margin.top + legendHeight + 19)
    .attr('fill', 'var(--muted)').attr('font-size', '0.62rem')
    .text(`<=${GUST_COLOR_GREY_BELOW_KT}kt`);
}

function loadGustScatterChart(data, forecastModel) {
  const container = d3.select('#gust-scatter-chart');
  container.selectAll('*').remove();
  const empty = d3.select('#gust-scatter-empty');

  if (!data || !data.length) {
    empty.attr('hidden', null).text('No gust data yet for this location/window.');
    return;
  }
  empty.attr('hidden', true);

  const width = Math.min(900, container.node().clientWidth || 900);
  const height = 340;
  const margin = { top: 20, right: 70, bottom: 55, left: 55 };

  const maxAxis = Math.max(
    d3.max(data, d => d.sustained_kt),
    d3.max(data, d => d.gust_kt)
  ) * 1.05;

  const x = d3.scaleLinear().domain([0, maxAxis]).range([margin.left, width - margin.right]).nice();
  const y = d3.scaleLinear().domain([0, maxAxis]).range([height - margin.bottom, margin.top]).nice();
  const color = gustColorScaleFor(data);

  const svg = container.append('svg').attr('width', width).attr('height', height);

  svg.append('g')
    .selectAll('line.grid-x')
    .data(x.ticks(6))
    .join('line')
    .attr('class', 'grid-line')
    .attr('y1', margin.top).attr('y2', height - margin.bottom)
    .attr('x1', d => x(d)).attr('x2', d => x(d));
  svg.append('g')
    .selectAll('line.grid-y')
    .data(y.ticks(6))
    .join('line')
    .attr('class', 'grid-line')
    .attr('x1', margin.left).attr('x2', width - margin.right)
    .attr('y1', d => y(d)).attr('y2', d => y(d));

  // y=x reference line: points ON this line mean gust == sustained (no
  // gusting at all); points further above it mean progressively gustier
  // conditions, in absolute-knots terms -- this is the visual equivalent
  // of the primary chart's delta metric, but shown against the full
  // wind-speed range at once instead of as a time series.
  svg.append('line')
    .attr('x1', x(0)).attr('y1', y(0))
    .attr('x2', x(maxAxis)).attr('y2', y(maxAxis))
    .attr('stroke', 'var(--muted)').attr('stroke-dasharray', '4,3').attr('opacity', 0.6);
  svg.append('text')
    .attr('x', x(maxAxis) - 4).attr('y', y(maxAxis) - 6)
    .attr('text-anchor', 'end').attr('fill', 'var(--muted)').attr('font-size', '0.65rem')
    .text('gust = sustained (steady)');

  const tooltip = getOrCreateTooltip('point-tooltip');
  svg.selectAll('circle.scatter-dot')
    .data(data)
    .join('circle')
    .attr('class', 'scatter-dot')
    .attr('cx', d => x(d.sustained_kt))
    .attr('cy', d => y(d.gust_kt))
    .attr('r', d => d.is_forecast ? 4 : 3)
    .attr('fill', d => d.is_forecast ? 'none' : gustColorFor(d, color))
    .attr('stroke', d => d.is_forecast ? gustColorFor(d, color) : 'none')
    .attr('stroke-width', d => d.is_forecast ? 1.8 : 0)
    .attr('opacity', 0.8)
    .on('mousemove', (event, d) => {
      const factorText = d.gust_factor != null ? `${d.gust_factor}x` : 'n/a (0kt sustained)';
      const label = d.is_forecast ? `<b>${forecastModel.toUpperCase()} forecast</b><br>` : '';
      tooltip.style('opacity', 1)
        .html(`${label}sustained: ${d.sustained_kt}kt, gust: ${d.gust_kt}kt<br>+${d.gust_delta_kt}kt, ratio: ${factorText}<br>${d.ts_utc}`)
        .style('left', (event.pageX + 12) + 'px')
        .style('top', (event.pageY - 10) + 'px');
    })
    .on('mouseleave', () => tooltip.style('opacity', 0));

  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(6).tickFormat(d => d + 'kt'));
  svg.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6).tickFormat(d => d + 'kt'));

  svg.append('text')
    .attr('x', width / 2).attr('y', height - 6)
    .attr('text-anchor', 'middle')
    .attr('fill', 'var(--muted)').attr('font-size', '0.72rem')
    .text('Sustained wind (kt)');
  svg.append('text')
    .attr('x', -height / 2).attr('y', 16)
    .attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle')
    .attr('fill', 'var(--muted)')
    .attr('font-size', '0.75rem')
    .text('Gust (kt)');

  drawGustColorLegend(svg, color, width, height, margin);
}

init().catch(err => {
  console.error(err);
  document.body.insertAdjacentHTML('afterbegin',
    `<div style="background:#c62828;color:#fff;padding:10px;">Failed to load dashboard: ${err.message}</div>`);
});
