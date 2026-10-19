#!/usr/bin/env python3
"""CBI Community Boating operating-hours helper.

UPDATED 2026-10-19 (user: "when the season ends November 1st only the
cbi flag tracker should stop collecting and displaying data everything
else keeps going" -> clarified: "it would be better if you
automatically tracked closures on the site and didn't use closed times
in calculations"). Previously this hardcoded a fixed 9am-sunset daily
window year-round with NO season concept -- wrong the moment CBI's
real season actually ends. Now reads scripts/scrape_cbi_hours.py's
cached scrape of CBI's own published "Hours and Directions" page
(db/cbi_hours_cache.json) -- the real source of truth for both the
season's start/end dates AND the varying weekday open time across
sub-ranges within season (1pm/3pm/1pm as of the 2026 season). That
scrape is refreshed MONTHLY via cron (this schedule changes at most a
few times a year, nowhere near daily), not on every call here --
this module just reads whatever was cached last.

FALLBACK: if the cache file doesn't exist yet (fresh checkout, scraper
hasn't run once) or fails to parse, falls back to the OLD hardcoded
9am-sunset-every-day-of-the-year behavior rather than crashing or
silently treating CBI as perpetually closed -- a degraded-but-safe
default, not a hard dependency.

Used in two places:
  1. scripts/ingest_flag.py -- force the recorded flag color to 'closed'
     for any poll that lands outside real operating hours (in-season
     weekday/weekend hours, or any day of the off-season entirely),
     regardless of what the live API happens to report (the API can
     return a stale/leftover color after hours since it's just a JS
     variable on CBI's site, not necessarily reset the moment they
     close -- and definitely not reset for the off-season, since the
     flag widget isn't CBI-season-aware at all).
  2. website/app.py's api_flag_prediction -- the Bayesian prediction
     must show 'closed' with certainty for forecast hours outside real
     operating hours (including the entire off-season), and must NOT
     show any nonzero 'closed' probability during real open hours.

Close time is always "sunset" (confirmed across every row CBI has ever
published) -- computed astronomically via the `astral` package, since
CBI's real closing time follows sunset, not a fixed clock time.
"""
import json
from datetime import date, datetime, timezone

from astral import LocationInfo
from astral.sun import sun

# CBI dockhouse coordinates (see db/seed_locations.py, corrected 2026-09-10
# via Nominatim lookup).
CBI_LAT = 42.3598
CBI_LON = -71.0731

CACHE_PATH = "/home/firescar96/.hermes/projects/sailing-weather/db/cbi_hours_cache.json"
# Fallback defaults, used ONLY if the scraped cache is missing/invalid --
# matches this module's old year-round behavior before the season-aware
# scrape existed, so a fresh checkout (or a temporarily-broken scrape)
# degrades to "always technically possible to be open" rather than
# either crashing or silently showing CBI as closed forever.
FALLBACK_OPEN_HOUR_LOCAL = 9

_LOCATION = LocationInfo("CBI", "USA", "America/New_York", CBI_LAT, CBI_LON)

_cache = None  # lazily loaded + memoized per process; see _load_cache()


def _load_cache():
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(CACHE_PATH) as f:
            _cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        _cache = {}
    return _cache


def _season_bounds(for_date):
    """Returns (season_start, season_end) as date objects, or None if
    no cache is available (caller should fall back to the old
    always-open-by-hours-only behavior)."""
    cache = _load_cache()
    if not cache.get("season_start") or not cache.get("season_end"):
        return None
    return date.fromisoformat(cache["season_start"]), date.fromisoformat(cache["season_end"])


def _open_hour_for(local_date):
    """Returns the real scraped weekday/weekend open hour (int, local
    time) for local_date, or None if no cache row covers that date
    (either genuinely off-season, or the cache is stale/missing)."""
    cache = _load_cache()
    rows = cache.get("rows") or []
    is_weekend = local_date.weekday() >= 5  # Saturday=5, Sunday=6
    for row in rows:
        start = date.fromisoformat(row["start_date"])
        end = date.fromisoformat(row["end_date"])
        if start <= local_date <= end:
            open_spec = row["weekend_open"] if is_weekend else row["weekday_open"]
            return open_spec[0]  # (hour, minute) -> hour; CBI's published times are always on the hour or :00/:30, minute kept at 0 for simplicity since every observed row so far is exactly on the hour
    return None


def cbi_open_window_utc(for_date_utc):
    """Returns (open_utc, close_utc) datetimes (UTC, tz-aware) for CBI's
    real operating window on the LOCAL calendar date that for_date_utc
    falls on (converted to America/New_York first, since the season
    table, weekday/weekend open times, and sunset are all local-time
    concepts). Returns (None, None) if that calendar date falls outside
    the scraped season entirely (CBI closed ALL DAY) -- callers must
    handle this, see is_cbi_open() below."""
    import zoneinfo

    eastern = zoneinfo.ZoneInfo("America/New_York")
    local_dt = for_date_utc.astimezone(eastern)
    local_date = local_dt.date()

    bounds = _season_bounds(local_date)
    if bounds is not None:
        season_start, season_end = bounds
        if not (season_start <= local_date <= season_end):
            return None, None  # entirely outside the real season -- closed all day

    open_hour = _open_hour_for(local_date)
    if open_hour is None:
        # No cached row covers this date (cache missing/stale, or a
        # genuine gap in CBI's published table) -- fall back to the
        # old fixed 9am behavior rather than guessing further.
        open_hour = FALLBACK_OPEN_HOUR_LOCAL

    open_local = datetime(local_date.year, local_date.month, local_date.day,
                           open_hour, 0, 0, tzinfo=eastern)
    s = sun(_LOCATION.observer, date=local_date, tzinfo=timezone.utc)
    close_utc = s["sunset"]
    open_utc = open_local.astimezone(timezone.utc)
    return open_utc, close_utc


def is_cbi_open(at_utc):
    """True if `at_utc` (tz-aware UTC datetime) falls within CBI's real
    operating window (season + weekday/weekend open time + sunset) on
    its own local calendar date. Always False for any date outside the
    scraped season entirely."""
    if at_utc.tzinfo is None:
        at_utc = at_utc.replace(tzinfo=timezone.utc)
    open_utc, close_utc = cbi_open_window_utc(at_utc)
    if open_utc is None:
        return False  # outside the season entirely
    return open_utc <= at_utc <= close_utc
