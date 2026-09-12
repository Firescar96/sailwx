#!/usr/bin/env python3
"""CBI Community Boating operating-hours helper.

CBI's Adult Program operates from 9:00 AM until sunset, daily (per user
2026-09-11). Used in two places:
  1. scripts/ingest_flag.py -- force the recorded flag color to 'closed'
     for any poll that lands outside 9am-sunset, regardless of what the
     live API happens to report (the API can return a stale/leftover
     color after hours since it's just a JS variable on CBI's site, not
     necessarily reset the moment they close).
  2. website/app.py's api_flag_prediction -- the Bayesian prediction
     must show 'closed' with certainty for forecast hours outside
     9am-sunset, and must NOT show any nonzero 'closed' probability
     during open hours (closed during open hours would only happen for
     e.g. severe weather closures, which the historical wind-based
     Bayesian estimate cannot represent well since sunset/hours-closure
     dominates the 'closed' class in the training data otherwise).

Sunset time is computed astronomically (varies ~4h from December to
June at this latitude) via the `astral` package -- CBI's real closing
time follows sunset, not a fixed clock time, so a hardcoded hour would
drift wrong across seasons.
"""
from datetime import datetime, timezone

from astral import LocationInfo
from astral.sun import sun

# CBI dockhouse coordinates (see db/seed_locations.py, corrected 2026-09-10
# via Nominatim lookup).
CBI_LAT = 42.3598
CBI_LON = -71.0731
OPEN_HOUR_LOCAL = 9  # 9:00 AM, per user 2026-09-11

_LOCATION = LocationInfo("CBI", "USA", "America/New_York", CBI_LAT, CBI_LON)


def cbi_open_window_utc(for_date_utc):
    """Returns (open_utc, close_utc) datetimes (UTC, tz-aware) for CBI's
    9am-sunset operating window on the LOCAL calendar date that
    for_date_utc falls on (converted to America/New_York first, since
    "9am" and "sunset" are both local-time concepts)."""
    import zoneinfo

    eastern = zoneinfo.ZoneInfo("America/New_York")
    local_dt = for_date_utc.astimezone(eastern)
    local_date = local_dt.date()

    open_local = datetime(local_date.year, local_date.month, local_date.day,
                           OPEN_HOUR_LOCAL, 0, 0, tzinfo=eastern)
    s = sun(_LOCATION.observer, date=local_date, tzinfo=timezone.utc)
    close_utc = s["sunset"]
    open_utc = open_local.astimezone(timezone.utc)
    return open_utc, close_utc


def is_cbi_open(at_utc):
    """True if `at_utc` (tz-aware UTC datetime) falls within CBI's
    9am-sunset window on its own local calendar date."""
    if at_utc.tzinfo is None:
        at_utc = at_utc.replace(tzinfo=timezone.utc)
    open_utc, close_utc = cbi_open_window_utc(at_utc)
    return open_utc <= at_utc <= close_utc
