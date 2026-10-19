#!/usr/bin/env python3
"""Scrape CBI's real, published Adult Program hours table.

ADDED 2026-10-19 (user: "when the season ends November 1st only the
cbi flag tracker should stop collecting and displaying data everything
else keeps going" -> clarified: "it would be better if you automatically
tracked closures on the site and didn't use closed times in
calculations"). Previously cbi_hours.py hardcoded a fixed 9am-sunset
daily window with NO season concept at all -- correct during the
season, but wrong the moment the season actually ends (CBI's real
Adult Program publishes seasonal date ranges with varying weekday
start times, confirmed live: 1pm Apr1-Jun14, 3pm Jun15-Aug21, 1pm
Aug22-Oct31, implicitly closed entirely outside Apr1-Oct31).

Rather than hardcode "November 1" (which could drift wrong next year,
or any year CBI changes its posted dates), this scrapes CBI's own
"Hours and Directions" page directly -- the real source of truth --
and caches the parsed schedule to a small JSON file that cbi_hours.py
reads. Confirmed live: the "Adult Program" table on that page is a
clean, structured HTML <table> (not a PDF/image), with 1 row per date
sub-range and 3 columns (date range, weekday start time, weekend start
time) -- parsed below with BeautifulSoup, not regex, since the table
structure is reliable.

Weekend rows are consistently "9:00 am to sunset" in every row CBI has
ever published (confirmed across all 3 sub-ranges) -- this script
still parses it explicitly rather than assuming, so a future change on
CBI's page is picked up automatically rather than silently ignored.

CLOSE time: always "sunset" per every row seen so far (not a fixed
clock time) -- cbi_hours.py's existing astral-based sunset calculation
is reused unchanged; this script only supplies the OPEN time (varies by
weekday/weekend and date sub-range) and the overall season boundary
(implicit: any date NOT covered by a row in the table means closed all
day, which is exactly what "off season" means).
"""
import json
import re
import ssl
import sys
from datetime import date, datetime, timezone

import requests
from bs4 import BeautifulSoup

CACHE_PATH = "/home/firescar96/.hermes/projects/sailing-weather/db/cbi_hours_cache.json"
URL = "https://www.community-boating.org/about-us/hours-and-directions/"

MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_date_range(text, year):
    """'April 1 - June 14' (or an en-dash, or with a trailing '*') ->
    (date(year,4,1), date(year,6,14))."""
    text = text.replace("*", "").strip()
    # Split on any dash-like character (hyphen, en-dash, em-dash)
    parts = re.split(r"[-\u2013\u2014]", text)
    if len(parts) != 2:
        raise ValueError(f"Could not split date range: {text!r}")
    start_s, end_s = parts[0].strip(), parts[1].strip()

    def parse_one(s, fallback_month=None):
        m = re.match(r"([A-Za-z]+)\s+(\d{1,2})", s)
        if m:
            month = MONTH_NAMES[m.group(1).lower()]
            day = int(m.group(2))
            return date(year, month, day), month
        # "14" alone (end of range reusing the start's month) -- not
        # observed yet on CBI's page (every row spells out both month
        # names), but handled defensively since table formatting could
        # change.
        day = int(re.match(r"(\d{1,2})", s).group(1))
        return date(year, fallback_month, day), fallback_month

    start_date, start_month = parse_one(start_s)
    end_date, _ = parse_one(end_s, fallback_month=start_month)
    return start_date, end_date


def parse_time(text):
    """'1:00 pm to sunset' -> (hour_24, minute) for the OPEN time only
    (the close time is always 'sunset' in every row CBI has published,
    handled separately by cbi_hours.py's existing astral calculation,
    not parsed here). BUG FOUND DURING TESTING: an earlier version
    checked `"sunset" in text` against the WHOLE cell text (e.g. "1:00
    pm to sunset"), which always matched since every row's close time
    is sunset -- so every open time silently came back None. Fixed by
    extracting only the portion before " to " and parsing a clock time
    from THAT, rather than checking the full string for "sunset"."""
    text = text.strip().lower()
    open_part = text.split(" to ")[0].strip()
    m = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)", open_part)
    if not m:
        raise ValueError(f"Could not parse open time from: {text!r} (open portion: {open_part!r})")
    hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    if ampm == "pm" and hour != 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    return hour, minute


def scrape():
    resp = requests.get(URL, timeout=30, headers={"User-Agent": "sailing-weather-archiver/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    heading = None
    for h in soup.find_all(["h3", "h4", "strong", "b"]):
        if "Adult Program" in h.get_text():
            heading = h
            break
    if heading is None:
        raise ValueError("Could not find 'Adult Program' heading on CBI's hours page -- page structure may have changed")
    table = heading.find_next("table")
    if table is None:
        raise ValueError("Found 'Adult Program' heading but no <table> immediately after it")

    # Table's own header row tells us which season year this is for
    # ("2026 Season") -- use that instead of assuming "this year",
    # since CBI could publish next year's schedule in advance.
    header_cells = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
    year_match = re.search(r"(\d{4})", header_cells[0])
    if not year_match:
        raise ValueError(f"Could not find a season year in header: {header_cells}")
    season_year = int(year_match.group(1))

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) != 3:
            continue
        date_range_text, weekday_text, weekend_text = cells
        start_date, end_date = parse_date_range(date_range_text, season_year)
        rows.append({
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "weekday_open": parse_time(weekday_text),
            "weekend_open": parse_time(weekend_text),
        })

    if not rows:
        raise ValueError("Parsed the Adult Program table but found zero usable rows")

    season_start = min(date.fromisoformat(r["start_date"]) for r in rows)
    season_end = max(date.fromisoformat(r["end_date"]) for r in rows)

    return {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "season_year": season_year,
        "season_start": season_start.isoformat(),
        "season_end": season_end.isoformat(),
        "rows": rows,
    }


def main():
    data = scrape()
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"CBI hours cache updated: season {data['season_start']} to {data['season_end']}, {len(data['rows'])} sub-ranges")
    for r in data["rows"]:
        print(f"  {r['start_date']} to {r['end_date']}: weekday open {r['weekday_open']}, weekend open {r['weekend_open']}")


if __name__ == "__main__":
    sys.exit(main())
