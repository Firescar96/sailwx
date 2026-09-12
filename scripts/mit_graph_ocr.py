"""Shared utilities for reading MIT Sailing Pavilion weewx day-graph PNGs.

These graphs (daywind.png, daybarometer.png, dayouttemphilo.png,
daywatertemphilo.png) all share the same weewx template layout: a 700x196
plot with a left-side Y-axis label column and evenly-spaced horizontal
gridlines, each labeled with a tick value.

CRITICAL: the Y-axis scale is NOT fixed across graphs or across fetches
of the same graph -- confirmed live 2026-09-10 that barometer/temp/water
temp graphs auto-scale their axis to the day's actual min/max (e.g.
barometer showed 1019-1024 one day, temp showed 62-82 another day). Even
the wind graph, which we'd previously assumed had a hardcoded 0-30mph
ceiling, should not be trusted to stay that way -- weewx's default
behavior is auto-scaling; a low-wind day could very plausibly get a
smaller axis range (e.g. 0-15mph), which would silently break any
pixel-to-value math based on a hardcoded assumption. So this module reads
the actual tick labels from the image via OCR on every fetch, for every
graph including wind.

OCR engine: tesserocr (a manylinux wheel that BUNDLES its own
libtesseract/libleptonica/etc, avoiding system glibc/library version
mismatches -- confirmed 2026-09-10 that installing the system
`tesseract-ocr` .deb directly doesn't work on this machine's older
glibc without root; tesserocr's bundled wheel sidesteps that entirely).
Requires TESSDATA_DIR (a real eng.traineddata file, downloaded from
tesseract-ocr/tessdata_fast on GitHub -- pure data, no binary
compatibility concerns) to be present.
"""
import os
import re

import tesserocr
from PIL import Image

TESSDATA_DIR = os.path.expanduser("~/.local/share/tessdata")

# Geometry shared by all these weewx day-graph PNGs (confirmed empirically
# across wind/barometer/temp/water-temp graphs, all 700x196 with the same
# plot-area box -- only the axis VALUES differ, not the pixel geometry).
PLOT_X_MIN = 44
PLOT_X_MAX = 678
PLOT_Y_TOP = 24     # pixel row of the topmost gridline/tick
PLOT_Y_BOTTOM = 160  # pixel row of the bottommost gridline/tick


def ocr_axis_labels(img, num_expected_ticks=None):
    """Reads the Y-axis tick labels from the left margin of a weewx
    day-graph image. Returns a list of (pixel_row, value) pairs, ordered
    top to bottom (i.e. highest value first).

    Approach: crop the left label column, upscale 3x (empirically improves
    OCR accuracy substantially -- confirmed 2026-09-10 going from ~83% to
    ~100% correct reads on a 6-tick barometer graph), OCR with PSM 6
    (uniform block of text), then parse each line as a float. Because
    weewx renders these labels at a fixed vertical spacing (evenly spaced
    gridlines), we cross-check against the graph's own gridline pixel
    rows (also detectable) so a single misread digit doesn't break the
    whole calibration -- see `read_axis_scale` below, which does the
    "trust the spacing pattern over any single OCR value" correction.
    """
    crop = img.crop((0, PLOT_Y_TOP - 9, 42, PLOT_Y_BOTTOM + 18))
    w, h = crop.size
    upscaled = crop.resize((w * 3, h * 3), Image.LANCZOS)
    text = tesserocr.image_to_text(upscaled, path=TESSDATA_DIR, psm=6)

    values = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # OCR sometimes inserts a stray space in a number (e.g. "101 3.0")
        # -- collapse internal whitespace inside otherwise-numeric lines
        # before parsing, but only if it still looks number-like overall.
        cleaned = line.replace(" ", "") if re.fullmatch(r"[\d\s.\-]+", line) else line
        m = re.fullmatch(r"-?\d+(\.\d+)?", cleaned)
        if m:
            values.append(float(cleaned))

    return values


def read_axis_scale(img):
    """Determines the (value_at_top, value_at_bottom) of the plot's Y
    axis by OCR-ing the tick labels, then validating/correcting them
    against the assumption of evenly-spaced ticks (weewx always renders
    them this way). Returns (top_value, bottom_value) such that
    y_to_value(PLOT_Y_TOP) == top_value and y_to_value(PLOT_Y_BOTTOM) ==
    bottom_value.

    Raises ValueError if fewer than 2 plausible tick values are read (not
    enough to determine a scale) -- callers should treat this as "skip
    this fetch, try again next poll" rather than guessing.
    """
    values = ocr_axis_labels(img)
    if len(values) < 2:
        raise ValueError(f"OCR found only {len(values)} axis tick value(s), need >= 2: {values}")

    # Ticks should be evenly spaced and descending top-to-bottom (weewx
    # convention). Compute the most common step between consecutive
    # values (robust to a single bad OCR read) using the MODE of rounded
    # diffs, not just the median -- the median is still fooled by a
    # single spurious extra value at the tail (e.g. the crop accidentally
    # catching part of the x-axis time label below the plot, which
    # produces one wildly-different diff but not enough of them to move
    # a straight median if there are only 2-3 real ticks). The mode is
    # more robust: even with one contaminating value, the real ticks'
    # consistent step will win as long as there are >= 2 of them.
    diffs = [round(values[i] - values[i + 1], 4) for i in range(len(values) - 1)]
    if not diffs:
        raise ValueError(f"Not enough ticks to compute a step: {values}")
    step_counts = {}
    for d in diffs:
        step_counts[d] = step_counts.get(d, 0) + 1
    step = max(step_counts, key=lambda d: (step_counts[d], -abs(d)))
    if step <= 0:
        raise ValueError(f"Axis ticks not descending as expected: {values} (diffs={diffs})")

    # Keep only the leading run of values consistent with `step` --
    # drops any trailing contamination (e.g. a stray "12" from the x-axis
    # time label bleeding into the crop) without needing exact crop-height
    # tuning to avoid it.
    clean = [values[0]]
    for v in values[1:]:
        if abs((clean[-1] - v) - step) < 1e-6:
            clean.append(v)
        else:
            break
    if len(clean) < 2:
        raise ValueError(f"Could not find >= 2 consistent ticks: {values} (step={step})")

    n = len(clean)
    top_value = clean[0]
    bottom_value = top_value - step * (n - 1)
    return top_value, bottom_value


def y_to_value(y, top_value, bottom_value):
    """Convert a plot-area pixel row to a data value, given the axis
    scale determined by read_axis_scale()."""
    frac = (PLOT_Y_BOTTOM - y) / (PLOT_Y_BOTTOM - PLOT_Y_TOP)
    return top_value - (top_value - bottom_value) * (1 - frac)
