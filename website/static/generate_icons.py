#!/usr/bin/env python3
"""Generates the Sailing Weather Dashboard's icon/logo.

Design (2026-09-11, reverted per user feedback "even worse, just give
me a basic shape as icon don't combine" -- every prior version combined
a pin + boat, or pin + boat + wind, and each combination read worse, not
better): ONE basic shape only. A plain sailboat silhouette (mast + sail
+ hull), nothing else layered on top or around it. No pin, no wind
lines, no cutout tricks.

White glyph on transparent background; separate opaque "maskable"
variants for Android and a flattened apple-touch-icon/favicon for iOS,
since neither platform supports transparent home-screen icons.
"""
from PIL import Image, ImageDraw

WHITE = (255, 255, 255)
DARK = (11, 22, 32)  # --bg from style.css -- used for flattened/maskable backgrounds
SIZES = [72, 96, 128, 144, 152, 192, 384, 512]


def draw_icon(size, padding_frac=0.14):
    """Draws at 4x supersampling then downsamples for clean antialiased
    edges -- PIL's own polygon/line drawing has no AA otherwise."""
    ss = 4
    s = size * ss
    pad = int(s * padding_frac)
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    cx = s / 2
    cy = s / 2
    scale = (s - 2 * pad) / 2

    # A single, plain sailboat silhouette: mast, one triangular sail,
    # and a simple hull. That's the whole icon -- no other shape.
    mast_top = (cx - scale * 0.05, cy - scale * 0.85)
    mast_bottom = (cx - scale * 0.05, cy + scale * 0.30)
    sail_tip = (cx + scale * 0.68, cy + scale * 0.05)
    d.polygon([mast_top, mast_bottom, sail_tip], fill=WHITE)
    mast_w = max(2, int(scale * 0.07))
    d.line([mast_top, mast_bottom], fill=WHITE, width=mast_w)

    hull_y = cy + scale * 0.30
    hull_half = scale * 0.85
    d.polygon(
        [
            (cx - hull_half, hull_y),
            (cx + hull_half * 0.92, hull_y),
            (cx + hull_half * 0.60, hull_y + scale * 0.22),
            (cx - hull_half * 0.65, hull_y + scale * 0.22),
        ],
        fill=WHITE,
    )

    img = img.resize((size, size), Image.LANCZOS)
    return img


def flatten_on_bg(img, bg):
    """For icon slots that traditionally don't expect transparency
    (apple-touch-icon, favicon) -- composite the white boat onto a dark
    solid background so it isn't invisible on light system chrome."""
    out = Image.new("RGBA", img.size, bg + (255,))
    out.alpha_composite(img)
    return out.convert("RGB")


def draw_maskable_icon(size, bg):
    """Android's 'maskable' purpose icons must NOT be transparent, and the
    glyph must sit inside a safe zone since the launcher can crop the
    icon into a circle/squircle/teardrop. Render the glyph at ~70% of
    the canvas, centered, on a solid background."""
    glyph = draw_icon(int(size * 0.72))
    canvas = Image.new("RGBA", (size, size), bg + (255,))
    offset = ((size - glyph.width) // 2, (size - glyph.height) // 2)
    canvas.alpha_composite(glyph, offset)
    return canvas.convert("RGB")


def main():
    dark_bg = DARK

    for size in SIZES:
        img = draw_icon(size)
        img.save(f"icons/icon-{size}.png")  # "any" purpose -- kept transparent
        print(f"wrote icons/icon-{size}.png")

    for size in (192, 512):
        maskable = draw_maskable_icon(size, dark_bg)
        maskable.save(f"icons/icon-{size}-maskable.png")
        print(f"wrote icons/icon-{size}-maskable.png")

    apple = flatten_on_bg(draw_icon(180), dark_bg)
    apple.save("icons/apple-touch-icon.png")
    print("wrote icons/apple-touch-icon.png")

    fav = flatten_on_bg(draw_icon(32), dark_bg)
    fav.save("favicon.png")
    print("wrote favicon.png")

    logo = draw_icon(256)  # kept transparent for use in the page header
    logo.save("logo.png")
    print("wrote logo.png")


if __name__ == "__main__":
    main()
