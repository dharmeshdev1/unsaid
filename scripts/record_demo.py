#!/usr/bin/env python3
"""Record docs/demo.gif from the running web interface.

    uv run --with playwright --with pillow scripts/record_demo.py http://127.0.0.1:8377 docs/demo.gif
"""
import io
import sys
import time

from PIL import Image
from playwright.sync_api import sync_playwright

QUERY = "early signs of dementia"
WIDTH, HEIGHT = 1100, 680
COLOURS = 48


def main():
    url, out = sys.argv[1], sys.argv[2]
    frames, durations = [], []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(url)
        page.wait_for_selector("body[data-state=ready]", timeout=180_000)
        # Wait until encryption prepared ahead has finished, as it would have for a real visitor.
        page.wait_for_selector("body[data-prepare-ms]", state="attached", timeout=60_000)
        time.sleep(4)

        def grab(duration):
            frames.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
            durations.append(duration)

        page.locator("#query").evaluate("el => el.blur()")
        grab(1400)
        page.click("#query")
        for i, letter in enumerate(QUERY):
            page.keyboard.type(letter)
            if i % 2 == 1 or i == len(QUERY) - 1:
                grab(110)
        grab(1100)
        page.keyboard.press("Enter")
        started = time.time()
        while time.time() - started < 0.7:
            grab(90)
        page.wait_for_selector("body[data-state=done]", timeout=60_000)
        time.sleep(0.6)
        grab(5200)
        browser.close()

    # One shared palette keeps the file small and stops colours flickering between frames.
    palette = frames[-1].quantize(colors=COLOURS, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(
        out, save_all=True, append_images=quantized[1:], duration=durations, loop=0, optimize=True
    )
    print(f"{len(frames)} frames, {out}")


if __name__ == "__main__":
    main()
