#!/usr/bin/env python3
"""Drive the web interface in a real browser and check what actually crosses the network.

    uv run --with playwright scripts/test_web.py http://127.0.0.1:8377 --shots out/

Works against `unsaid serve` and against the static demo site. It fails if a query appears in
any URL or request body, if two different searches differ in shape, or if the page logs an error.
"""
import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

QUERIES = ["early signs of dementia", "what happens when you go bankrupt"]
SECRET = "zebra crossing rules in winter"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("url")
    parser.add_argument("--shots", type=Path, help="directory for screenshots")
    args = parser.parse_args()
    if args.shots:
        args.shots.mkdir(parents=True, exist_ok=True)

    problems, console, sent = [], [], []

    def shot(page, name):
        if args.shots:
            page.screenshot(path=str(args.shots / f"{name}.png"))

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", lambda m: console.append(m.text) if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(str(e)))

        def record(request):
            body = request.post_data_buffer or b""
            sent.append((request.method, request.url, body))

        page.on("request", record)
        page.goto(args.url)
        shot(page, "1-loading")
        page.wait_for_selector("body[data-state=ready]", timeout=180_000)
        print("ready:", page.inner_text("#status"))
        shot(page, "2-ready")

        results = []
        for query in QUERIES + [SECRET]:
            before = len(sent)
            page.fill("#query", "")
            page.type("#query", query, delay=4)
            logged = page.inner_text("#log-query")
            if query.split()[0] not in logged:
                problems.append(f"the log preview did not follow typing: {logged!r}")
            page.keyboard.press("Enter")
            page.wait_for_function("document.body.dataset.state !== 'ready'")
            page.wait_for_selector("body[data-state=done]", timeout=120_000)
            titles = page.locator(".hit h3").all_inner_texts()
            print(f"{query!r}\n   {page.inner_text('#status')}\n   {titles[:5]}")
            if not titles:
                problems.append(f"no results for {query!r}")
            results.append(sent[before:])
            shot(page, "3-" + query.split()[0])

        posts = [[(u.rsplit("/", 1)[-1], len(b)) for m, u, b in r if m == "POST"] for r in results]
        print("requests per search:", posts)
        if len({tuple(x) for x in posts}) != 1:
            problems.append(f"searches differ in shape: {posts}")

        needles = [q.split()[-1].encode() for q in QUERIES + [SECRET]]
        for method, url, body in sent:
            for needle in needles:
                if needle in url.encode() or needle in body:
                    problems.append(f"query text {needle!r} appeared in {method} {url}")
        compare = page.inner_text("#led-compare")
        print("compare:", compare)
        print("caption:", page.inner_text("#pattern-caption"))
        print("log:", page.inner_text("#log-unsaid-code").replace("\n", " | "))

        width_ok = page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        if not width_ok:
            problems.append("the desktop page scrolls sideways")

        phone = browser.new_page(viewport={"width": 390, "height": 844})
        phone.goto(args.url)
        phone.wait_for_selector("body[data-state=ready]", timeout=180_000)
        phone.fill("#query", "why is the sky blue")
        phone.keyboard.press("Enter")
        phone.wait_for_selector("body[data-state=done]", timeout=120_000)
        if not phone.evaluate("document.documentElement.scrollWidth <= innerWidth"):
            problems.append("the phone layout scrolls sideways")
        if args.shots:
            phone.screenshot(path=str(args.shots / "4-phone.png"), full_page=True)
        browser.close()

    problems += [f"console: {line}" for line in console]
    for problem in problems:
        print("PROBLEM:", problem)
    print("ok" if not problems else f"{len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
