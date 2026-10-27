"""Run after `npm run preview` in web; Playwright is an optional dependency."""

import sys

from playwright.sync_api import sync_playwright


def main(url: str):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        desktop = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        desktop.on("pageerror", lambda error: errors.append(str(error)))
        desktop.goto(url)
        desktop.locator("#lab-content").wait_for(state="visible")
        desktop.keyboard.press("Tab")
        assert desktop.evaluate("document.activeElement.classList.contains('skip-link')")
        assert desktop.evaluate("getComputedStyle(document.activeElement).outlineStyle") == "solid"
        assert desktop.locator(".sample-button").count() == 10
        assert desktop.locator("#loss-chart polyline").count() == 2
        demo = desktop.request.get(f"{url}data/demo.json").json()
        for index, name in enumerate(("vit", "cnn")):
            displayed = desktop.locator(".result-row strong").nth(index).inner_text()
            displayed_accuracy = float(displayed.rstrip("%")) / 100
            assert abs(displayed_accuracy - demo["metrics"][name]["test"]["accuracy"]) < 0.001

        for index, title in enumerate(("Potong gambar", "Bentuk token", "Hubungkan token", "Baca kelas")):
            desktop.locator(f'[data-stage="{index}"]').click()
            assert desktop.locator("#stage-title").inner_text() == title
        desktop.locator('[data-stage="3"]').focus()
        desktop.keyboard.press("Enter")
        assert desktop.locator("#stage-title").inner_text() == "Baca kelas"

        for index, sample in enumerate(demo["samples"]):
            desktop.locator(".sample-button").nth(index).click()
            assert demo["classes"][sample["label"]] in desktop.locator("#image-caption").inner_text()
        for query in (0, 1, 49):
            desktop.locator("#query-select").select_option(str(query))
            assert desktop.locator("#query-select").input_value() == str(query)
        summaries = set()
        for layer in range(demo["config"]["depth"]):
            desktop.locator("#layer-select").select_option(str(layer))
            for head in range(demo["config"]["heads"]):
                desktop.locator("#head-select").select_option(str(head))
                summaries.add(desktop.locator("#attention-summary").inner_text())
        assert len(summaries) > 1
        desktop.locator("#sample-canvas").click(position={"x": 60, "y": 20})
        assert desktop.locator("#query-select").input_value() == "2"
        for link in desktop.locator('a[href^="#"]').all():
            assert desktop.locator(link.get_attribute("href")).count() == 1
        assert not errors, errors
        desktop.screenshot(path="artifacts/web-desktop.png", full_page=True)

        mobile = browser.new_page(viewport={"width": 375, "height": 812}, device_scale_factor=1)
        mobile.on("pageerror", lambda error: errors.append(str(error)))
        mobile.goto(url)
        mobile.locator("#lab-content").wait_for(state="visible")
        mobile.locator(".sample-button").nth(9).click()
        mobile.locator("#query-select").select_option("0")
        assert mobile.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        assert not errors, errors
        mobile.screenshot(path="artifacts/web-mobile.png", full_page=True)

        narrow = browser.new_page(viewport={"width": 320, "height": 700})
        narrow.goto(url)
        narrow.locator("#lab-content").wait_for(state="visible")
        assert narrow.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")

        zoomed = browser.new_page(viewport={"width": 400, "height": 900})
        zoomed.goto(url)
        zoomed.locator("#lab-content").wait_for(state="visible")
        assert zoomed.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")

        failed = browser.new_page()
        failed.route("**/data/demo.json", lambda route: route.abort())
        failed.goto(url)
        failed.locator("#error-state").wait_for(state="visible")
        failed.unroute("**/data/demo.json")
        failed.locator("#retry-button").click()
        failed.locator("#lab-content").wait_for(state="visible")

        empty = browser.new_page()
        empty.route(
            "**/data/demo.json",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body='{"schema_version":1,"samples":[]}',
            ),
        )
        empty.goto(url)
        empty.locator("#empty-state").wait_for(state="visible")
        empty.unroute("**/data/demo.json")
        empty.locator("#empty-retry-button").click()
        empty.locator("#lab-content").wait_for(state="visible")
        browser.close()
        print("Desktop, mobile, keyboard, all controls, links, empty and error retries passed.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:4173/")
