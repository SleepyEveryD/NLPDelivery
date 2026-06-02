"""Headless-browser article fetch -- for what `requests` CANNOT do (JS consent walls, JS redirects).

The live-News bottleneck: the answer ("four Canadians quarantining..", a quote's speaker) lives in the
article BODY, but every keyless way to reach it from Colab failed -- the Google-News `<link>` is a JS
consent-wall redirect, link-decoding broke (format changed), and DuckDuckGo is blocked on the Colab IP.
A real browser does what `requests` cannot: it RUNS the JavaScript, so it sails past the consent wall
(cookie pre-set + a button click) and follows the redirect to the real publisher article -- then we read
the RENDERED text (RAW article content, no synthesis -- rule-compliant; name "headless Chromium" in the video).

Two hard-won design points:
  * ONE Chromium, lazily launched and REUSED for the whole session. Launch is ~1-2s; each nav ~2-4s.
  * Run it in a DEDICATED WORKER THREAD. Jupyter/Colab already own an asyncio loop, and Playwright's
    sync API refuses to run inside one ("Sync API inside asyncio loop"); a private thread its own loop has.
Crash-safe ENTIRELY: Playwright missing, launch failure, a slow/blocked page -> "" (the caller keeps headlines).
"""

from __future__ import annotations

import queue
import re
import threading

# Pre-seed Google's consent so the wall is skipped; a button click is the fallback when it still appears.
_CONSENT_COOKIES = [
    {"name": "SOCS", "value": "CAESEwgDEgk0ODE3Nzk3MjQaAmVuIAEaBgiA_LyaBg", "domain": ".google.com", "path": "/"},
    {"name": "CONSENT", "value": "PENDING+987", "domain": ".google.com", "path": "/"},
]
_TAG = re.compile(r"<[^>]+>")


def _is_prose(t: str) -> bool:
    """True when REAL article prose this looks like -- not nav/menu/cookie/JS boilerplate."""
    if len(t) < 50 or len(t.split()) < 10:
        return False
    if "window." in t or "function(" in t or t[:30].count("{"):
        return False
    if ". " not in t and not t.endswith("."):
        return False
    return (t.count(" ") / len(t)) >= 0.12


class BrowserArticleFetcher:
    """A reusable headless Chromium, served from one worker thread. `fetch(url)` -> prose text or ""."""

    def __init__(self, nav_timeout_s: float = 8.0):
        self._nav_timeout = nav_timeout_s
        self._req_q: "queue.Queue" = queue.Queue()
        self._ready = threading.Event()
        self._ok = False
        self._started = False
        self._lock = threading.Lock()

    def _ensure_worker(self) -> bool:
        """Lazily start the browser thread. True when a live browser we have; False if Playwright/launch failed."""
        with self._lock:
            if self._started:
                return self._ok
            self._started = True
            t = threading.Thread(target=self._run, name="browser-fetch", daemon=True)
            t.start()
        self._ready.wait(timeout=60)   # the cold launch (+ first Chromium download already done), wait for it.
        return self._ok

    def _run(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            self._ready.set()
            return
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                ctx = browser.new_context(locale="en-US")
                ctx.add_cookies(_CONSENT_COOKIES)
                self._ok = True
                self._ready.set()
                while True:
                    job = self._req_q.get()
                    if job is None:
                        break
                    url, max_chars, resq = job
                    try:
                        resq.put(self._extract(ctx, url, max_chars))
                    except Exception:
                        resq.put("")
                browser.close()
        except Exception:
            self._ok = False
            self._ready.set()

    def _extract(self, ctx, url: str, max_chars: int) -> str:
        pg = ctx.new_page()
        try:
            pg.goto(url, wait_until="domcontentloaded", timeout=int(self._nav_timeout * 1000))
            if "consent.google" in pg.url:      # the wall the cookie did not skip -- click through it we do.
                for sel in ("button:has-text('Reject all')", "button:has-text('Accept all')",
                            "form[action*='consent'] button"):
                    try:
                        pg.click(sel, timeout=2000)
                        break
                    except Exception:
                        continue
            try:
                pg.wait_for_load_state("networkidle", timeout=2500)   # trimmed (was 4000) -- the 30s-wall
            except Exception:                                          # margin matters more than late-loading ads.
                pass
            paras = pg.eval_on_selector_all("p", "els => els.map(e => e.innerText)")
            out, total = [], 0
            for raw in paras:
                t = re.sub(r"\s+", " ", raw or "").strip()
                if not _is_prose(t):
                    continue
                out.append(t)
                total += len(t)
                if total >= max_chars:
                    break
            return " ".join(out).strip()[:max_chars]
        finally:
            try:
                pg.close()
            except Exception:
                pass

    def fetch(self, url: str, max_chars: int = 1500) -> str:
        """The article's prose text, via the shared browser. "" on ANY failure (caller keeps headlines)."""
        if not url or not self._ensure_worker():
            return ""
        resq: "queue.Queue" = queue.Queue(1)
        try:
            self._req_q.put((url, max_chars, resq), timeout=2)
            return resq.get(timeout=self._nav_timeout + 8)
        except Exception:
            return ""


# A process-wide singleton -- the browser ONCE for the whole sweep we launch, not per question.
_SINGLETON: "BrowserArticleFetcher | None" = None
_SINGLETON_LOCK = threading.Lock()


def get_browser_fetcher(nav_timeout_s: float = 8.0) -> "BrowserArticleFetcher":
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = BrowserArticleFetcher(nav_timeout_s=nav_timeout_s)
        return _SINGLETON
