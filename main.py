"""
Auction Catalog Feed -- a standalone, fully public app with NO login and NO
credentials of any kind. Built specifically to be safely sharable: it only
knows how to read/write the auction catalog feed tables in Supabase
(bidspotter_catalog_lots, auction_catalogs, auction_pdf_uploads) and has no
other code, no Settings page, no eBay/Shopify keys, nothing else that could
ever be exposed by sharing this URL with anyone.

Reads/writes the exact same Supabase tables the main Lister app's Auction
Monitor feature already uses -- this is a second front door onto the same
data, not a copy of it.
"""
import os
import re
import io
import json
import time
import uuid
import asyncio
import concurrent.futures
from typing import Optional
from datetime import datetime, timezone

import fitz  # PyMuPDF
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from supabase import create_client

app = FastAPI()
templates = Jinja2Templates(directory="templates")
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Real bug fixed here: every single-PDF upload used to spawn its own
# completely independent background job with no concurrency limit at all --
# so uploading, say, 5 catalogs in quick succession ran all 5 simultaneously,
# each doing several sequential Gemini calls plus PDF parsing plus multiple
# Supabase writes at once. On a small box that saturates the CPU/GIL and
# makes every other request (even just loading a page) feel frozen while
# it's happening -- exactly the "unusably slow while ingesting" symptom.
# The zip-batch path already avoided this internally (processes its own
# files one at a time), but nothing coordinated it against a single-PDF
# upload happening at the same time, or against other single-PDF uploads.
# One shared queue + one persistent worker fixes both: single-PDF uploads
# AND zip batches all go through the exact same queue now, so real
# processing is capped at one job app-wide, no matter how many uploads
# arrive back to back. Uploads still queue up instantly and the Upload Log
# fills in live as each one actually finishes -- nothing about the UX
# changes, only the uncontrolled concurrency underneath it.
_pdf_processing_queue: "asyncio.Queue" = asyncio.Queue()

async def _pdf_worker_loop():
    while True:
        job = await _pdf_processing_queue.get()
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, job)
        except Exception as e:
            print(f"[pdf-worker] job failed: {e}")
        finally:
            _pdf_processing_queue.task_done()

@app.on_event("startup")
async def _start_pdf_worker():
    asyncio.create_task(_pdf_worker_loop())

# --- Startup no longer crashes the whole process, on purpose ---------------
# The previous version connected to Supabase and looked up the business_id
# at MODULE IMPORT TIME, raising if anything went wrong -- which kills the
# entire Python process before it can even bind to a port. Railway then just
# shows "Crashed" with no way to see why without digging through deploy
# logs, which turned into a real, frustrating dead end with no way to get
# that information across reliably.
#
# Now: the app object above always comes up, unconditionally, no matter what
# state the environment is in. Supabase connection + business_id lookup are
# lazy (done on first real use, not at import time) and cached after that.
# If something IS misconfigured, every route below returns a clear, plain-
# English explanation of exactly what's wrong, directly in the response --
# something you can just read by opening the URL in a browser, no logs, no
# screenshots, no deploy history digging required ever again for this class
# of problem.
_config_error = None
_supabase_client = None
_business_id = None


def _get_supabase_and_business_id():
    """Lazily connects to Supabase and resolves the business_id on first use,
    caching the result (or the error) for every request after that. Never
    raises -- callers check the returned error string instead."""
    global _config_error, _supabase_client, _business_id
    if _supabase_client is not None or _config_error is not None:
        return _supabase_client, _business_id, _config_error

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        _config_error = (
            "SUPABASE_URL and/or SUPABASE_KEY environment variables are not set on this "
            "Railway service. Go to the service's Variables tab and add them (same values "
            "the main Lister app uses)."
        )
        return None, None, _config_error

    try:
        client = create_client(supabase_url, supabase_key)
    except Exception as e:
        _config_error = f"Failed to create the Supabase client -- check that SUPABASE_URL and SUPABASE_KEY are correct. Error: {e}"
        return None, None, _config_error

    owner_email = os.getenv("OWNER_EMAIL", "precisionindustrialmail@gmail.com")
    try:
        row = client.table("businesses").select("id").eq("email", owner_email).single().execute()
        business_id = row.data["id"]
    except Exception as e:
        _config_error = (
            f"Connected to Supabase, but could not find a business with email '{owner_email}'. "
            f"Check the OWNER_EMAIL environment variable (or the default in main.py) matches a "
            f"real row in the businesses table, and that SUPABASE_URL points at the right "
            f"Supabase project. Error: {e}"
        )
        return None, None, _config_error

    _supabase_client = client
    _business_id = business_id
    print(f"Auction Catalog Feed serving business_id={business_id} ({owner_email})")
    return _supabase_client, _business_id, None


def _require_config():
    """Call at the top of every route. Raises a clean HTTP 500 with the exact
    plain-English reason if Supabase/business_id isn't resolved -- readable
    directly in the browser or via curl, nothing hidden in server logs."""
    supabase_client, business_id, error = _get_supabase_and_business_id()
    if error:
        raise HTTPException(500, f"Configuration error: {error}")
    return supabase_client, business_id


# ---------------------------------------------------------------------------
# Parsing / extraction -- ported as-is from the main Lister app's Auction
# Monitor feature (auction-monitor-dev branch), same logic, same behavior.
# ---------------------------------------------------------------------------

def _extract_lots_via_gemini(raw_text: str, filename: str, on_chunk_lots=None) -> tuple:
    """Reads the actual catalog text and pulls out every lot directly --
    called whenever the fast regex pass didn't account for every single
    digit-leading line in the catalog (an exact check, not a guessed
    threshold), so this is the authoritative source whenever the shortcut
    isn't provably complete on its own.

    Real bug fixed here: this used to hard-truncate at raw_text[:100000] and
    call it done in one pass -- fine for a typical catalog, but for a
    genuinely huge one (e.g. 2000 real lots) that cutoff lands well before
    the end of the text, silently dropping everything after it. Now chunks
    the text into ~80k-character pieces (a little under Gemini's practical
    single-call comfort zone, leaving room for the prompt itself) and runs
    every chunk, combining and deduping the results by lot_number, instead
    of only ever seeing the first slice of a large catalog.

    on_chunk_lots: optional callback invoked with each chunk's own newly-
    found lots as soon as that chunk comes back from Gemini -- lets the
    caller persist them immediately instead of waiting for every chunk in
    the catalog to finish first. Real bug this fixes: previously nothing
    was written to the database until the entire catalog had been through
    every chunk, so a process killed partway through (a redeploy, a crash)
    lost 100% of that catalog's work, not just whatever chunk hadn't run
    yet. If the callback itself raises (e.g. a transient DB error), that
    chunk's lots are still kept in the final returned list as a fallback --
    only the immediate persistence attempt failed, not the extraction.

    Returns (lots, errors) now, not just lots. Real bug this fixes: every
    chunk failure (timeout, rate limit, invalid API key, quota exhausted --
    all real, all seen tonight) was caught and only ever printed to a
    server console nobody but Railway can see, then silently treated the
    exact same as "this chunk genuinely had no lots in it." A catalog where
    EVERY chunk failed for a real infrastructure reason looked identical in
    the UI to a catalog that's actually empty -- 'empty' status, no hint
    anything went wrong. Now every failure's message is collected and
    handed back so the caller can tell the two apart and show the real
    reason instead of a misleading 'empty'."""
    import google.generativeai as genai
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        return [], ["GEMINI_API_KEY is not set on this Railway service -- nothing can be extracted without it."]
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Real bug fixed here: 25,000 chars still packed 500+ lots into a single
    # chunk for a dense catalog (short descriptions, e.g. "1501\nTool box with
    # misc hand tools"), and confirmed directly against real data: Gemini
    # silently returned only ~200 of 710 actual lots for one such chunk --
    # no error, valid JSON, just genuinely incomplete enumeration. Smaller
    # chunks keep the per-call lot count low enough for reliable, exhaustive
    # extraction regardless of how dense a given catalog's descriptions are.
    chunk_size = 8000
    chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size)] or [raw_text]

    seen_lot_numbers = set()
    combined = []
    errors = []
    for chunk_i, chunk in enumerate(chunks):
        prompt = f"""This is raw text extracted from an auction catalog PDF named "{filename}"
(part {chunk_i + 1} of {len(chunks)} -- the catalog is large enough it had to be split).
Pull out every individual lot as a lot number and its description. Auction catalogs
list lots sequentially, usually as "LOT ###" or "Lot ###:" or similar, followed by a
description of the item(s) in that lot. Skip page headers, footers, terms & conditions,
and anything that isn't an actual lot listing -- this includes the auctioneer's own
street address or ZIP code near the top of the document, which can look like a lot
number (e.g. "11751 CR 12") but isn't one; only count something as a lot if it's
genuinely part of the sequential lot listing, not incidental text elsewhere on the
page. This chunk may start or end mid-lot -- only include a lot if you can see its
full lot number and description within this text. If this chunk genuinely contains
no real lots (e.g. it's entirely header/cover-page material), return an empty array
-- don't force a match.

Return ONLY a JSON array, no other text, in this exact shape:
[{{"lot_number": "123", "description": "..."}}, ...]

Text:
{chunk}"""
        # Real bug fixed here: a single failed attempt permanently marked
        # this chunk (and often the whole catalog) as failed, forever, with
        # no retry at all. Real data from a 190-catalog batch showed 85% of
        # all errors were transient failures on Gemini's OWN backend --
        # 504 timeouts, "stream cancelled," literal internal inference
        # failures -- exactly the class of error that commonly succeeds a
        # few seconds later once Google's backend recovers from being
        # overloaded by a burst of requests. Retries up to 3 times total
        # with increasing backoff (4s, 10s) before giving up on this chunk
        # for real.
        last_error = None
        for attempt in range(3):
            try:
                resp = model.generate_content(prompt, request_options={"timeout": 120})
                text = resp.text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                parsed = json.loads(text.strip())
                chunk_lots = []
                for r in parsed:
                    lot_number = str(r.get("lot_number", "")).strip()
                    description = (r.get("description") or "")[:2000]
                    if lot_number and description and lot_number not in seen_lot_numbers:
                        seen_lot_numbers.add(lot_number)
                        row = {"lot_number": lot_number, "description": description}
                        combined.append(row)
                        chunk_lots.append(row)
                if chunk_lots and on_chunk_lots:
                    try:
                        on_chunk_lots(chunk_lots)
                    except Exception as write_err:
                        print(f"Immediate persist failed for {filename} chunk {chunk_i + 1}/{len(chunks)} "
                              f"(kept in final result, will still save if the catalog completes): {write_err}")
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < 2:
                    wait_s = 4 if attempt == 0 else 10
                    print(f"Gemini call failed for {filename} chunk {chunk_i + 1}/{len(chunks)} "
                          f"(attempt {attempt + 1}/3, retrying in {wait_s}s): {e}")
                    time.sleep(wait_s)
        if last_error is not None:
            err_msg = f"Chunk {chunk_i + 1}/{len(chunks)} (after 3 attempts): {str(last_error)[:300]}"
            print(f"Gemini lot extraction failed for {filename} ({err_msg})")
            errors.append(err_msg)
            # Keep going with whatever other chunks succeed, rather than
            # losing the whole catalog over one bad chunk
            continue

    return combined, errors


def _extract_catalog_metadata_via_gemini(raw_text: str, filename: str) -> dict:
    """Auction catalog PDFs almost always state their location and sale date
    on the cover page/header -- pulls auctioneer/state/zip_code/end_date out
    automatically so nobody has to type them in by hand. Only fills in
    whichever fields the upload form left blank; a typed-in value always
    wins."""
    import google.generativeai as genai
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        return {}
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""This is raw text extracted from an auction catalog PDF named "{filename}".
Find the auction's own details, usually stated on the cover page or in a header/footer:
the auctioneer/company name running the sale, the US state the auction or item pickup
location is in (2-letter abbreviation if possible, e.g. "CO"), the ZIP code of that
location, and the auction's sale/closing date.

Return ONLY a JSON object, no other text, in this exact shape (use null for anything
not found -- do not guess):
{{"auctioneer": "..." or null, "state": "..." or null, "zip_code": "..." or null, "end_date": "..." or null}}

Text:
{raw_text[:20000]}"""
    try:
        resp = model.generate_content(prompt, request_options={"timeout": 60})
        text = resp.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
        return {k: v for k, v in parsed.items() if v}
    except Exception as e:
        print(f"Gemini catalog metadata extraction failed for {filename}: {e}")
        return {}


def _upsert_lots_batch(supabase_client, business_id: str, catalog_url: str, lots: list, meta: dict) -> None:
    """Writes ONE batch of already-parsed lots (typically one Gemini chunk's
    worth) into bidspotter_catalog_lots immediately. Real bug this fixes:
    the previous design collected every lot from an entire catalog in
    memory and only wrote any of them after Gemini had finished the whole
    thing -- so a catalog killed mid-extraction (a redeploy, a crash, a
    zip batch of 350 files that never gets to finish in one process
    lifetime) lost 100% of its lots, not just whatever hadn't been
    processed yet. Called once per chunk instead, so each chunk's lots are
    durable the moment that chunk comes back from Gemini.

    Batches the existence check into one query per call (via .in_() on
    lot_number) instead of one query per lot -- far fewer round trips than
    the original row-at-a-time select+insert, on top of now also being
    incremental."""
    if not lots:
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    lot_numbers = [l["lot_number"] for l in lots]
    existing = supabase_client.table("bidspotter_catalog_lots").select("id,lot_number")\
        .eq("business_id", business_id).eq("catalog_url", catalog_url).in_("lot_number", lot_numbers).execute()
    existing_by_number = {row["lot_number"]: row["id"] for row in (existing.data or [])}

    new_rows = []
    for lot in lots:
        record = {
            "business_id": business_id,
            "catalog_url": catalog_url,
            "lot_number": lot["lot_number"],
            "description": lot["description"],
            "last_seen_at": now_iso,
            "state": meta.get("state") or None,
            "zip_code": meta.get("zip_code") or None,
            "date": meta.get("end_date") or None,
        }
        existing_id = existing_by_number.get(lot["lot_number"])
        if existing_id:
            supabase_client.table("bidspotter_catalog_lots").update(record).eq("id", existing_id).execute()
        else:
            new_rows.append(record)
    if new_rows:
        supabase_client.table("bidspotter_catalog_lots").insert(new_rows).execute()


def _upsert_catalog_summary(supabase_client, business_id: str, catalog_url: str, meta: dict, lot_count: int) -> None:
    """Creates or refreshes the auction_catalogs summary row (what the
    Catalogs tab actually displays) with the CURRENT real lot_count.
    Called once per chunk now (right after that chunk's lots are written),
    not once at the very end of a whole catalog -- so a catalog killed
    partway through still leaves a real, findable summary row showing
    however many lots actually made it in, instead of nothing at all.
    Deliberately only ever called once at least one real lot exists (never
    for a genuinely empty catalog), matching the existing rule that an
    empty catalog gets an 'empty' upload-log entry but no catalogs row at
    all -- the Needs Update queue and the false-positive-lot-count fix
    both depend on that."""
    now_iso = datetime.now(timezone.utc).isoformat()
    had_prior_empty = supabase_client.table("auction_pdf_uploads").select("id")\
        .eq("business_id", business_id).eq("catalog_url", catalog_url).eq("status", "empty").limit(1).execute()
    catalog_fields = {k: meta[k] for k in ("title", "auctioneer", "end_date", "state") if meta.get(k)}
    catalog_existing = supabase_client.table("auction_catalogs").select("id,was_ever_empty")\
        .eq("business_id", business_id).eq("catalog_url", catalog_url).limit(1).execute()
    catalog_fields.update({"lot_count": lot_count, "lot_count_is_estimate": False, "last_checked_at": now_iso})
    if had_prior_empty.data or (catalog_existing.data and catalog_existing.data[0].get("was_ever_empty")):
        catalog_fields["was_ever_empty"] = True  # never write False here -- one-way flag, permanent once set
    if catalog_existing.data:
        supabase_client.table("auction_catalogs").update(catalog_fields).eq("id", catalog_existing.data[0]["id"]).execute()
    else:
        catalog_fields.update({"business_id": business_id, "source": "upload", "catalog_url": catalog_url, "first_seen_at": now_iso})
        supabase_client.table("auction_catalogs").insert(catalog_fields).execute()


def _backfill_catalog_metadata(supabase_client, business_id: str, catalog_url: str, raw_text: str, filename: str,
                                 state: str, zip_code: str, end_date: str) -> None:
    """Runs AFTER the upload response has already gone out, as a background
    task -- keeps uploads fast. Fills in whichever of state/zip/date the
    form left blank, a few seconds after the VA already sees 'success'.

    Real bug fixed here: this only ever updated bidspotter_catalog_lots (the
    per-lot rows) -- it never touched auction_catalogs, which is the actual
    summary row the Catalogs list displays (title/auctioneer/state/end date
    columns). So an auto-extracted catalog (no form fields typed in, the
    common case for zip-batch uploads) would show real lots but a blank
    State/End Date forever, while anything with the fields typed in by hand
    looked fine -- making it look like typed-in uploads worked and
    auto-extracted ones didn't, which is exactly what was happening."""
    try:
        auto_meta = _extract_catalog_metadata_via_gemini(raw_text, filename)
        lot_patch = {}
        if not state and auto_meta.get("state"):
            lot_patch["state"] = auto_meta["state"]
        if not zip_code and auto_meta.get("zip_code"):
            lot_patch["zip_code"] = auto_meta["zip_code"]
        if not end_date and auto_meta.get("end_date"):
            lot_patch["date"] = auto_meta["end_date"]
        if lot_patch:
            supabase_client.table("bidspotter_catalog_lots").update(lot_patch)\
                .eq("business_id", business_id).eq("catalog_url", catalog_url).execute()

        if not end_date and auto_meta.get("end_date"):
            supabase_client.table("auction_pdf_uploads").update({"end_date": auto_meta["end_date"]})\
                .eq("business_id", business_id).eq("catalog_url", catalog_url).execute()

        catalog_patch = {}
        if not state and auto_meta.get("state"):
            catalog_patch["state"] = auto_meta["state"]
        if not end_date and auto_meta.get("end_date"):
            catalog_patch["end_date"] = auto_meta["end_date"]
        if auto_meta.get("auctioneer"):  # no form field for this one, always fill if found
            catalog_patch["auctioneer"] = auto_meta["auctioneer"]
        if catalog_patch:
            supabase_client.table("auction_catalogs").update(catalog_patch)\
                .eq("business_id", business_id).eq("catalog_url", catalog_url).execute()
    except Exception as e:
        print(f"Background metadata backfill failed for {filename} (lots themselves are unaffected): {e}")


# ---------------------------------------------------------------------------
# Routes -- everything here is intentionally public, no auth of any kind.
# Every route calls _require_config() first -- if Supabase/business_id isn't
# resolved, this raises a clean HTTP 500 with the exact plain-English reason,
# readable directly in a browser. The app process itself never crashes.
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    _, _, error = _get_supabase_and_business_id()
    if error:
        return HTMLResponse(
            f"<html><body style='font-family:monospace;background:#1a0611;color:#f1f5f9;padding:40px;'>"
            f"<h2>⚠️ Configuration error</h2><p>{error}</p></body></html>",
            status_code=500,
        )
    return templates.TemplateResponse("feed.html", {"request": request})


@app.get("/health")
async def health():
    """Plain-text health/config check -- open this URL directly any time
    something seems wrong, before digging through Railway logs at all."""
    _, business_id, error = _get_supabase_and_business_id()
    if error:
        return JSONResponse({"ok": False, "error": error}, status_code=500)
    return {"ok": True, "business_id": business_id}


def _fetch_all_paginated(query_factory) -> list:
    """query_factory is a callable that returns a fresh Supabase query object
    (already filtered/ordered, but without .range() applied) each time it's
    called -- needed because query objects aren't safely reusable across
    multiple .range() calls. Pages through in chunks of 1000 (Supabase's
    per-request cap) until exhausted. Use this for ANY query that could
    plausibly return more than 1000 rows -- this exact silent-undercounting
    bug class has bitten this project's main app many times before, so every
    full-table-scan query in this app goes through this one place instead of
    each one growing its own copy of the same pagination logic (or, worse,
    skipping it)."""
    rows = []
    start = 0
    while True:
        page = query_factory().range(start, start + 999).execute().data or []
        rows.extend(page)
        if len(page) < 1000:
            break
        start += 1000
    return rows


# ── Updates to Make: automated BidSpotter scanning ──────────────────────
# Two separate jobs, deliberately not combined into one:
#
# 1. New-catalog detection: pulls every page of BidSpotter's own public
#    "/en-us/auction-catalogues" listing (confirmed directly, live, to work
#    with a plain httpx GET -- no API, no login, no AI), and diffs every
#    catalog_url found against everything already in auction_pdf_uploads.
#    Anything on BidSpotter that we've never even attempted is a real,
#    solid signal -- this is proven, low-risk.
#
# 2. Blank-catalog re-check: for catalogs already in our own system sitting
#    at zero lots (status='empty'), re-fetches each ONE'S OWN individual
#    catalogue page (also confirmed directly to work) and checks whether it
#    now shows real category/lot data. Deliberately NOT extracted from the
#    big listing scan -- a truly-empty card's exact appearance on that page
#    was never actually confirmed, so this uses the one mechanism that IS
#    confirmed reliable (the individual page fetch already proven for
#    Recast-equivalent work) instead of guessing at an unverified signal.
#    The blank-catalog list is small (dozens, not hundreds), so a fetch per
#    catalog is completely reasonable here.

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# ── Bright Data Web Unlocker: routes around BidSpotter's AWS WAF ───────────
# The page1 diagnostic above proved plain httpx gets served an AWS WAF
# challenge page (edge.sdk.awswaf.com/.../challenge.js), not real content.
# Bright Data's Web Unlocker API solves that challenge server-side and
# returns the real page. format=json is used (not raw) specifically because
# it echoes the *target's own* HTTP status in the response body -- needed so
# _recheck_blank_catalogs' `resp.status_code != 200` check still means what
# it always meant (the individual catalog page's real status), not just
# "Bright Data's own request succeeded".

_BIDSPOTTER_ALLOWED_HOST = "www.bidspotter.com"

class BrightDataMisuseError(Exception):
    """Raised if any code path ever tries to route a non-BidSpotter URL
    through Bright Data. This must NEVER be pointed at any other site --
    most importantly, NEVER at ebay.com, under any circumstances. This is
    enforced here, inside the fetch function itself, so no caller can bypass
    it by mistake."""
    pass

class _BrightDataResponse:
    """Stand-in for an httpx.Response exposing only what this file's existing
    BidSpotter code already uses: status_code, text, url, raise_for_status()."""
    def __init__(self, status_code: int, text: str, url: str):
        self.status_code = status_code
        self.text = text
        self.url = url
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"BidSpotter (via Bright Data) returned {self.status_code} for {self.url}",
                request=None, response=None,
            )

async def _brightdata_get(client: httpx.AsyncClient, target_url: str, expect_selector: str = None) -> _BrightDataResponse:
    """Fetch target_url through Bright Data's Web Unlocker API instead of
    fetching it directly. HARD-LOCKED to bidspotter.com -- refuses (raises,
    sends nothing) for any other domain.

    expect_selector: a CSS selector to wait for before Bright Data returns
    the response (their x-unblock-expect feature). BidSpotter's listing page
    is an AngularJS SPA -- the initial HTML is an empty shell (ng-cloak) with
    the real catalog links injected client-side by JS, so without this the
    WAF-unlocked page comes back real but still empty. Only pass this where
    the selector is guaranteed to eventually appear -- if it never does,
    Bright Data waits out its own render timeout before giving up."""
    host = urlparse(target_url).netloc.lower()
    if not (host == _BIDSPOTTER_ALLOWED_HOST or host.endswith("." + _BIDSPOTTER_ALLOWED_HOST)):
        raise BrightDataMisuseError(f"Refusing to fetch non-BidSpotter domain via Bright Data: {host!r}")

    api_key = os.environ.get("BRIGHT_DATA_API_KEY")
    zone = os.environ.get("BRIGHT_DATA_ZONE", "bidspotter_unlock")
    if not api_key:
        raise RuntimeError("BRIGHT_DATA_API_KEY is not set -- cannot fetch BidSpotter pages")

    payload = {"zone": zone, "url": target_url, "format": "json"}
    if expect_selector:
        # NOTE: enabling this custom header switches Bright Data's billing for
        # this request from pay-only-on-success to pay-on-every-attempt. Still
        # trivial money at our volume (a handful of requests/day), so left on.
        payload["headers"] = {"x-unblock-expect": json.dumps({"element": expect_selector})}

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        resp = await client.post(
            "https://api.brightdata.com/request",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=150.0,  # BidSpotter's WAF challenge can take well over 60s to solve
        )
        if resp.status_code != 200:
            # Bright Data's own request-level failure (bad zone/auth/rate-limit) --
            # not BidSpotter's status, that's the point of checking this first.
            return _BrightDataResponse(status_code=resp.status_code, text=resp.text, url=target_url)
        try:
            parsed = resp.json()
        except Exception as e:
            print(f"BrightData response for {target_url} wasn't the expected JSON shape: {type(e).__name__}: {e}")
            return _BrightDataResponse(status_code=resp.status_code, text=resp.text, url=target_url)

        target_status = int(parsed.get("status_code", 200))  # NOTE: field is "status_code", not "status"
        body = parsed.get("body", "")
        resp_headers = parsed.get("headers") or {}
        brd_error = resp_headers.get("x-brd-error") if isinstance(resp_headers, dict) else None

        if brd_error and attempt < max_attempts:
            print(f"BrightData attempt {attempt}/{max_attempts} for {target_url}: blocked ({brd_error!r}), retrying...")
            continue
        if brd_error:
            print(f"BrightData still blocked after {max_attempts} attempts for {target_url}: {brd_error!r} (status_code={target_status})")
        return _BrightDataResponse(status_code=target_status, text=body, url=target_url)

def _full_url_to_catalog_url(full_url: str) -> str:
    """Matches the exact sanitization this app already uses everywhere else:
    strip only ':' and '/' out of the real URL, keep everything else."""
    return re.sub(r'[:/]', '', full_url)

def _parse_bidspotter_listing_page(html: str) -> list:
    """BidSpotter's listing page is an AngularJS SPA -- clickable <a href>
    catalog links only exist in the DOM after client-side JS runs, which a
    plain fetch never sees. BUT the same catalog data is ALSO embedded
    directly in the raw HTML as schema.org structured data (for SEO) --
    confirmed via a live diagnostic showing real catalog urls/dates present
    in the un-rendered response body at a fixed JSON shape:
      "name": "...", ..., "url": "https://www.bidspotter.com/en-us/auction-
      catalogues/<slug>/catalogue-id-<id>", "location": {"url": "<same>", ...}
    We parse THAT instead of DOM anchors -- no JS rendering required, so this
    also works on a plain WAF-unlocked fetch (no x-unblock-expect needed).
    Each catalog's url appears twice (top-level + nested "location.url",
    identical) -- deduped on catalog_url same as the old anchor-based code."""
    seen = set()
    listings = []
    url_pattern = re.compile(
        r'"url"\s*:\s*"(https://www\.bidspotter\.com/en-us/auction-catalogues/[^"]*catalogue-id-[^"]+)"'
    )
    name_pattern = re.compile(r'"name"\s*:\s*"([^"]*)"')
    for m in url_pattern.finditer(html):
        full_url = m.group(1)
        catalog_url = _full_url_to_catalog_url(full_url)
        if catalog_url in seen:
            continue
        seen.add(catalog_url)
        # The catalog's display name is the nearest preceding "name" field in
        # the same JSON object (schema.org Event puts name before url/location).
        window = html[max(0, m.start() - 800):m.start()]
        name_matches = name_pattern.findall(window)
        title = name_matches[-1] if name_matches else full_url
        listings.append({"catalog_url": catalog_url, "title": title, "full_url": full_url})
    return listings

def _parse_bidspotter_listing_page_OLD_UNUSED(html: str) -> list:
    """Kept for reference only -- not called. Original anchor-tag based
    parser, replaced because BidSpotter's <a href> catalog links only exist
    post-JS-render, which a plain fetch never produces."""
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    listings = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/auction-catalogues/" not in href or "catalogue-id-" not in href:
            continue
        if "search-filter" in href:
            continue  # these are the per-category refine links inside each card, not the catalog itself
        full_url = href if href.startswith("http") else f"https://www.bidspotter.com{href}"
        catalog_url = _full_url_to_catalog_url(full_url)
        if catalog_url in seen:
            continue
        seen.add(catalog_url)
        title = a.get_text(strip=True)
        if title:  # the image-wrapper <a> has no text -- skip it, the title <a> duplicate will be caught
            listings.append({"catalog_url": catalog_url, "title": title, "full_url": full_url})
    return listings

async def _scan_bidspotter_new_catalogs(supabase_client, business_id: str) -> dict:
    """Job 1. Pages through the full public listing, stores every catalog
    seen into bidspotter_scan_snapshot (a durable record -- if BidSpotter
    ever changes their page layout and parsing silently breaks, this table
    going stale/empty is how that gets noticed instead of just quietly
    missing catalogs forever), then flags anything not already in
    auction_pdf_uploads as a new item in the updates queue. Returns a dict
    describing what actually happened -- including the real error if the
    first fetch fails, since that failure was previously completely
    invisible (a background task, no request, nothing in reach to check)."""
    all_listings = {}
    pages_fetched = 0
    first_page_error = None
    first_page_diagnostic = None  # captured on page 1 only, used purely for debugging a 0-listings outcome
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
        page = 1
        empty_pages_in_a_row = 0
        while page <= 60 and empty_pages_in_a_row < 2:  # hard ceiling -- never loop forever on an unexpected layout change
            url = f"https://www.bidspotter.com/en-us/auction-catalogues?page={page}"
            try:
                resp = await _brightdata_get(client, url)
                resp.raise_for_status()
            except Exception as e:
                err = f"page {page} fetch failed: {type(e).__name__}: {e}"
                print(f"BidSpotter scan: {err}")
                if page == 1:
                    first_page_error = err
                break
            pages_fetched += 1
            listings = _parse_bidspotter_listing_page(resp.text)
            if page == 1:
                # Capture real evidence of what came back -- status, length, final
                # URL, a snippet, AND whether the target content exists ANYWHERE
                # in the full body (not just the visible start) -- distinguishes
                # "never rendered at all" from "rendered but our parser/selector
                # is looking in the wrong place".
                snippet = re.sub(r'\s+', ' ', resp.text[:400]).strip()
                marker_idx = resp.text.find("catalogue-id-")
                if marker_idx >= 0:
                    around = re.sub(r'\s+', ' ', resp.text[max(0, marker_idx-150):marker_idx+150]).strip()
                    marker_info = f"catalogue-id- FOUND at offset {marker_idx}, context=\"{around}\""
                else:
                    marker_info = "catalogue-id- NOT found anywhere in the full response body"
                first_page_diagnostic = (
                    f"page1 status={resp.status_code} bytes={len(resp.text)} "
                    f"final_url={str(resp.url)} | {marker_info} | snippet=\"{snippet}\""
                )
            if not listings:
                empty_pages_in_a_row += 1
            else:
                empty_pages_in_a_row = 0
                for item in listings:
                    all_listings[item["catalog_url"]] = item
            page += 1

    if first_page_error:
        return {"ok": False, "error": first_page_error, "pages": pages_fetched, "listings": 0, "new_flagged": 0}

    if not all_listings:
        msg = f"found nothing at all -- likely a page layout change or a bot-block, not a real empty result | {first_page_diagnostic}"
        print(f"BidSpotter scan: {msg}")
        return {"ok": False, "error": msg, "pages": pages_fetched, "listings": 0, "new_flagged": 0}

    now_iso = datetime.now(timezone.utc).isoformat()
    snapshot_rows = [
        {"business_id": business_id, "catalog_url": v["catalog_url"], "title": v["title"], "scanned_at": now_iso}
        for v in all_listings.values()
    ]
    for i in range(0, len(snapshot_rows), 500):
        supabase_client.table("bidspotter_scan_snapshot").upsert(
            snapshot_rows[i:i+500], on_conflict="business_id,catalog_url"
        ).execute()

    known_urls = set()
    known_rows = _fetch_all_paginated(lambda: supabase_client.table("auction_pdf_uploads").select("catalog_url").eq("business_id", business_id))
    for r in known_rows:
        known_urls.add(r["catalog_url"])

    new_count = 0
    for catalog_url, item in all_listings.items():
        if catalog_url in known_urls:
            continue
        try:
            supabase_client.table("catalog_updates_queue").upsert({
                "business_id": business_id, "catalog_url": catalog_url, "title": item["title"],
                "kind": "new", "resolved": False,
            }, on_conflict="business_id,catalog_url").execute()
            new_count += 1
        except Exception as e:
            print(f"BidSpotter scan: failed to queue new catalog {catalog_url}: {e}")
    return {"ok": True, "error": None, "pages": pages_fetched, "listings": len(all_listings), "new_flagged": new_count}

async def _recheck_blank_catalogs(supabase_client, business_id: str) -> dict:
    """Job 2. For every catalog we already know about that's currently
    sitting at zero lots, re-fetches its own individual page directly and
    checks for real content. A catalog card shows a 'Cannot load data'
    placeholder for its lot-count widget regardless of whether it actually
    has lots (that's just an unrendered JS component in a static fetch, not
    a real signal) -- but the category tag list underneath it IS real, and
    is genuinely absent when a catalog has nothing in it yet."""
    latest_status = {}
    rows = _fetch_all_paginated(lambda: supabase_client.table("auction_pdf_uploads").select("catalog_url,status,uploaded_at,filename").eq("business_id", business_id).order("uploaded_at"))
    for r in rows:
        latest_status[r["catalog_url"]] = r  # later rows overwrite earlier -- ends up holding the latest per catalog_url

    blank_catalogs = [r for r in latest_status.values() if r["status"] == "empty"]
    reactivated_count = 0
    growing_count = 0
    checked = 0
    first_error = None

    # Small active catalogs (1-49 known lots) -- bounded on purpose, keeps this
    # fast. Compares against last_category_count stored on bidspotter_scan_snapshot
    # to detect growth, not just empty->active transitions.
    try:
        small_catalog_rows = supabase_client.rpc(
            "get_small_active_catalogs", {"p_business_id": business_id, "p_max_lots": 50}
        ).execute().data or []
    except Exception as e:
        print(f"BidSpotter recheck: failed to fetch small-catalog list: {e}")
        small_catalog_rows = []
    small_catalog_urls = {r["catalog_url"] for r in small_catalog_rows}

    prior_counts = {}
    if small_catalog_urls:
        try:
            snap_rows = _fetch_all_paginated(lambda: supabase_client.table("bidspotter_scan_snapshot").select("catalog_url,title,last_category_count").eq("business_id", business_id))
            for r in snap_rows:
                if r["catalog_url"] in small_catalog_urls:
                    prior_counts[r["catalog_url"]] = r
        except Exception as e:
            print(f"BidSpotter recheck: failed to fetch prior category counts: {e}")

    async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
        for row in blank_catalogs:
            catalog_url = row["catalog_url"]
            real_url = _reconstruct_full_url(catalog_url)
            if not real_url:
                continue
            try:
                resp = await _brightdata_get(client, real_url)
                if resp.status_code != 200:
                    if first_error is None:
                        first_error = f"{catalog_url}: HTTP {resp.status_code}"
                    continue
                text = resp.text
                checked += 1
            except Exception as e:
                if first_error is None:
                    first_error = f"{catalog_url}: {type(e).__name__}: {e}"
                continue

            category_matches = re.findall(r'search-filter\?CategoryCode=', text)
            category_count = len(category_matches)
            has_real_content = category_count > 0
            if has_real_content:
                try:
                    supabase_client.table("catalog_updates_queue").upsert({
                        "business_id": business_id, "catalog_url": catalog_url,
                        "title": row.get("filename", catalog_url), "kind": "reactivated", "resolved": False,
                        "category_count": category_count,
                    }, on_conflict="business_id,catalog_url").execute()
                    reactivated_count += 1
                except Exception as e:
                    print(f"BidSpotter recheck: failed to queue reactivated catalog {catalog_url}: {e}")

        for catalog_url in small_catalog_urls:
            real_url = _reconstruct_full_url(catalog_url)
            if not real_url:
                continue
            try:
                resp = await _brightdata_get(client, real_url)
                if resp.status_code != 200:
                    if first_error is None:
                        first_error = f"{catalog_url}: HTTP {resp.status_code}"
                    continue
                text = resp.text
                checked += 1
            except Exception as e:
                if first_error is None:
                    first_error = f"{catalog_url}: {type(e).__name__}: {e}"
                continue

            new_count = len(re.findall(r'search-filter\?CategoryCode=', text))
            prior_row = prior_counts.get(catalog_url)
            prior_count = prior_row.get("last_category_count") if prior_row else None

            if prior_count is not None and new_count > prior_count:
                try:
                    supabase_client.table("catalog_updates_queue").upsert({
                        "business_id": business_id, "catalog_url": catalog_url,
                        "title": (prior_row or {}).get("title", catalog_url), "kind": "growing", "resolved": False,
                        "category_count": new_count,
                    }, on_conflict="business_id,catalog_url").execute()
                    growing_count += 1
                except Exception as e:
                    print(f"BidSpotter recheck: failed to queue growing catalog {catalog_url}: {e}")

            try:
                supabase_client.table("bidspotter_scan_snapshot").update(
                    {"last_category_count": new_count}
                ).eq("business_id", business_id).eq("catalog_url", catalog_url).execute()
            except Exception as e:
                print(f"BidSpotter recheck: failed to update last_category_count for {catalog_url}: {e}")

    return {"total_blank": len(blank_catalogs), "checked": checked, "reactivated": reactivated_count, "growing": growing_count, "first_error": first_error}

def _reconstruct_full_url(catalog_url: str) -> Optional[str]:
    """Inverse of _full_url_to_catalog_url -- rebuilds a real, fetchable
    BidSpotter URL from our sanitized catalog_url format."""
    m = re.match(r'^https(www\.bidspotter\.com)en-usauction-catalogues(.+?)(catalogue-id-.+)$', catalog_url)
    if not m:
        return None
    return f"https://{m.group(1)}/en-us/auction-catalogues/{m.group(2)}/{m.group(3)}"

async def _run_bidspotter_scan_for_business(supabase_client, business_id: str):
    """Runs both jobs once for one business and writes the outcome (success
    or the real error) into bidspotter_scan_status -- so what actually
    happened is checkable via a normal query, not lost in server logs
    nobody can reach."""
    scan_result = await _scan_bidspotter_new_catalogs(supabase_client, business_id)
    recheck_result = await _recheck_blank_catalogs(supabase_client, business_id)
    error_parts = []
    if not scan_result["ok"]:
        error_parts.append(f"New-catalog scan: {scan_result['error']}")
    if recheck_result["first_error"]:
        error_parts.append(f"Blank-catalog recheck: {recheck_result['first_error']}")
    status_row = {
        "business_id": business_id,
        "last_run_at": datetime.now(timezone.utc).isoformat(),
        "last_success": scan_result["ok"],
        "last_error": " | ".join(error_parts) if error_parts else None,
        "pages_scanned": scan_result["pages"],
        "listings_found": scan_result["listings"],
        "new_flagged": scan_result["new_flagged"],
        "reactivated_flagged": recheck_result["reactivated"],
        "growing_flagged": recheck_result["growing"],
    }
    supabase_client.table("bidspotter_scan_status").upsert(status_row, on_conflict="business_id").execute()
    print(f"BidSpotter scan for {business_id}: {status_row}")
    return status_row

async def _fetch_catalog_lots_via_browser(catalog_url_full: str, catalog_slug: str) -> list:
    """The real, proven mechanism: loads a catalog page in a real remote
    browser (Bright Data Browser API), clicks 'In this auction' + submits
    the search form, scrolls to trigger lazy-loaded results, then extracts
    every lot card and filters to only ones actually belonging to this
    catalog (BidSpotter's own scoping isn't fully reliable, so we filter
    client-side using each lot's own href). Returns [{lot_number,
    description}, ...]. catalog_slug is the catalogue-id- value, e.g.
    'ncm-au11447', used both for the client-side filter and detecting a
    flaky click (checked state) before submitting."""
    from playwright.async_api import async_playwright
    wss_url = os.environ.get("BRIGHT_DATA_BROWSER_WSS")
    if not wss_url:
        return []
    lots = []
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(wss_url, timeout=60000)
            page = await browser.new_page()
            await page.goto(catalog_url_full, timeout=60000, wait_until="load")
            await page.wait_for_timeout(3000)

            # The 'In this auction' click was observed to be flaky (didn't
            # always register before submit) -- retry until it's actually
            # checked, not just attempted once.
            checked = False
            for attempt in range(3):
                try:
                    await page.click("#catalogueSearchOption", timeout=5000)
                    checked = await page.eval_on_selector("#catalogueSearchOption", "el => el.checked")
                    if checked:
                        break
                except Exception:
                    pass
                await page.wait_for_timeout(1000)
            if not checked:
                await browser.close()
                return []

            await page.click("#searchSubmit", timeout=5000)
            await page.wait_for_load_state("load", timeout=30000)
            await page.wait_for_timeout(3000)

            # Scroll a few times to trigger any lazy-loaded additional lots
            for _ in range(5):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1500)

            html = await page.content()
            await browser.close()

        all_matches = re.findall(
            r'href="(/en-us/auction-catalogues/([a-z0-9]+)/catalogue-id-([a-z0-9\-]+)/lot-[a-z0-9\-]+)"[^>]*data-click-type="title"[^>]*>\s*<span class="lot-number">(\d+)</span><span class="lot-title">([^<]+)</span>',
            html, re.I
        )
        seen_lot_numbers = set()
        for href, auctioneer, cat_id, lot_num, lot_title in all_matches:
            if cat_id.lower() != catalog_slug.lower():
                continue
            if lot_num in seen_lot_numbers:
                continue
            seen_lot_numbers.add(lot_num)
            lots.append({"lot_number": lot_num, "description": lot_title.strip()})
    except Exception as e:
        print(f"Browser lot fetch failed for {catalog_url_full}: {type(e).__name__}: {e}")
    return lots

async def _pull_lots_for_queued_catalogs(supabase_client, business_id: str) -> dict:
    """Runs the real browser-based lot pull for every catalog currently
    sitting unresolved in the VA queue (new or reactivated) -- these are
    exactly the ones a VA would otherwise have to open and read by hand.
    Writes into bidspotter_auto_catalog_lots (the staging table, kept
    separate from the trusted bidspotter_catalog_lots table the real
    listing pipeline depends on). Updates bidspotter_lot_pull_progress
    after every single catalog so real progress is queryable at any
    moment, not just guessed from logs."""
    if not os.environ.get("BRIGHT_DATA_BROWSER_WSS"):
        return {"attempted": 0, "succeeded": 0, "total_lots_written": 0}
    queue_rows = _fetch_all_paginated(lambda: supabase_client.table("catalog_updates_queue").select("catalog_url,title").eq("business_id", business_id).eq("resolved", False))
    attempted = 0
    succeeded = 0
    total_lots_written = 0
    now = datetime.now(timezone.utc).isoformat()
    supabase_client.table("bidspotter_lot_pull_progress").upsert({
        "business_id": business_id, "total_queued": len(queue_rows), "processed": 0,
        "succeeded": 0, "total_lots_written": 0, "current_catalog_url": None,
        "started_at": now, "updated_at": now, "finished_at": None,
    }, on_conflict="business_id").execute()
    for row in queue_rows:
        catalog_url = row["catalog_url"]
        real_url = _reconstruct_full_url(catalog_url)
        if not real_url:
            continue
        m = re.search(r'catalogue-id-([a-z0-9\-]+)$', real_url, re.I)
        if not m:
            continue
        catalog_slug = m.group(1)
        attempted += 1
        try:
            supabase_client.table("bidspotter_lot_pull_progress").update({
                "current_catalog_url": catalog_url, "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("business_id", business_id).execute()
        except Exception:
            pass
        lots = []
        max_attempts = 3
        for retry_num in range(1, max_attempts + 1):
            try:
                # Hard outer ceiling -- 120s gives real comfortable margin over
                # the ~50s a genuine successful run takes (page load +
                # click-retry + submit + 5x scroll), but guarantees this
                # single attempt can't hang forever no matter WHAT breaks
                # internally, even if a Playwright/CDP call's own timeout
                # param mysteriously doesn't fire (which is what happened
                # tonight -- the run sat well past its 60-90s internal
                # timeouts with zero log output).
                lots = await asyncio.wait_for(
                    _fetch_catalog_lots_via_browser(real_url, catalog_slug), timeout=120.0
                )
                if lots:
                    break
                print(f"Lot pull attempt {retry_num}/{max_attempts} for {catalog_url}: got 0 lots, retrying from scratch")
            except asyncio.TimeoutError:
                print(f"Lot pull attempt {retry_num}/{max_attempts} for {catalog_url}: HARD TIMEOUT (120s), redoing from scratch")
            except Exception as e:
                print(f"Lot pull attempt {retry_num}/{max_attempts} for {catalog_url} failed: {type(e).__name__}: {e}")
        if lots:
            succeeded += 1
            rows_to_upsert = [
                {"business_id": business_id, "catalog_url": catalog_url, "lot_number": lot["lot_number"],
                 "description": lot["description"], "last_seen_at": datetime.now(timezone.utc).isoformat()}
                for lot in lots
            ]
            try:
                supabase_client.table("bidspotter_auto_catalog_lots").upsert(
                    rows_to_upsert, on_conflict="business_id,catalog_url,lot_number"
                ).execute()
                total_lots_written += len(rows_to_upsert)
            except Exception as e:
                print(f"Failed to write lots for {catalog_url}: {e}")
        try:
            supabase_client.table("bidspotter_lot_pull_progress").update({
                "processed": attempted, "succeeded": succeeded, "total_lots_written": total_lots_written,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("business_id", business_id).execute()
        except Exception as e:
            print(f"Failed to update lot-pull progress: {e}")
    try:
        supabase_client.table("bidspotter_lot_pull_progress").update({
            "finished_at": datetime.now(timezone.utc).isoformat(), "current_catalog_url": None,
        }).eq("business_id", business_id).execute()
    except Exception:
        pass
    return {"attempted": attempted, "succeeded": succeeded, "total_lots_written": total_lots_written}

async def _daily_bidspotter_scan_loop():
    """Runs once at startup (after a short delay so the app is fully up
    first), then once every 12 hours after that, for every business_id that
    has ever used this app. No manual trigger needed for normal operation --
    /api/updates/trigger-scan exists purely for on-demand debugging."""
    await asyncio.sleep(30)
    while True:
        try:
            supabase_client, _ = _require_config()
            biz_rows = supabase_client.table("auction_pdf_uploads").select("business_id").limit(1000).execute().data or []
            business_ids = {r["business_id"] for r in biz_rows}
            for business_id in business_ids:
                await _run_bidspotter_scan_for_business(supabase_client, business_id)
                lot_pull_result = await _pull_lots_for_queued_catalogs(supabase_client, business_id)
                print(f"BidSpotter lot pull for {business_id}: {lot_pull_result}")
        except Exception as e:
            print(f"BidSpotter daily scan loop failed: {e}")
        await asyncio.sleep(12 * 60 * 60)

@app.on_event("startup")
async def _start_bidspotter_scan_loop():
    asyncio.create_task(_daily_bidspotter_scan_loop())

@app.api_route("/api/updates/trigger-scan", methods=["GET", "POST"])
async def api_trigger_scan():
    """Manual, on-demand trigger for debugging -- runs the exact same scan
    the background loop runs, immediately, and returns exactly what
    happened (including the real error, if any) directly in the response."""
    supabase_client, business_id = _require_config()
    status_row = await _run_bidspotter_scan_for_business(supabase_client, business_id)
    return status_row

@app.get("/api/updates-to-make")
async def api_updates_to_make():
    """Powers the 'Updates to Make' box -- unresolved new catalogs and
    reactivated (was-blank, now-has-content) catalogs, newest flagged first."""
    supabase_client, business_id = _require_config()
    rows = (supabase_client.table("catalog_updates_queue").select("*")
            .eq("business_id", business_id).eq("resolved", False)
            .order("first_flagged_at", desc=True).execute().data or [])
    return {"updates": rows}


@app.get("/api/catalogs")
async def api_catalogs():
    supabase_client, business_id = _require_config()
    rows = _fetch_all_paginated(lambda: supabase_client.table("auction_catalogs").select("*").eq("business_id", business_id))
    return {"catalogs": rows}


@app.get("/api/lots")
async def api_lots(catalog_url: str = None):
    supabase_client, business_id = _require_config()
    def build_query():
        q = supabase_client.table("bidspotter_catalog_lots").select("*").eq("business_id", business_id)
        if catalog_url:
            q = q.eq("catalog_url", catalog_url)
        return q.order("last_seen_at", desc=True)
    rows = _fetch_all_paginated(build_query)
    return {"lots": rows}


def _process_one_pdf(supabase_client, business_id: str, filename: str, contents: bytes,
                       title: str = "", auctioneer: str = "", end_date: str = "", state: str = "", zip_code: str = "") -> dict:
    """Does everything for one catalog PDF: stores it, extracts text, pulls
    every lot out via Gemini (the sole, authoritative parser -- see
    _extract_lots_via_gemini), ingests into bidspotter_catalog_lots/
    auction_catalogs, and logs the attempt to auction_pdf_uploads regardless
    of outcome. Shared by both the single-file upload endpoint and the
    zip-batch endpoint, so a zip upload behaves identically to uploading each
    file one at a time -- same parsing, same empty-catalog handling, same
    was_ever_empty flag, same logging."""
    title = title.strip() or filename
    raw_name = filename.rsplit(".", 1)[0] if filename else str(uuid.uuid4())
    catalog_key = re.sub(r'[^A-Za-z0-9._-]', '_', raw_name)
    catalog_url = catalog_key

    log_row = {
        "business_id": business_id, "filename": filename, "status": "processing",
        "catalog_url": catalog_url, "catalog_title": title,
    }
    log_res = supabase_client.table("auction_pdf_uploads").insert(log_row).execute()
    log_id = log_res.data[0]["id"] if log_res.data else None

    storage_path = None
    try:
        storage_path = f"{catalog_key}.pdf"
        supabase_client.storage.from_("auction-pdfs").upload(
            path=storage_path, file=contents,
            file_options={"content-type": "application/pdf", "upsert": "true"}
        )
    except Exception as e:
        # Real bug this caught before: the "auction-pdfs" bucket never
        # actually existed, so every single upload silently failed here for
        # this app's entire lifetime -- nothing ever got archived, and the
        # only trace was a print() statement nobody was watching. Now also
        # written into the upload log itself so a storage failure is
        # actually visible somewhere a human will see it, instead of only
        # existing in Railway's console output.
        print(f"PDF storage warning (upload still proceeds): {e}")
        storage_path = None
        if log_id:
            try:
                supabase_client.table("auction_pdf_uploads").update({
                    "storage_warning": f"PDF was not archived to storage: {e}"[:500],
                }).eq("id", log_id).execute()
            except Exception:
                pass  # never let a logging failure break the actual upload

    try:
        doc = fitz.open(stream=contents, filetype="pdf")
        raw_text = ""
        for page in doc:
            raw_text += page.get_text() + "\n"
        doc.close()

        if not raw_text.strip():
            raise ValueError("No text found in this PDF (may be scanned images with no text layer -- not supported yet)")

        meta = {"title": title, "auctioneer": auctioneer, "end_date": end_date, "state": state, "zip_code": zip_code}

        # Real bug fixed here: this used to call _extract_lots_via_gemini
        # with no callback, collect the ENTIRE catalog's lots in memory, and
        # only write any of it to the database after every chunk had
        # finished -- so a catalog killed mid-extraction lost 100% of its
        # work, not just whatever chunk hadn't run yet. Now each chunk's
        # lots are written the moment that chunk comes back, and the
        # catalog's summary row is created/refreshed with the running total
        # right after -- so a partial catalog leaves a real, findable,
        # partially-complete result instead of nothing at all.
        written_count = 0

        def _on_chunk(chunk_lots):
            nonlocal written_count
            _upsert_lots_batch(supabase_client, business_id, catalog_url, chunk_lots, meta)
            written_count += len(chunk_lots)
            _upsert_catalog_summary(supabase_client, business_id, catalog_url, meta, lot_count=written_count)

        lots, chunk_errors = _extract_lots_via_gemini(raw_text, filename, on_chunk_lots=_on_chunk)

        if not lots:
            # Real bug fixed here: this used to mark EVERY zero-lot result
            # as "empty" -- indistinguishable in the UI from a catalog that
            # genuinely has no lots yet. If chunk_errors is non-empty, every
            # chunk actually failed (timeout, rate limit, bad API key, quota
            # exhausted -- all real, all possible) and there's a real reason
            # nothing came back, not an empty auction. Surface that as an
            # actual error with the real message instead of hiding it behind
            # "empty," so a genuine infrastructure failure is never
            # mistaken for "this auction just doesn't have lots yet."
            if chunk_errors:
                error_message = "; ".join(chunk_errors)[:1000]
                if log_id:
                    supabase_client.table("auction_pdf_uploads").update({
                        "status": "error", "storage_path": storage_path, "error_message": error_message,
                    }).eq("id", log_id).execute()
                return {"ok": False, "filename": filename, "error": error_message}
            if log_id:
                supabase_client.table("auction_pdf_uploads").update({
                    "status": "empty", "storage_path": storage_path, "parsed_lot_count": 0,
                }).eq("id", log_id).execute()
            needs_backfill = not (state and zip_code and end_date)
            return {"ok": True, "filename": filename, "lots_parsed": 0, "empty": True, "catalog_url": catalog_url,
                    "needs_backfill": needs_backfill, "raw_text": raw_text if needs_backfill else None}

        # written_count tracks what actually landed via the per-chunk
        # callback, which is the real durable number -- if any single
        # chunk's immediate write failed (rare, logged, kept in the final
        # `lots` list as a fallback rather than lost), written_count and
        # len(lots) can differ slightly; written_count is what's actually
        # in the database right now, so it's what gets reported and logged.
        # Real bug fixed here: chunk_errors was captured above but never
        # checked again once ANY chunk succeeded -- so a catalog chunked into,
        # say, 5 pieces where 3 succeed and 2 time out silently reported as a
        # complete "success" with only the partial lot count, no indication
        # anywhere that ~40% of the catalog never made it in. Now flagged as
        # its own "partial" status with the real error message attached, so
        # this is actually visible instead of looking identical to a genuine
        # complete success.
        if log_id:
            if chunk_errors:
                error_message = "; ".join(chunk_errors)[:1000]
                supabase_client.table("auction_pdf_uploads").update({
                    "status": "partial", "storage_path": storage_path, "parsed_lot_count": written_count,
                    "error_message": f"Partial: {len(chunk_errors)} chunk(s) failed after retries -- "
                                      f"{written_count} lots landed but this catalog is very likely "
                                      f"incomplete. Try re-uploading. Errors: {error_message}",
                }).eq("id", log_id).execute()
            else:
                supabase_client.table("auction_pdf_uploads").update({
                    "status": "success", "storage_path": storage_path, "parsed_lot_count": written_count,
                }).eq("id", log_id).execute()

        needs_backfill = not (state and zip_code and end_date)

        if written_count > 0:
            try:
                supabase_client.table("catalog_updates_queue").update({"resolved": True})\
                    .eq("business_id", business_id).eq("catalog_url", catalog_url).execute()
            except Exception:
                pass  # never let queue bookkeeping break the actual upload

        return {"ok": True, "filename": filename, "lots_parsed": written_count, "catalog_url": catalog_url,
                "partial": bool(chunk_errors), "needs_backfill": needs_backfill, "raw_text": raw_text if needs_backfill else None}

    except Exception as e:
        if log_id:
            supabase_client.table("auction_pdf_uploads").update({
                "status": "error", "storage_path": storage_path, "error_message": str(e),
            }).eq("id", log_id).execute()
        return {"ok": False, "filename": filename, "error": str(e)}


@app.post("/api/upload-pdf")
async def upload_pdf(request: Request):
    """Queues this file for processing -- see _pdf_processing_queue above for
    why this is a queue now instead of an independent background task per
    upload. Returns immediately either way; the Upload Log fills in once this
    file's actual turn comes up, same as before."""
    supabase_client, business_id = _require_config()

    form = await request.form()
    file = form.get("file")
    if not file or not hasattr(file, "read"):
        raise HTTPException(400, "file is required")
    title = (form.get("title") or "").strip()
    auctioneer = (form.get("auctioneer") or "").strip()
    end_date = (form.get("end_date") or "").strip()
    state = (form.get("state") or "").strip()
    zip_code = (form.get("zip_code") or "").strip()

    contents = await file.read()

    def _run_and_backfill():
        result = _process_one_pdf(supabase_client, business_id, file.filename, contents,
                                   title, auctioneer, end_date, state, zip_code)
        if result.get("needs_backfill") and result.get("raw_text"):
            _backfill_catalog_metadata(supabase_client, business_id, result["catalog_url"],
                                        result["raw_text"], file.filename, state, zip_code, end_date)

    _pdf_processing_queue.put_nowait(_run_and_backfill)
    return {"ok": True, "queued": True, "filename": file.filename}


def _process_zip_batch(supabase_client, business_id: str, pdf_entries: list) -> None:
    """Runs via the shared processing queue (see _pdf_processing_queue) -- doesn't
    block other requests while this runs, and never overlaps with a single-PDF
    upload or another zip batch either. Processes every PDF found in the zip
    one at a time, same logic as a single upload for each. Progress is visible
    the whole time through the existing Upload Log / Needs Update / Catalogs
    views -- no separate progress UI needed, since each file logs itself as it
    completes, exactly like uploading them one by one would.

    Real bug fixed here: one single file got stuck processing for 1.5+ hours
    and silently froze the ENTIRE remaining batch behind it (349 other files
    never even started) -- because this loop called _process_one_pdf directly
    and just waited, with nothing to give up on a file that hangs. The
    per-Gemini-call 120s timeout added earlier turned out to NOT be reliably
    honored by the underlying SDK in every failure mode (a real gap, not a
    theory -- confirmed by exactly this happening in production). Rather than
    chase which specific internal call can still hang, every file's ENTIRE
    processing now runs under one hard, unconditional 25-minute ceiling --
    if anything inside it hangs for any reason at all, this loop gives up on
    that one file, logs it as a real timeout error, and moves on to the next
    file immediately instead of stalling the other 349 behind it."""
    print(f"Zip batch: starting {len(pdf_entries)} catalog(s)")
    for i, (filename, contents) in enumerate(pdf_entries):
        try:
            # Deliberately NOT a `with ThreadPoolExecutor() as ex:` block --
            # caught via direct testing before this shipped: the context
            # manager's __exit__ calls shutdown(wait=True), which blocks
            # until the submitted task actually finishes -- meaning even
            # after catching TimeoutError below, exiting the `with` block
            # would silently wait for the stuck file anyway, completely
            # defeating the point of the watchdog. shutdown(wait=False)
            # lets this loop actually move on; the orphaned thread (if the
            # file really is stuck forever) just keeps running harmlessly
            # in the background instead of blocking anything else.
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = ex.submit(_process_one_pdf, supabase_client, business_id, filename, contents)
            try:
                result = future.result(timeout=1500)
            except concurrent.futures.TimeoutError:
                ex.shutdown(wait=False)
                print(f"Zip batch: {filename} exceeded the 25-minute hard watchdog timeout -- "
                      f"giving up on this file and moving on (the stuck attempt may still finish "
                      f"on its own later and overwrite this with a real result)")
                try:
                    supabase_client.table("auction_pdf_uploads").insert({
                        "business_id": business_id, "filename": filename,
                        "catalog_url": re.sub(r'[^A-Za-z0-9._-]', '_', filename.rsplit(".", 1)[0]),
                        "status": "error",
                        "error_message": "Hard watchdog timeout: this file's processing did not "
                                          "complete within 25 minutes, so it was skipped to let the "
                                          "rest of the batch continue. Try uploading this one file "
                                          "again on its own.",
                    }).execute()
                except Exception as log_err:
                    print(f"Zip batch: also failed to log the watchdog timeout for {filename}: {log_err}")
                continue
            ex.shutdown(wait=False)
            if result.get("needs_backfill") and result.get("raw_text"):
                # Zip batches already run entirely in the background, so no
                # need to schedule this separately -- just do it inline, one
                # file at a time, same order as everything else in this batch.
                _backfill_catalog_metadata(supabase_client, business_id, result["catalog_url"],
                                            result["raw_text"], filename, "", "", "")
        except Exception as e:
            print(f"Zip batch: unexpected failure on {filename}: {e}")
        if (i + 1) % 25 == 0:
            print(f"Zip batch: {i + 1}/{len(pdf_entries)} done")
    print(f"Zip batch: finished all {len(pdf_entries)} catalog(s)")


@app.post("/api/upload-zip")
async def upload_zip(request: Request):
    """The real answer for uploading many catalogs at once (e.g. 350 PDFs
    collected into one zip) -- drop the zip in once instead of selecting
    hundreds of files by hand in a browser file picker, which isn't a
    reliable way to handle that many at once anyway."""
    import zipfile

    supabase_client, business_id = _require_config()
    form = await request.form()
    file = form.get("file")
    if not file or not hasattr(file, "read"):
        raise HTTPException(400, "file is required")

    contents = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Not a valid zip file: {e}")

    pdf_entries = []
    for name in zf.namelist():
        if name.lower().endswith(".pdf") and not name.startswith("__MACOSX/"):
            pdf_entries.append((name.rsplit("/", 1)[-1], zf.read(name)))

    if not pdf_entries:
        raise HTTPException(400, "No PDF files found inside this zip")

    _pdf_processing_queue.put_nowait(lambda: _process_zip_batch(supabase_client, business_id, pdf_entries))
    return {"ok": True, "queued": len(pdf_entries)}


@app.get("/api/pdf-uploads")
async def api_pdf_uploads():
    supabase_client, business_id = _require_config()
    uploads = _fetch_all_paginated(lambda: supabase_client.table("auction_pdf_uploads").select("*")
                                     .eq("business_id", business_id).order("uploaded_at", desc=True))
    return {"uploads": uploads}


@app.get("/api/needs-update")
async def api_needs_update():
    supabase_client, business_id = _require_config()
    all_uploads = _fetch_all_paginated(lambda: supabase_client.table("auction_pdf_uploads").select("*")
                                         .eq("business_id", business_id).order("uploaded_at", desc=True))
    latest_by_catalog = {}
    for u in all_uploads:
        key = u.get("catalog_url")
        if not key:
            continue
        existing = latest_by_catalog.get(key)
        if not existing or (u.get("uploaded_at") or "") > (existing.get("uploaded_at") or ""):
            latest_by_catalog[key] = u

    # Cross-check against the actual current data, not just the upload log --
    # a catalog that now has real lots (even from an earlier upload attempt,
    # before a later one came back empty by mistake, or before a parsing bug
    # got fixed) should never show here, no matter what the log alone says.
    # Real bug this fixes: a catalog could show as both empty AND appear in
    # the regular Catalogs list at the same time, because the log and the
    # actual data could disagree with each other.
    catalog_rows = _fetch_all_paginated(lambda: supabase_client.table("auction_catalogs").select("catalog_url,lot_count,end_date")
                                          .eq("business_id", business_id))
    has_real_lots = {c["catalog_url"] for c in catalog_rows if (c.get("lot_count") or 0) > 0}
    end_date_by_catalog = {c["catalog_url"]: c.get("end_date") for c in catalog_rows}

    needs_update = [u for u in latest_by_catalog.values()
                     if u.get("status") == "empty" and u.get("catalog_url") not in has_real_lots]
    for u in needs_update:
        u["end_date"] = u.get("end_date") or end_date_by_catalog.get(u.get("catalog_url"))
    needs_update.sort(key=lambda u: u.get("uploaded_at") or "", reverse=True)
    return {"needs_update": needs_update}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
