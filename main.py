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
    asyncio.create_task(_recover_orphaned_uploads())

async def _recover_orphaned_uploads():
    """Runs once, ~10s after startup. Finds any upload stuck at
    status='processing' with a real storage_path (meaning the file itself
    survived, per _create_upload_record's durability fix) -- these are
    uploads whose queued processing job was wiped by a container restart
    before it ran. Downloads the file back from storage and re-queues it
    for real, automatically. This is the recovery half of the durability
    fix: the file being safe in storage only matters if something also
    goes and finishes the job later, or it just sits there forever looking
    like 'processing' with no path to actually resolve."""
    await asyncio.sleep(10)
    try:
        supabase_client, business_id, error = _get_supabase_and_business_id()
        if error:
            return
        stuck = supabase_client.table("auction_pdf_uploads") \
            .select("id,filename,catalog_url,catalog_title,storage_path") \
            .eq("business_id", business_id).eq("status", "processing") \
            .not_.is_("storage_path", "null").execute().data or []
        if not stuck:
            return
        print(f"[upload-recovery] found {len(stuck)} orphaned upload(s) from a prior container restart, re-queuing")
        for row in stuck:
            try:
                file_bytes = supabase_client.storage.from_("auction-pdfs").download(row["storage_path"])
            except Exception as e:
                print(f"[upload-recovery] could not download {row['storage_path']} for recovery: {e}")
                continue
            record = {"log_id": row["id"], "catalog_url": row["catalog_url"], "storage_path": row["storage_path"]}
            filename = row["filename"]
            title = row.get("catalog_title") or filename

            def _job(fb=file_bytes, fn=filename, t=title, rec=record):
                _process_one_pdf(supabase_client, business_id, fn, fb, t, existing_record=rec)
            _pdf_processing_queue.put_nowait(_job)
            print(f"[upload-recovery] re-queued {filename} ({row['catalog_url']})")
    except Exception as e:
        print(f"[upload-recovery] recovery pass failed: {type(e).__name__}: {e}")


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
        try:
            # A fresh client (and thus a fresh TCP connection, no keep-alive
            # reuse) on EVERY attempt -- tonight's real hang was a REUSED
            # connection going silently dead on a later request after an
            # earlier one on the same connection succeeded fine. Neither
            # httpx's own timeout= nor an outer asyncio.wait_for caught it,
            # consistent with a truly stuck low-level socket read that
            # doesn't yield control back cleanly. A brand new connection per
            # attempt sidesteps the whole class of problem instead of trying
            # to detect/interrupt it after the fact.
            async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "Mozilla/5.0"}) as fresh_client:
                resp = await asyncio.wait_for(
                    fresh_client.post(
                        "https://api.brightdata.com/request",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json=payload,
                        timeout=150.0,  # BidSpotter's WAF challenge can take well over 60s to solve
                    ),
                    timeout=180.0,
                )
        except Exception as e:
            if attempt < max_attempts:
                print(f"BrightData attempt {attempt}/{max_attempts} for {target_url}: HARD TIMEOUT/error ({type(e).__name__}: {e}), retrying with a fresh connection...")
                continue
            print(f"BrightData still failing after {max_attempts} attempts for {target_url}: {type(e).__name__}: {e}")
            raise
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
    end_date_pattern = re.compile(r'"endDate"\s*:\s*"([^"]*)"')
    for m in url_pattern.finditer(html):
        full_url = m.group(1)
        catalog_url = _full_url_to_catalog_url(full_url)
        if catalog_url in seen:
            continue
        seen.add(catalog_url)
        # The catalog's display name and end date are the nearest preceding
        # fields in the same JSON object (schema.org Event puts name/dates
        # before url/location).
        window = html[max(0, m.start() - 800):m.start()]
        name_matches = name_pattern.findall(window)
        title = name_matches[-1] if name_matches else full_url
        end_date_matches = end_date_pattern.findall(window)
        end_date = end_date_matches[-1] if end_date_matches else None
        listings.append({"catalog_url": catalog_url, "title": title, "full_url": full_url, "end_date": end_date})
    return listings

def _parse_bidspotter_listing_page_OLD_UNUSED(html: str) -> list:
    """Kept for reference only -- not called directly by name, but its
    logic is now the active parser below (_parse_bidspotter_listing_page_
    from_dom) now that Job 1 loads real rendered pages via a browser."""
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

def _parse_bidspotter_listing_page_from_dom(html: str) -> list:
    """ACTIVE parser for the rebuilt Job 1. html here is page.content()
    from a real, JS-rendered browser session -- the country filter has
    genuinely been applied by Angular by this point, same as what a human
    sees in their own browser.

    Real bug fixed here: each listing card has TWO anchors with the same
    href -- an image-wrapper <a> (no text) and a title <a> (real text).
    The dedup check used to run BEFORE the title check, so whichever
    anchor BeautifulSoup happened to encounter first got permanently
    marked 'seen' -- if that was the empty-text image-wrapper (confirmed
    live: it comes first in the DOM), the real title anchor right after it
    was skipped as a duplicate before its text was ever read. This dropped
    EVERY single listing on the page (1,338 real anchors -> 0 extracted,
    confirmed live), not a partial miss. Now: track the best title seen so
    far PER catalog_url, so a later anchor with real text can fill in for
    an earlier title-less one instead of being blocked by it."""
    # REAL BUG FIXED HERE: this parser never captured an end date at all --
    # the returned listing dicts only ever had catalog_url/title/full_url,
    # so every downstream `item.get("end_date")` came back None
    # unconditionally. That's the actual reason 100% of queued rows show no
    # date, not a flaky per-catalog extraction issue. BidSpotter's rendered
    # card text includes the date directly (e.g. "Aug 13, 2026 10am ET" or
    # "Ends from Aug 18, 2026 10am CT") -- pulled via a generic text regex
    # over the anchor's nearest card-sized ancestor, same resilience
    # tradeoff already used elsewhere in this codebase (e.g. current-bid
    # scraping) since a bespoke CSS selector for this page isn't reliably
    # known and is easy to silently break on a layout tweak.
    date_pattern = re.compile(
        r'((?:Ends\s+(?:today|from)\s+)?[A-Z][a-z]{2}\s+\d{1,2},\s*\d{4}'
        r'(?:\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*[A-Za-z]{2,3})?)'
    )
    soup = BeautifulSoup(html, "html.parser")
    best_by_url = {}  # catalog_url -> {"full_url":..., "title": "" or real text, "end_date": ...}
    order = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/auction-catalogues/" not in href or "catalogue-id-" not in href:
            continue
        if "search-filter" in href:
            continue  # these are the per-category refine links inside each card, not the catalog itself
        full_url = href if href.startswith("http") else f"https://www.bidspotter.com{href}"
        catalog_url = _full_url_to_catalog_url(full_url)
        title = a.get_text(strip=True)
        end_date = None
        if title:
            node = a
            for _ in range(6):  # small hop count -- stay within this one card, not a whole page/section
                node = node.parent
                if node is None:
                    break
                card_text = node.get_text(" ", strip=True)
                if title in card_text:
                    m = date_pattern.search(card_text)
                    if m:
                        end_date = m.group(1).strip()
                        break
        if catalog_url not in best_by_url:
            best_by_url[catalog_url] = {"full_url": full_url, "title": title, "end_date": end_date}
            order.append(catalog_url)
        else:
            if title and not best_by_url[catalog_url]["title"]:
                # An earlier anchor for this same catalog had no text (the
                # image-wrapper) -- this later one does, so use it.
                best_by_url[catalog_url]["title"] = title
            if end_date and not best_by_url[catalog_url]["end_date"]:
                best_by_url[catalog_url]["end_date"] = end_date
    listings = [
        {"catalog_url": u, "title": best_by_url[u]["title"], "full_url": best_by_url[u]["full_url"],
         "end_date": best_by_url[u]["end_date"]}
        for u in order if best_by_url[u]["title"]  # still require SOME real title before including it
    ]
    return listings

def _update_live_activity(supabase_client, business_id: str, text: str) -> None:
    """Writes a real, human-readable 'what's happening right now' status --
    called continuously throughout Job 1/2/lot-pull (every page, every
    catalog), not just once at the end. This is what makes progress
    actually checkable in real time instead of silence for hours."""
    try:
        supabase_client.table("bidspotter_scan_status").upsert({
            "business_id": business_id, "current_activity": text,
            "activity_updated_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="business_id").execute()
    except Exception as e:
        print(f"Failed to update live activity: {e}")

async def _scan_bidspotter_new_catalogs(supabase_client, business_id: str) -> dict:
    """Job 1. Pages through the full US-filtered public listing, stores
    every catalog seen into bidspotter_scan_snapshot (a durable record --
    if BidSpotter ever changes their page layout and parsing silently
    breaks, this table going stale/empty is how that gets noticed instead
    of just quietly missing catalogs forever), then flags anything not
    already in auction_pdf_uploads as a new item in the updates queue.

    REBUILT to use a real Bright Data BROWSER session (Playwright,
    connect_over_cdp) instead of the plain-HTTP Web Unlocker fetch this
    used before. Root cause of Canadian catalogs (confirmed: Infinity
    Asset Solutions) leaking into a "US-only" queue: the previous version
    parsed schema.org JSON-LD embedded in the RAW, un-rendered HTML -- data
    that exists on the page regardless of whether the countryName=United
    States filter was ever actually applied. It was never proven that
    embedded blob respects the filter at all, and live evidence (the user
    directly compared BidSpotter's own filtered UI, which correctly shows
    ZERO Infinity Asset Solutions results for a Canadian catalog, against
    our snapshot, which had TEN of them) confirms it does not. This version
    loads the real page, lets Angular actually apply the filter client-side
    exactly as a human's browser would, and reads the real rendered DOM --
    the same class of result the user's own screenshot showed, not a raw
    data blob of unknown fidelity to the filter."""
    from playwright.sync_api import sync_playwright
    wss_url = os.environ.get("BRIGHT_DATA_BROWSER_WSS")
    if not wss_url:
        return {"ok": False, "error": "BRIGHT_DATA_BROWSER_WSS not set", "pages": 0, "listings": 0, "new_flagged": 0}

    def _sync_scan_work():
        """Runs entirely SYNCHRONOUSLY in a real OS thread (via
        run_in_executor below), using Playwright's SYNC api, not async.
        CONFIRMED live tonight: wrapping the ASYNC connect_over_cdp call in
        asyncio.wait_for did NOT stop a real hang -- watched it sit past its
        own 90s timeout with zero effect, meaning the hang wasn't yielding
        control back to the event loop for asyncio's cooperative
        cancellation to ever take hold. This is a genuinely different fix,
        not a bigger version of the same one: a real OS thread's blocked
        call can simply be ABANDONED by the caller (matching the exact
        pattern already proven in production for the zip-batch watchdog --
        see _process_zip_batch) without needing the blocked code to
        cooperate with cancellation at all."""
        result = {"all_listings": {}, "pages_fetched": 0, "first_page_error": None, "first_page_diagnostic": None}
        with sync_playwright() as pw:
            browser = None
            connect_error = None
            for connect_attempt in range(1, 4):
                try:
                    browser = pw.chromium.connect_over_cdp(wss_url, timeout=60000)
                    connect_error = None
                    break
                except Exception as e:
                    connect_error = e
                    print(f"BidSpotter scan: connect_over_cdp attempt {connect_attempt}/3 failed: {type(e).__name__}: {e}")
            if connect_error is not None:
                result["first_page_error"] = f"could not open a browser session after 3 attempts: {connect_error}"
                return result
            try:
                page_obj = browser.new_page()
                page_num = 1
                empty_pages_in_a_row = 0
                while page_num <= 60 and empty_pages_in_a_row < 2:  # hard ceiling -- never loop forever on an unexpected layout change
                    _update_live_activity(supabase_client, business_id, f"Job 1: fetching listing page {page_num}")
                    url = f"https://www.bidspotter.com/en-us/auction-catalogues/search-filter?countryName=United%20States&page={page_num}"
                    try:
                        page_obj.goto(url, timeout=90000, wait_until="domcontentloaded")
                        # Real proof the country-filtered results actually
                        # rendered -- either a real catalog link shows up,
                        # or the page settles on a genuine "no results"
                        # state. Not a fixed sleep: waits for the actual
                        # thing that matters, same principle proven
                        # reliable in the lot-pull tonight.
                        try:
                            page_obj.wait_for_function(
                                "document.querySelectorAll('a[href*=\"catalogue-id-\"]').length > 0"
                                " || document.body.innerText.toLowerCase().includes('no results')"
                                " || document.body.innerText.toLowerCase().includes('0 auctions')",
                                timeout=30000
                            )
                        except Exception:
                            pass  # fall through -- still try to read whatever DOM state exists
                        html = page_obj.content()
                    except Exception as e:
                        err = f"page {page_num} fetch failed: {type(e).__name__}: {e}"
                        print(f"BidSpotter scan: {err}")
                        if page_num == 1:
                            result["first_page_error"] = err
                        break

                    result["pages_fetched"] += 1
                    listings = _parse_bidspotter_listing_page_from_dom(html)
                    if page_num == 1:
                        marker_idx = html.find("catalogue-id-")
                        snippet = re.sub(r'\s+', ' ', html[:400]).strip()
                        if marker_idx >= 0:
                            around = re.sub(r'\s+', ' ', html[max(0, marker_idx-150):marker_idx+150]).strip()
                            marker_info = f"catalogue-id- FOUND at offset {marker_idx}, context=\"{around}\""
                        else:
                            marker_info = "catalogue-id- NOT found anywhere in the rendered DOM"
                        # TARGETED: specifically look for real <a> tags with
                        # catalogue-id- in their href, not just any
                        # occurrence of that text anywhere on the page
                        # (which could be inside a JSON blob instead).
                        anchor_matches = re.findall(r'<a\b[^>]*href="[^"]*catalogue-id-[^"]*"[^>]*>', html)
                        anchor_info = f"real <a href> tags with catalogue-id-: {len(anchor_matches)}"
                        if anchor_matches:
                            anchor_info += f" | first: {anchor_matches[0][:300]}"
                        result["first_page_diagnostic"] = f"page1 bytes={len(html)} | {marker_info} | {anchor_info} | snippet=\"{snippet}\""
                    if not listings:
                        empty_pages_in_a_row += 1
                    else:
                        empty_pages_in_a_row = 0
                        for item in listings:
                            result["all_listings"][item["catalog_url"]] = item
                    page_num += 1
            finally:
                try:
                    browser.close()
                except Exception:
                    pass
        return result

    try:
        loop = asyncio.get_event_loop()
        # The real hard timeout: this future comes from a genuine OS thread
        # (run_in_executor), so asyncio.wait_for's cancellation here just
        # means "stop waiting" -- it does NOT depend on the thread's blocked
        # code cooperating, unlike wrapping an async coroutine directly.
        scan_result = await asyncio.wait_for(loop.run_in_executor(None, _sync_scan_work), timeout=25 * 60.0)
    except asyncio.TimeoutError:
        err = "hit the 25-minute hard ceiling waiting for the scan thread -- abandoning it and reporting failure (the thread itself may still be stuck in the background, harmlessly)"
        print(f"BidSpotter scan: {err}")
        return {"ok": False, "error": err, "pages": 0, "listings": 0, "new_flagged": 0}
    except Exception as e:
        err = f"browser session failed: {type(e).__name__}: {e}"
        print(f"BidSpotter scan: {err}")
        return {"ok": False, "error": err, "pages": 0, "listings": 0, "new_flagged": 0}

    all_listings = scan_result["all_listings"]
    pages_fetched = scan_result["pages_fetched"]
    first_page_error = scan_result["first_page_error"]
    first_page_diagnostic = scan_result["first_page_diagnostic"]

    if first_page_error:
        return {"ok": False, "error": first_page_error, "pages": pages_fetched, "listings": 0, "new_flagged": 0}

    if not all_listings:
        msg = f"found nothing at all -- likely a page layout change or a bot-block, not a real empty result | {first_page_diagnostic}"
        print(f"BidSpotter scan: {msg}")
        return {"ok": False, "error": msg, "pages": pages_fetched, "listings": 0, "new_flagged": 0}

    now_iso = datetime.now(timezone.utc).isoformat()
    snapshot_rows = [
        {"business_id": business_id, "catalog_url": v["catalog_url"], "title": v["title"],
         "end_date": v.get("end_date"), "scanned_at": now_iso}
        for v in all_listings.values()
    ]
    loop = asyncio.get_event_loop()
    for i in range(0, len(snapshot_rows), 500):
        chunk = snapshot_rows[i:i+500]
        await loop.run_in_executor(None, lambda chunk=chunk: supabase_client.table("bidspotter_scan_snapshot").upsert(
            chunk, on_conflict="business_id,catalog_url"
        ).execute())

    known_urls = set()
    known_rows = await loop.run_in_executor(None, lambda: _fetch_all_paginated(lambda: supabase_client.table("auction_pdf_uploads").select("catalog_url").eq("business_id", business_id)))
    for r in known_rows:
        known_urls.add(r["catalog_url"])

    # REAL BUG FIXED HERE: Job 1 previously only checked auction_pdf_uploads
    # (the VA's manual upload history) to decide what's already handled --
    # it had zero awareness of the automated lot-pull or any manual queue
    # cleanup. That meant every catalog the automation successfully pulled
    # (or that got manually resolved after confirming non-US, etc.) got
    # RE-FLAGGED as "new" the very next time Job 1 ran, undoing that work
    # every single cycle. Now also excludes: (1) anything already resolved
    # in catalog_updates_queue, (2) anything with real rows already in
    # bidspotter_auto_catalog_lots (the automated staging table) -- belt
    # and suspenders in case a resolve was ever missed.
    resolved_rows = await loop.run_in_executor(None, lambda: _fetch_all_paginated(lambda: supabase_client.table("catalog_updates_queue").select("catalog_url").eq("business_id", business_id).eq("resolved", True)))
    for r in resolved_rows:
        known_urls.add(r["catalog_url"])
    try:
        staged_rows = await loop.run_in_executor(None, lambda: _fetch_all_paginated(lambda: supabase_client.table("bidspotter_auto_catalog_lots").select("catalog_url").eq("business_id", business_id)))
        for r in staged_rows:
            known_urls.add(r["catalog_url"])
    except Exception as e:
        print(f"BidSpotter scan: failed to fetch staged catalog_urls (non-fatal, continuing): {e}")

    # REAL BUG FIXED HERE: a catalog already sitting UNRESOLVED in
    # catalog_updates_queue (kind='new' or kind='blank') was never added to
    # known_urls, so it kept showing up in `candidates` below on every
    # subsequent Job 1 run and getting silently re-verified and overwritten
    # a second time -- on top of the dedicated backlog-cleanup loop above
    # that already owns re-checking blank rows. Two uncoordinated code
    # paths writing the same row, each trusting a single fetch with no
    # retry, is exactly how a correct kind='new' classification could get
    # clobbered back to 'blank' (or vice versa) by one bad/blocked
    # response. Each unresolved row now has exactly one owner: the backlog
    # loop above. This loop only ever handles a catalog_url it has never
    # seen before.
    already_queued = await loop.run_in_executor(None, lambda: _fetch_all_paginated(lambda: supabase_client.table("catalog_updates_queue").select("catalog_url").eq("business_id", business_id).eq("resolved", False)))
    for r in already_queued:
        known_urls.add(r["catalog_url"])

    new_count = 0
    skipped_no_lots_yet = 0
    candidates = [(catalog_url, item) for catalog_url, item in all_listings.items() if catalog_url not in known_urls]
    async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "Mozilla/5.0"}) as verify_client:
        # BACKLOG CLEANUP FIRST: everything queued BEFORE this verify step
        # existed (or from the brief bad deploy that skipped verification
        # entirely) is still sitting in catalog_updates_queue unresolved,
        # most of it genuinely blank. Deliberately runs before the brand-new
        # candidates below -- these are the ones actually blocking the VA
        # right now, so they get checked first instead of waiting behind a
        # fresh batch of new candidates.
        #
        # REAL FIX: confirmed-blank rows used to be DELETED outright -- that
        # was wrong. Deleting means no record of what was checked or why it
        # was removed, and the person has no way to verify or spot-check the
        # work. Now they're UPDATED in place to kind='blank' and stay
        # visible (in their own "Confirmed Blank" section, not the
        # actionable Updates-to-Make list) instead of disappearing. Still
        # excluded from known_urls below only via the resolved flag (stays
        # False), so a future Job 1 run naturally re-checks and upgrades it
        # the moment BidSpotter actually posts real lots.
        # REAL BUG FIXED HERE: this used to re-check EVERY unresolved row on
        # every single Job 1 run, including ones already confirmed to have
        # real lots hours earlier. One bad/blocked fetch (BidSpotter's WAF
        # challenge, or Bright Data getting rate-limited from checking the
        # same catalog repeatedly across back-to-back runs) was then enough
        # to wrongly flip a confirmed-good catalog to blank with no second
        # opinion -- confirmed live: "Pace Industries - AR - Day 2" is a
        # real upcoming Oct 2026 auction and got marked blank by exactly
        # this. Only re-check rows that are still unconfirmed (kind is
        # 'blank', to see if real content has since appeared) -- a row
        # already confirmed 'new' (has real lots) is trusted and left
        # alone here; Job 2/3's normal ongoing lot-pull is what tracks
        # whether an active catalog later goes stale, not this check.
        backlog_removed = 0
        try:
            backlog_rows = await loop.run_in_executor(None, lambda: _fetch_all_paginated(lambda: supabase_client.table("catalog_updates_queue").select("id,catalog_url").eq("business_id", business_id).eq("resolved", False).eq("kind", "blank")))
        except Exception as e:
            print(f"Job 1 backlog cleanup: failed to fetch existing queue rows: {e}")
            backlog_rows = []
        for i, row in enumerate(backlog_rows, 1):
            catalog_url = row["catalog_url"]
            if i % 5 == 0 or i == 1:
                await loop.run_in_executor(None, lambda i=i: _update_live_activity(supabase_client, business_id, f"Job 1: cleaning old queue backlog {i}/{len(backlog_rows)}"))
            real_url = _reconstruct_full_url(catalog_url)
            if not real_url:
                continue
            try:
                resp = await _brightdata_get(verify_client, real_url)
                if resp.status_code != 200:
                    continue  # can't confirm either way -- leave it, don't touch it on a failed check
                category_count = len(re.findall(r'search-filter\?CategoryCode=', resp.text))
            except Exception as e:
                print(f"Job 1 backlog cleanup: verify fetch failed for {catalog_url} (leaving it, not removing on a flaky check): {type(e).__name__}: {e}")
                continue
            if category_count == 0:
                continue  # already kind='blank' -- nothing to change, still confirmed blank
            # REAL BUG FIXED HERE: this loop only ever handled the
            # category_count==0 case -- there was no path to upgrade a
            # catalog back to kind='new' once BidSpotter actually posted
            # real lots for it. A catalog confirmed blank would stay
            # invisible to the VA forever, even after content appeared.
            try:
                await loop.run_in_executor(None, lambda row_id=row["id"], catalog_url=catalog_url, category_count=category_count: supabase_client.table("catalog_updates_queue").update({
                    "kind": "new", "category_count": category_count,
                }).eq("id", row_id).execute())
                new_count += 1
                print(f"Job 1 backlog cleanup: {catalog_url} now has real lots (category_count={category_count}) -- upgraded back to Updates to Make")
            except Exception as e:
                print(f"Job 1 backlog cleanup: failed to upgrade newly-active row {catalog_url}: {e}")

        for i, (catalog_url, item) in enumerate(candidates, 1):
            # REAL FIX: every one of these Supabase calls is the SYNC client,
            # called directly inside this async function -- exactly the same
            # event-loop-blocking bug already fixed for the HTTP routes, just
            # relocated here. With ~120 calls across this loop it froze the
            # whole server again during every Job 1 run. run_in_executor
            # moves each one off the event loop onto a real thread.
            if i % 5 == 0 or i == 1:
                await loop.run_in_executor(None, lambda i=i: _update_live_activity(supabase_client, business_id, f"Job 1: verifying new catalog {i}/{len(candidates)} actually has lots"))
            real_url = _reconstruct_full_url(catalog_url)
            category_count = 0
            if real_url:
                # An earlier version of this tried to read category tags
                # straight off the bulk listing page Job 1 already loads --
                # looked right in a one-off check, but confirmed live it
                # ALWAYS reads 0 in production (that widget loads via a
                # separate lazy AJAX call per card, not present in the page
                # by the time Job 1 reads it -- same "Cannot load data"
                # placeholder visible on the live site). Reverted to the one
                # mechanism actually proven reliable (Job 2 already uses it
                # daily): fetch the catalog's OWN individual page, where the
                # same category tags are real static content, not lazy-
                # loaded. Costs one extra fetch per NEW catalog only (not
                # the full listing), which is a small number most runs.
                try:
                    resp = await _brightdata_get(verify_client, real_url)
                    if resp.status_code == 200:
                        category_count = len(re.findall(r'search-filter\?CategoryCode=', resp.text))
                        # A WAF/bot-check interstitial can return a real 200
                        # with placeholder content -- that would read as 0
                        # categories and get confidently written as
                        # "confirmed blank" for a catalog that actually has
                        # real lots. Only trust a 0 when the page actually
                        # mentions this catalog's own title (a real render);
                        # otherwise treat it the same as a failed fetch.
                        title_text = (item.get("title") or "").strip().lower()
                        if category_count == 0 and title_text and title_text not in resp.text.lower():
                            print(f"Job 1: verify fetch for {catalog_url} returned 0 categories but page doesn't mention its own title -- likely a blocked/placeholder page, not trusting the 0")
                            category_count = 1  # fail open, same reasoning as a failed fetch below
                except Exception as e:
                    print(f"Job 1: verify fetch failed for {catalog_url} (queuing anyway rather than losing it): {type(e).__name__}: {e}")
                    category_count = 1  # fail open -- don't silently drop a real new catalog over a flaky verify fetch
            else:
                category_count = 1  # couldn't reconstruct the URL to verify -- fail open, same reasoning as above

            if category_count == 0:
                skipped_no_lots_yet += 1
                try:
                    def _upsert_blank(catalog_url=catalog_url, item=item):
                        supabase_client.table("catalog_updates_queue").upsert({
                            "business_id": business_id, "catalog_url": catalog_url, "title": item["title"],
                            "end_date": item.get("end_date"), "kind": "blank", "resolved": False,
                            "category_count": 0,
                        }, on_conflict="business_id,catalog_url").execute()
                    await loop.run_in_executor(None, _upsert_blank)
                except Exception as e:
                    print(f"BidSpotter scan: failed to record blank catalog {catalog_url}: {e}")
                continue
            try:
                def _upsert(catalog_url=catalog_url, item=item, category_count=category_count):
                    supabase_client.table("catalog_updates_queue").upsert({
                        "business_id": business_id, "catalog_url": catalog_url, "title": item["title"],
                        "end_date": item.get("end_date"), "kind": "new", "resolved": False,
                        "category_count": category_count,
                    }, on_conflict="business_id,catalog_url").execute()
                await loop.run_in_executor(None, _upsert)
                new_count += 1
            except Exception as e:
                print(f"BidSpotter scan: failed to queue new catalog {catalog_url}: {e}")

    return {"ok": True, "error": None, "pages": pages_fetched, "listings": len(all_listings),
            "new_flagged": new_count, "skipped_no_lots_yet": skipped_no_lots_yet, "backlog_removed": backlog_removed}

async def _recheck_blank_catalogs(supabase_client, business_id: str) -> dict:
    """Job 2. For every catalog we already know about that's currently
    sitting at zero lots, re-fetches its own individual page directly and
    checks for real content. A catalog card shows a 'Cannot load data'
    placeholder for its lot-count widget regardless of whether it actually
    has lots (that's just an unrendered JS component in a static fetch, not
    a real signal) -- but the category tag list underneath it IS real, and
    is genuinely absent when a catalog has nothing in it yet.

    ROOT-CAUSE FIX: this function draws its candidate list from
    auction_pdf_uploads (the VA's own known-catalog history), which is NOT
    country-filtered -- unlike Job 1's listing scan, which only ever visits
    BidSpotter's countryName=United%20States URL. That mismatch was the
    actual source of Canadian/UK/Mexican catalogs reaching
    catalog_updates_queue, not a scraping failure -- fixing it at the lot-
    pull stage was always going to be whack-a-mole. Real fix: cross-
    reference against bidspotter_scan_snapshot (which IS built exclusively
    from the US-filtered listing pages) before re-queuing anything here. A
    catalog not seen on the current US listing simply never gets queued."""
    us_catalog_urls = set()
    try:
        snap_rows = _fetch_all_paginated(lambda: supabase_client.table("bidspotter_scan_snapshot").select("catalog_url").eq("business_id", business_id))
        us_catalog_urls = {r["catalog_url"] for r in snap_rows}
    except Exception as e:
        print(f"BidSpotter recheck: failed to fetch US-filtered snapshot, cannot safely re-queue anything this run: {e}")
        return {"ok": False, "error": f"snapshot fetch failed: {e}", "reactivated": 0, "growing": 0, "checked": 0}

    latest_status = {}
    rows = _fetch_all_paginated(lambda: supabase_client.table("auction_pdf_uploads").select("catalog_url,status,uploaded_at,filename").eq("business_id", business_id).order("uploaded_at"))
    for r in rows:
        latest_status[r["catalog_url"]] = r  # later rows overwrite earlier -- ends up holding the latest per catalog_url

    # Same disconnect fixed elsewhere tonight: skip anything the AUTOMATED
    # pull already has real data for. Without this, a catalog Job 3 already
    # resolved gets needlessly re-fetched from BidSpotter and re-queued as
    # "reactivated" here too, wasting a request and cluttering the queue
    # with something already done.
    already_automated = set()
    try:
        auto_rows = _fetch_all_paginated(lambda: supabase_client.table("bidspotter_auto_catalog_lots").select("catalog_url").eq("business_id", business_id))
        already_automated = {r["catalog_url"] for r in auto_rows}
    except Exception as e:
        print(f"BidSpotter recheck: failed to fetch automated staging table (non-fatal, continuing without this filter): {e}")

    blank_catalogs = [r for r in latest_status.values()
                       if r["status"] == "empty" and r["catalog_url"] in us_catalog_urls
                       and r["catalog_url"] not in already_automated]
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
    small_catalog_urls = {r["catalog_url"] for r in small_catalog_rows if r["catalog_url"] in us_catalog_urls and r["catalog_url"] not in already_automated}

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
        for i, row in enumerate(blank_catalogs, 1):
            catalog_url = row["catalog_url"]
            _update_live_activity(supabase_client, business_id, f"Job 2: rechecking blank catalog {i}/{len(blank_catalogs)}")
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

        for i, catalog_url in enumerate(small_catalog_urls, 1):
            _update_live_activity(supabase_client, business_id, f"Job 2: checking growth on small catalog {i}/{len(small_catalog_urls)}")
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
    """Runs Job 1 (detect new/existing catalogs), then IMMEDIATELY pulls real
    lot data for anything newly queued (the actual point of this whole
    pipeline), THEN runs Job 2 (recheck blanks + growth detection) -- Job 2
    doesn't need to happen before the lot-pull, and was previously blocking
    it for no real reason. Writes the outcome into bidspotter_scan_status --
    so what actually happened is checkable via a normal query, not lost in
    server logs nobody can reach.

    Set SKIP_JOB1=1 as a Railway variable to skip straight to the lot-pull
    using whatever's already queued -- avoids re-running the ~10-page
    listing scan on every single restart during active debugging."""
    if os.environ.get("SKIP_JOB1") == "1":
        print(f"SKIP_JOB1 is set -- skipping Job 1, going straight to the lot-pull for {business_id}")
        scan_result = {"ok": True, "error": None, "pages": 0, "listings": 0, "new_flagged": 0}
    else:
        scan_result = await _scan_bidspotter_new_catalogs(supabase_client, business_id)
    lot_pull_result = await _pull_lots_for_queued_catalogs(supabase_client, business_id)
    print(f"BidSpotter lot pull for {business_id}: {lot_pull_result}")
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
    _invalidate_cache(f"catalogs:{business_id}")
    _invalidate_cache(f"lots:{business_id}")
    _invalidate_cache(f"bright-lots:{business_id}")
    _invalidate_cache(f"needs-update:{business_id}")
    _invalidate_cache(f"updates-to-make:{business_id}")
    _invalidate_cache(f"confirmed-blank:{business_id}")
    return status_row

class _NoLotsPublishedYet(Exception):
    """Fast verdict: real full-size catalog page with no search form and no
    lot cards -- the auctioneer hasn't published lots yet, nothing to pull.
    Used to short-circuit retries in 30s instead of ~4 minutes."""
    pass

async def _fetch_catalog_lots_via_browser(catalog_url_full: str, catalog_slug: str, known_total: int = None) -> dict:
    """ONE Bright Data browser session per catalog (reverted from
    session-per-page): does 'In this auction' search-submit ONCE, then
    pages through ALL results within that SAME session using BidSpotter's
    page=N URL parameter. This is deliberately back to the original
    architecture -- session-per-page was built earlier to fix what looked
    like session degradation, but the REAL root cause (confirmed via live
    diagnostic) was that direct page=N navigation returns zero lots without
    first submitting the search; session-per-page never fixed anything real
    and its dozens of fresh connections per catalog were very likely what
    triggered Bright Data's "cooldown (no_peers)" proxy error seen live.
    One session per catalog matches the two confirmed-exact runs from
    earlier tonight (722/722, 858/858) and opens far fewer connections.
    Hedge against a session genuinely dying mid-catalog: on any hard
    failure, reconnect a fresh session, redo the search-submit, and RESUME
    from the current page (checkpointed via seen_lot_numbers/lots/page_num
    in this function's scope) instead of restarting the whole catalog."""
    from playwright.async_api import async_playwright
    from urllib.parse import quote
    import uuid
    wss_url = os.environ.get("BRIGHT_DATA_BROWSER_WSS")
    if not wss_url:
        return {"lots": [], "state": None, "zip_code": None, "complete": False}
    session_id = f"catalog-{catalog_slug}-{uuid.uuid4().hex[:8]}"
    lots = []
    state = None
    zip_code = None
    country = None
    complete = True
    lot_pattern = re.compile(
        r'href="(/en-us/auction-catalogues/([a-z0-9]+)/catalogue-id-([a-z0-9\-]+)/lot-[a-z0-9\-]+)"[^>]*data-click-type="title"[^>]*>\s*<span class="lot-number">([^<]+)</span><span class="lot-title">([^<]+)</span>',
        re.I
    )

    def extract_matching_lots(html: str) -> list:
        found = []
        for href, auctioneer, cat_id, lot_num, lot_title in lot_pattern.findall(html):
            if cat_id.lower() == catalog_slug.lower():
                found.append((lot_num, lot_title.strip()))
        return found

    def _itemprop_from(html: str, name: str) -> str:
        m = re.search(rf'<[^>]*itemprop="{name}"[^>]*\scontent="([^"]*)"', html)
        if not m:
            m = re.search(rf'<[^>]*itemprop="{name}"[^>]*>([^<]*)<', html)
        val = (m.group(1).strip() if m else "")
        return val if val and val != "." else ""

    path_part = catalog_url_full.replace("https://www.bidspotter.com", "")
    where_to_search = quote(path_part, safe="")

    def _page_url(page_num: int) -> str:
        return f"{catalog_url_full}?searchTerm=&whereToSearch={where_to_search}&page={page_num}"

    MAX_PAGES = 60
    seen_lot_numbers = set()

    def _record(new_lots):
        for lot_num, lot_title in new_lots:
            seen_lot_numbers.add(lot_num)
            lots.append({"lot_number": lot_num, "description": lot_title})

    async def _open_fresh_session_and_search(pw):
        """Connect one browser session, pin the shared proxy peer, load the
        catalog root (capturing location data on the way if not already
        captured), then perform the 'In this auction' search-submit ONCE.
        Returns the (browser, page) to reuse for all subsequent pages."""
        nonlocal state, zip_code, country
        browser = await pw.chromium.connect_over_cdp(wss_url, timeout=120000)
        page = await browser.new_page()
        try:
            cdp_client = await page.context.new_cdp_session(page)
            await cdp_client.send("Proxy.useSession", {"sessionId": session_id})
        except Exception as e:
            print(f"Proxy.useSession failed (continuing without pinning): {type(e).__name__}: {e}")
        await page.goto(catalog_url_full, timeout=120000, wait_until="domcontentloaded")
        if state is None and zip_code is None and country is None:
            # domcontentloaded can fire before the schema.org microdata is
            # actually rendered into the DOM -- brief settle wait before
            # reading, matching the original working capture behavior.
            await page.wait_for_timeout(2500)
            root_html = await page.content()
            state = _itemprop_from(root_html, "addressRegion") or None
            zip_code = _itemprop_from(root_html, "postalCode") or None
            country = _itemprop_from(root_html, "addressCountry") or None
        try:
            # state="attached", NOT "visible": confirmed via live log that
            # on some pages this element exists as a HIDDEN radio input
            # (the label is the visible UI) -- waiting for visibility never
            # resolves even though our force=True click works on it fine.
            await page.wait_for_selector("#catalogueSearchOption", timeout=30000, state="attached")
        except Exception:
            try:
                cur_html = await page.content()
                real_lot_cards = cur_html.count('class="lot-number"')
                if "catalogueSearchOption" not in cur_html and real_lot_cards > 0:
                    print(f"[FORM-LESS VARIANT] {page.url}: no search form in HTML but {real_lot_cards} real lot cards present -- proceeding without search-submit")
                    return browser, page
                title_match = re.search(r'<title[^>]*>([^<]*)</title>', cur_html, re.I)
                title = title_match.group(1) if title_match else ""
                # FAST NON-US CLASSIFICATION from the title when microdata
                # failed: live-confirmed leak case was bsccr10074 whose
                # title literally says "(Toronto, ON)" while schema.org
                # location came back empty, so the non-US skip no-op'd and
                # burned full retries. Canadian province abbreviations in a
                # "(City, XX)" pattern, or explicit country words.
                nonus = re.search(r'\(\s*[^)]*,\s*(ON|BC|QC|AB|MB|SK|NS|NB|NL|PE)\s*\)|\b(Canada|Canadian|United Kingdom|Ontario|Quebec|British Columbia|Alberta)\b', title)
                if nonus:
                    print(f"[TITLE NON-US] {page.url}: title indicates non-US ({nonus.group(0)!r}) -- classifying as non-US, no retries. title={title!r}")
                    country = f"non-US (from title: {nonus.group(0)})"
                    return browser, page
                # FAST SHELL CLASSIFICATION: a real, full-size page (not the
                # tiny WAF stub) with NO search form AND NO lot cards is the
                # live-confirmed "no lots published yet" variant (new
                # catalogs where the auctioneer hasn't uploaded lots) --
                # there is genuinely nothing to pull. Verdict in 30s instead
                # of ~4 minutes of pointless retries per cycle.
                if len(cur_html) > 100000 and real_lot_cards == 0 and "catalogueSearchOption" not in cur_html:
                    print(f"[NO LOTS PUBLISHED YET] {page.url}: full page ({len(cur_html)}b) with no form and no lot cards -- nothing to pull yet, will recheck next cycle. title={title!r}")
                    try:
                        await browser.close()
                    except Exception:
                        pass
                    raise _NoLotsPublishedYet()
                print(f"[SEARCH-FORM MISSING DIAGNOSTIC] url={page.url} html_length={len(cur_html)} title={title!r} real-lot-cards={real_lot_cards}")
            except _NoLotsPublishedYet:
                raise
            except Exception as de:
                print(f"[SEARCH-FORM MISSING DIAGNOSTIC] could not read page content either: {type(de).__name__}: {de}")
            raise
        await page.click("#catalogueSearchOption", timeout=15000, force=True)
        await page.click("#searchSubmit", timeout=15000, force=True)
        await page.wait_for_load_state("domcontentloaded", timeout=90000)
        return browser, page

    async def _read_page(page, target_url: str) -> str:
        """Navigate the EXISTING session to target_url and return real
        content, waiting for proof the AWS WAF challenge stub (if served)
        has resolved before reading -- same proof-of-content check proven
        during tonight's diagnostics."""
        response = await page.goto(target_url, timeout=120000, wait_until="domcontentloaded")
        if response is not None:
            try:
                hdrs = response.headers
                brd_err_code = hdrs.get("x-brd-err-code")
                brd_err_msg = hdrs.get("x-brd-err-msg")
                proxy_status = hdrs.get("proxy-status")
                if brd_err_code or brd_err_msg or proxy_status:
                    print(f"[BRIGHTDATA ERROR HEADERS] status={response.status} code={brd_err_code!r} msg={brd_err_msg!r} proxy-status={proxy_status!r} url={target_url}")
            except Exception:
                pass
        try:
            await page.wait_for_function(
                "document.body && document.body.innerHTML.includes('lot-number')"
                " || (document.title && document.title.length > 0"
                " && !window.awsWafCookieDomainList)",
                timeout=15000
            )
        except Exception:
            pass
        return await page.content()

    try:
        async with async_playwright() as pw:
            browser = None
            page = None
            last_open_error = None
            for open_attempt in range(1, 4):
                try:
                    browser, page = await _open_fresh_session_and_search(pw)
                    last_open_error = None
                    break
                except _NoLotsPublishedYet:
                    # Fast verdict, no retries: full real page, no form, no
                    # lot cards = auctioneer hasn't published lots yet.
                    # Nothing to pull; catalog stays queued and gets
                    # rechecked next cycle in 30s, not 4 minutes.
                    if browser is not None:
                        try:
                            await browser.close()
                        except Exception:
                            pass
                    return {"lots": [], "state": state, "zip_code": zip_code, "country": country, "complete": False, "no_lots_published": True}
                except Exception as e:
                    last_open_error = e
                    print(f"Initial session open/search-submit attempt {open_attempt}/3 failed for {catalog_url_full}: {type(e).__name__}: {e}")
                    if browser is not None:
                        try:
                            await browser.close()
                        except Exception:
                            pass
                        browser = None
            if last_open_error is not None:
                print(f"Could not open a session for {catalog_url_full} after 3 attempts -- PULL IS INCOMPLETE")
                return {"lots": [], "state": None, "zip_code": None, "country": None, "complete": False}

            if country and country.lower() not in ("united states", "usa", "us"):
                print(f"Skipping non-US catalog ({country}): {catalog_url_full}")
                try:
                    await browser.close()
                except Exception:
                    pass
                return {"lots": [], "state": None, "zip_code": None, "country": country, "complete": True}

            pending_empty = []
            page_num = 1
            reconnects_used = 0
            MAX_RECONNECTS = 4  # hedge budget for the whole catalog, not per-page
            while page_num <= MAX_PAGES:
                try:
                    page_html = await _read_page(page, _page_url(page_num))
                except Exception as e:
                    print(f"Page {page_num} failed on current session for {catalog_url_full}: {type(e).__name__}: {e}")
                    if reconnects_used >= MAX_RECONNECTS:
                        print(f"Reconnect budget ({MAX_RECONNECTS}) exhausted for {catalog_url_full} -- PULL IS INCOMPLETE at page {page_num} ({len(lots)} lots so far)")
                        complete = False
                        break
                    reconnects_used += 1
                    print(f"Reconnecting fresh session ({reconnects_used}/{MAX_RECONNECTS}) and resuming at page {page_num} ({len(lots)} lots checkpointed)")
                    try:
                        try:
                            await browser.close()
                        except Exception:
                            pass
                        browser, page = await _open_fresh_session_and_search(pw)
                    except Exception as e2:
                        print(f"Reconnect itself failed for {catalog_url_full}: {type(e2).__name__}: {e2}")
                        complete = False
                        break
                    continue  # retry the SAME page_num on the new session

                page_lots = extract_matching_lots(page_html)
                new_on_this_page = [l for l in page_lots if l[0] not in seen_lot_numbers]

                if new_on_this_page:
                    if pending_empty:
                        print(f"Page(s) {pending_empty} for {catalog_url_full} were empty but page {page_num} has real lots -- re-fetching the skipped page(s), not a real end")
                        for p in pending_empty:
                            try:
                                r_html = await _read_page(page, _page_url(p))
                                r_new = [l for l in extract_matching_lots(r_html) if l[0] not in seen_lot_numbers]
                                if r_new:
                                    _record(r_new)
                            except Exception as e:
                                print(f"Re-fetch of skipped page {p} failed: {type(e).__name__}: {e}")
                        pending_empty = []
                    _record(new_on_this_page)
                    print(f"Page {page_num} for {catalog_url_full}: running total {len(lots)} lots")
                    page_num += 1
                    continue

                # Empty page (fetched fine, zero new lots on THIS session).
                if page_num == 1:
                    complete = False
                    break
                pending_empty.append(page_num)
                if len(pending_empty) >= 2:
                    if known_total is not None and len(lots) < known_total:
                        print(f"Pages {pending_empty} both empty for {catalog_url_full}, but only {len(lots)}/{known_total} known lots collected -- known_total says NOT done, continuing past the heuristic")
                        pending_empty = []
                        page_num += 1
                        continue
                    break
                page_num += 1
            else:
                print(f"Pagination hit the {MAX_PAGES}-page ceiling for {catalog_url_full} without confirming end-of-results -- PULL IS INCOMPLETE")
                complete = False

            try:
                await browser.close()
            except Exception:
                pass
    except Exception as e:
        print(f"Browser lot fetch failed for {catalog_url_full}: {type(e).__name__}: {e} -- PULL IS INCOMPLETE ({len(lots)} lots collected before failure)")
        complete = False
    return {"lots": lots, "state": state, "zip_code": zip_code, "country": country, "complete": complete}


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
    queue_rows = _fetch_all_paginated(lambda: supabase_client.table("catalog_updates_queue").select("catalog_url,title,auctioneer,end_date").eq("business_id", business_id).eq("resolved", False))
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
        _update_live_activity(supabase_client, business_id, f"Job 3 (lot pull): catalog {attempted}/{len(queue_rows)} -- {catalog_url}")
        try:
            supabase_client.table("bidspotter_lot_pull_progress").update({
                "current_catalog_url": catalog_url, "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("business_id", business_id).execute()
        except Exception:
            pass
        lots = []
        fetch_result = {"lots": [], "state": None, "zip_code": None, "country": None}
        max_attempts = 2  # reduced from 3 -- each attempt can now legitimately take much longer
        for retry_num in range(1, max_attempts + 1):
            try:
                # Hard outer ceiling -- raised from 120s to 600s. The
                # per-page retry logic inside (added to fix soft-blocks
                # being misread as end-of-results) means a real catalog can
                # legitimately need several minutes now, especially with
                # several blocked pages each eating up to 4 retries. The old
                # 120s ceiling was firing WHILE the inner logic was still
                # legitimately working, forcibly killing the browser
                # mid-operation (TargetClosedError cascade) -- this wasn't
                # protecting against a hang, it was causing failures on
                # every catalog that took normal-but-longer real time.
                fetch_result = await asyncio.wait_for(
                    _fetch_catalog_lots_via_browser(real_url, catalog_slug), timeout=600.0
                )
                lots = fetch_result["lots"]
                if lots and fetch_result.get("complete", False):
                    break
                if lots and not fetch_result.get("complete", False):
                    # Got SOME lots but the pull died before the real end --
                    # this was the silent partial-data bug: previously any
                    # nonzero lot count was treated as a full success.
                    print(f"Lot pull attempt {retry_num}/{max_attempts} for {catalog_url}: INCOMPLETE ({len(lots)} lots before failure), retrying from scratch")
                    continue
                if fetch_result.get("no_lots_published"):
                    # Real page, no lots exist yet -- a second full attempt
                    # cannot change that. Stays queued for next cycle.
                    print(f"Lot pull for {catalog_url}: no lots published yet, not retrying this cycle")
                    break
                if fetch_result.get("country") and fetch_result["country"].lower() not in ("united states", "usa", "us"):
                    # Confirmed non-US -- retrying won't change the country,
                    # and rechecking it every future cycle forever is pure
                    # waste. Resolve it PERMANENTLY, not just skip this run.
                    print(f"Lot pull for {catalog_url}: confirmed non-US ({fetch_result['country']}), resolving permanently (never rechecking)")
                    try:
                        supabase_client.table("catalog_updates_queue").update(
                            {"resolved": True}
                        ).eq("business_id", business_id).eq("catalog_url", catalog_url).execute()
                    except Exception as e:
                        print(f"Failed to permanently resolve non-US catalog {catalog_url}: {e}")
                    break
                print(f"Lot pull attempt {retry_num}/{max_attempts} for {catalog_url}: got 0 lots, retrying from scratch")
            except asyncio.TimeoutError:
                print(f"Lot pull attempt {retry_num}/{max_attempts} for {catalog_url}: HARD TIMEOUT (600s), redoing from scratch")
            except Exception as e:
                print(f"Lot pull attempt {retry_num}/{max_attempts} for {catalog_url} failed: {type(e).__name__}: {e}")
        if lots:
            pull_complete = fetch_result.get("complete", False)
            if pull_complete:
                succeeded += 1
            rows_to_upsert = [
                {"business_id": business_id, "catalog_url": catalog_url, "lot_number": lot["lot_number"],
                 "description": lot["description"], "last_seen_at": datetime.now(timezone.utc).isoformat(),
                 "catalog_title": row.get("title"), "state": fetch_result.get("state"),
                 "zip_code": fetch_result.get("zip_code"), "date": row.get("end_date"),
                 "country": fetch_result.get("country")}
                for lot in lots
            ]
            try:
                supabase_client.table("bidspotter_auto_catalog_lots").upsert(
                    rows_to_upsert, on_conflict="business_id,catalog_url,lot_number"
                ).execute()
                total_lots_written += len(rows_to_upsert)
                if pull_complete:
                    # A COMPLETE automated pull is treated the same as the VA
                    # having handled this catalog manually -- clears it from
                    # the front-facing queue. A partial pull deliberately does
                    # NOT resolve: its lots are saved (upsert -- a later full
                    # pull just fills in the rest), but the catalog stays in
                    # the queue so the next 12h cycle finishes the job.
                    supabase_client.table("catalog_updates_queue").update(
                        {"resolved": True}
                    ).eq("business_id", business_id).eq("catalog_url", catalog_url).execute()
                else:
                    print(f"Wrote {len(rows_to_upsert)} PARTIAL lots for {catalog_url} -- left in queue for the next cycle")
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
    /api/updates/trigger-scan exists purely for on-demand debugging.

    PAUSE_SCAN=1 skips this entirely -- added during the Supabase Disk IO
    Budget incident so the background jobs can be fully silenced (zero
    further database load) while the budget recovers, without needing a
    code change/redeploy each time. Checked every cycle, not just once at
    startup, so flipping it off takes effect on the very next 12h tick too."""
    await asyncio.sleep(30)
    while True:
        if os.environ.get("PAUSE_SCAN") == "1":
            print("PAUSE_SCAN=1 -- skipping this cycle entirely (no DB/BidSpotter activity)")
        else:
            try:
                supabase_client, _ = _require_config()
                biz_rows = supabase_client.table("auction_pdf_uploads").select("business_id").limit(1000).execute().data or []
                business_ids = {r["business_id"] for r in biz_rows}
                for business_id in business_ids:
                    if os.environ.get("PAUSE_SCAN") == "1":
                        print("PAUSE_SCAN=1 -- stopping mid-cycle before the next business_id")
                        break
                    await _run_bidspotter_scan_for_business(supabase_client, business_id)
            except Exception as e:
                print(f"BidSpotter daily scan loop failed: {e}")
        await asyncio.sleep(12 * 60 * 60)

def _scan_for_count(html: str, label: str) -> None:
    for pattern_name, pattern in [
        ("N lots/Lots", r'(\d[\d,]*)\s*[Ll]ots?\b'),
        ("Lots: N or Lots (N)", r'[Ll]ots?[:\s\(]+(\d[\d,]*)'),
        ("N results", r'(\d[\d,]*)\s*[Rr]esults?\b'),
        ("results count/total-count classes", r'class="[^"]*(?:results?-count|total-count|lot-count)[^"]*"[^>]*>([^<]{0,20})'),
        ("of N (pagination)", r'\bof\s+(\d[\d,]*)\b'),
        ("N Items", r'(\d[\d,]*)\s*[Ii]tems?\b'),
        ("totalCount/totalItems/itemCount JSON keys", r'"(?:total[A-Za-z]*|item[A-Za-z]*)[Cc]ount"\s*:\s*(\d+)'),
        ("totalRecords/recordCount JSON keys", r'"(?:total)?[Rr]ecord[Cc]ount"\s*:\s*(\d+)'),
        ("numberOfItems (schema.org)", r'"numberOfItems"\s*:\s*(\d+)'),
    ]:
        matches = re.findall(pattern, html)
        print(f"[{label}] Pattern {pattern_name!r}: {matches[:15]}")
    for m in list(re.finditer(r'.{40}\d[\d,]{1,4}.{0,15}[Ll]ot.{40}', html))[:10]:
        print(f"[{label}] CONTEXT: ...{m.group(0)!r}...")
    # Specifically identify what surrounds every occurrence of "Items" near a
    # number -- this is what fired above, need the real key name it belongs to.
    for m in list(re.finditer(r'.{60}[Ii]tems?.{20}', html))[:15]:
        print(f"[{label}] ITEMS-CONTEXT: ...{m.group(0)!r}...")
    # Go straight at the literal number that fired the N-Items match earlier
    # (858) -- the generic "Items" dump above was swamped by unrelated CSS.
    for m in list(re.finditer(r'.{50}858.{50}', html))[:10]:
        print(f"[{label}] 858-CONTEXT: ...{m.group(0)!r}...")

async def _debug_find_lot_count_display() -> None:
    """TEMP, runs immediately at startup. The user can see a real total lot
    count displayed directly on a catalog page (e.g. '425 lots') -- if we
    can read that number directly, it's a far better ground-truth signal
    for 'are we actually done paginating' than inferring end-of-results
    from page behavior, which is the whole source of tonight's back-and-
    forth. Checking the real catalog page for this."""
    from playwright.async_api import async_playwright
    wss_url = os.environ.get("BRIGHT_DATA_BROWSER_WSS")
    if not wss_url:
        print("=== LOT COUNT DISPLAY DEBUG: BRIGHT_DATA_BROWSER_WSS not set, skipping ===")
        return
    catalog_url = "https://www.bidspotter.com/en-us/auction-catalogues/bsctabauc/catalogue-id-tab-au10058"  # known 867 lots
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(wss_url, timeout=60000)
            page = await browser.new_page()
            await page.goto(catalog_url, timeout=60000, wait_until="load")
            await page.wait_for_timeout(3000)
            html = await page.content()
            print(f"=== LOT COUNT DISPLAY DEBUG: plain catalog page is {len(html)} bytes ===")
            _scan_for_count(html, "PLAIN PAGE")

            # Now do what the real lot pull does: click "In this auction" +
            # submit, and check the RESULTS page -- far more likely to show
            # a total, since that's the actual results/pagination view.
            try:
                await page.click("#catalogueSearchOption", timeout=5000)
                await page.click("#searchSubmit", timeout=5000)
                await page.wait_for_load_state("load", timeout=30000)
                await page.wait_for_timeout(3000)
                html = await page.content()
                print(f"=== LOT COUNT DISPLAY DEBUG: results page is {len(html)} bytes ===")
                _scan_for_count(html, "RESULTS PAGE")
            except Exception as e:
                print(f"=== LOT COUNT DISPLAY DEBUG: search-submit step failed: {type(e).__name__}: {e} ===")

            await browser.close()
    except Exception as e:
        print(f"=== LOT COUNT DISPLAY DEBUG FAILED: {type(e).__name__}: {e} ===")
    print("=== END LOT COUNT DISPLAY DEBUG ===")

async def _debug_single_catalog_test() -> None:
    """TEMP. Runs ONE catalog, alone, with the daily loop disabled -- no
    competing Bright Data browser session, which is what corrupted the last
    isolated test (211/395 while the full batch ran alongside it). Set
    TEST_CATALOG_SLUG + TEST_CATALOG_URL to pick the target."""
    # Multi-catalog mode: TEST_CATALOG_URLS = comma-separated full catalog
    # URLs, slug derived from each. Runs them one after another (still only
    # ONE Bright Data session at a time), reporting lots + completeness for
    # each -- built to verify LONG catalogs pull fully, without running the
    # whole Job 1 scan or the entire queue.
    # Known VA-verified totals for these 3 test catalogs, from
    # bidspotter_catalog_lots -- used ONLY here to score this test run
    # against ground truth. Not used anywhere in the real pull logic.
    KNOWN_TOTALS = {
        "bsctaylor10010": 954,
        "bsctaylor10009": 953,
        "tab-au10058": 867,
        "grs-au10047": 804,
        "moecke10089": 768,
        "schneider10594": 722,
        "schneider10606": 713,
        "bsckee10064": 702,
        "schneider10603": 701,
        "moecke10088": 701,
    }
    urls_env = os.environ.get("TEST_CATALOG_URLS")
    if urls_env:
        targets = []
        for u in urls_env.split(","):
            u = u.strip().rstrip("/")
            m = re.search(r'catalogue-id-([a-z0-9\-]+)$', u, re.I)
            if m:
                targets.append((u, m.group(1)))
            else:
                print(f"=== CATALOG TEST: could not parse slug from {u}, skipping ===")
        print(f"=== CATALOG TEST: {len(targets)} catalogs, sequential ===")
        for u, s in targets:
            print(f"=== CATALOG TEST START: {u} ===")
            try:
                known = KNOWN_TOTALS.get(s)
                result = await asyncio.wait_for(
                    _fetch_catalog_lots_via_browser(u, s, known_total=known), timeout=900.0
                )
                lots = result.get("lots", [])
                nums = sorted(int(l["lot_number"]) for l in lots if l["lot_number"].isdigit())
                if known is not None:
                    verdict = "PASS (exact match)" if len(lots) == known else f"FAIL (short by {known - len(lots)})" if len(lots) < known else f"FAIL (over by {len(lots) - known})"
                    print(f"=== CATALOG TEST vs KNOWN TOTAL: {len(lots)}/{known} -- {verdict} ===")
                print(f"=== CATALOG TEST RESULT: {len(lots)} lots | range {nums[0] if nums else '-'}-{nums[-1] if nums else '-'} | complete={result.get('complete')} | state={result.get('state')} zip={result.get('zip_code')} country={result.get('country')} | {u} ===")
            except asyncio.TimeoutError:
                print(f"=== CATALOG TEST: timed out at 900s | {u} ===")
            except Exception as e:
                print(f"=== CATALOG TEST FAILED: {type(e).__name__}: {e} | {u} ===")
        print("=== END CATALOG TESTS ===")
        return
    slug = os.environ.get("TEST_CATALOG_SLUG")
    url = os.environ.get("TEST_CATALOG_URL")
    if not (slug and url):
        print("=== SINGLE CATALOG TEST: TEST_CATALOG_SLUG/URL not set, skipping ===")
        return
    print(f"=== SINGLE CATALOG TEST (isolated, nothing else running): {url} ===")
    try:
        result = await asyncio.wait_for(
            _fetch_catalog_lots_via_browser(url, slug), timeout=900.0
        )
        lots = result.get("lots", [])
        nums = sorted(int(l["lot_number"]) for l in lots if l["lot_number"].isdigit())
        print(f"=== SINGLE CATALOG TEST RESULT: {len(lots)} lots | range {nums[0] if nums else '-'}-{nums[-1] if nums else '-'} | state={result.get('state')} zip={result.get('zip_code')} country={result.get('country')} ===")
    except asyncio.TimeoutError:
        print("=== SINGLE CATALOG TEST: timed out at 900s ===")
    except Exception as e:
        print(f"=== SINGLE CATALOG TEST FAILED: {type(e).__name__}: {e} ===")
    print("=== END SINGLE CATALOG TEST ===")

async def _debug_capture_network() -> None:
    """TEMP, runs at startup. Loads a real catalog page + does the search-
    submit flow the real pull does, but instead of reading the final HTML,
    listens to every network response the PAGE ITSELF makes while doing it.
    If BidSpotter's own Angular frontend fetches lot data from a JSON API
    under the hood (rather than the page being server-rendered), this finds
    the real URL -- which would let us hit that API directly instead of
    browser-scraping rendered HTML, sidestepping tonight's entire class of
    timeout/session problems. Set TEST_CAPTURE_NETWORK=1."""
    from playwright.async_api import async_playwright
    from urllib.parse import quote
    wss_url = os.environ.get("BRIGHT_DATA_BROWSER_WSS")
    if not wss_url:
        print("=== NETWORK CAPTURE: BRIGHT_DATA_BROWSER_WSS not set, skipping ===")
        return
    catalog_url = "https://www.bidspotter.com/en-us/auction-catalogues/bsctabauc/catalogue-id-tab-au10058"
    captured = []

    def _on_response(response):
        try:
            ct = response.headers.get("content-type", "")
            url = response.url
            if "json" in ct.lower() or re.search(r'/api/|/lots?/|search', url, re.I):
                captured.append((response.status, ct, url))
        except Exception:
            pass

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(wss_url, timeout=120000)
            page = await browser.new_page()
            page.on("response", _on_response)

            print(f"=== NETWORK CAPTURE: loading {catalog_url} ===")
            await page.goto(catalog_url, timeout=120000, wait_until="load")
            await page.wait_for_timeout(3000)
            print(f"=== NETWORK CAPTURE: {len(captured)} candidate responses after initial load ===")

            print("=== NETWORK CAPTURE: doing search-submit (In this auction) ===")
            try:
                await page.click("#catalogueSearchOption", timeout=5000)
                await page.click("#searchSubmit", timeout=5000)
                await page.wait_for_load_state("load", timeout=30000)
                await page.wait_for_timeout(3000)
            except Exception as e:
                print(f"=== NETWORK CAPTURE: search-submit failed: {type(e).__name__}: {e} ===")
            print(f"=== NETWORK CAPTURE: {len(captured)} candidate responses after search-submit ===")

            print("=== NETWORK CAPTURE: navigating to page=2 ===")
            try:
                path_part = catalog_url.replace("https://www.bidspotter.com", "")
                where_to_search = quote(path_part, safe="")
                page2_url = f"{catalog_url}?searchTerm=&whereToSearch={where_to_search}&page=2"
                await page.goto(page2_url, timeout=120000, wait_until="load")
                await page.wait_for_timeout(3000)
            except Exception as e:
                print(f"=== NETWORK CAPTURE: page=2 nav failed: {type(e).__name__}: {e} ===")
            print(f"=== NETWORK CAPTURE: {len(captured)} candidate responses after page=2 ===")

            # Found "en-us/lots-images?take=2&skip=1&imageSize=175" tied to
            # a real auction GUID -- probe sibling endpoint names that might
            # return actual lot data (title/number), not just images, using
            # the SAME take/skip convention. Fired from inside the page via
            # fetch() so the browser's own cookies/session/WAF-clearance
            # carry over automatically -- a raw new request from us would
            # hit the WAF challenge again.
            auction_id_match = re.search(r'auctionid=([a-f0-9\-]{36})', str(captured))
            auction_id = auction_id_match.group(1) if auction_id_match else None
            if auction_id:
                print(f"=== NETWORK CAPTURE: probing lot-data API variants for auctionid={auction_id} ===")
                candidate_paths = [
                    f"/en-us/lots?take=5&skip=0&auctionid={auction_id}",
                    f"/en-us/lots-data?take=5&skip=0&auctionid={auction_id}",
                    f"/en-us/auction-catalogues/lots?take=5&skip=0&auctionid={auction_id}",
                    f"/en-us/catalogue/lots?take=5&skip=0&auctionid={auction_id}",
                    f"/en-us/lot/search?take=5&skip=0&auctionid={auction_id}",
                ]
                for path in candidate_paths:
                    full_url = f"https://www.bidspotter.com{path}"
                    try:
                        result = await page.evaluate(
                            """async (url) => {
                                const r = await fetch(url, {credentials: 'include'});
                                const text = await r.text();
                                return {status: r.status, ct: r.headers.get('content-type'), body: text.slice(0, 500)};
                            }""",
                            full_url
                        )
                        print(f"[PROBE] {result.get('status')} | {result.get('ct')} | {full_url}")
                        print(f"[PROBE BODY] {result.get('body')!r}")
                    except Exception as e:
                        print(f"[PROBE FAILED] {full_url}: {type(e).__name__}: {e}")
            else:
                print("=== NETWORK CAPTURE: no auction GUID found in captured URLs, cannot probe ===")

            # All 5 guessed endpoint names failed -- read the ACTUAL Angular
            # JS source that renders the search results (these exact files
            # were captured above) and grep for the real endpoint URL it
            # calls, instead of guessing more names.
            js_files = [
                "https://www.bidspotter.com/js/controls/search?v=0INMo08eeB72t5JMHu8K8UxJoTZYESW1BUFC4pXK1SY1",
                "https://www.bidspotter.com/js/lot/auctioncataloguedetails?v=FBUR22a_mlT2kObYC6FaCuNfyxNMWO4bIptGgS9RSs41",
                "https://www.bidspotter.com/js/controls/searchbox?v=1fgPenCdjBFuZ0NRdz0dhS7voK16PpPCyk0O2H6374I1",
                "https://www.bidspotter.com/js/controls/searchassist?v=U1rG3LQbQ1QAoRYxbC594TLlSVIlq66PVHI5VQP1yTs1",
            ]
            for js_url in js_files:
                print(f"=== NETWORK CAPTURE: fetching JS source {js_url} ===")
                try:
                    js_text = await page.evaluate(
                        """async (url) => {
                            const r = await fetch(url, {credentials: 'include'});
                            return await r.text();
                        }""",
                        js_url
                    )
                    # Look for API-looking paths in the source: anything
                    # under /en-us/ that isn't an obvious static asset, plus
                    # explicit fetch/ajax/get/post calls with a URL string.
                    found_paths = set(re.findall(r'["\'](/en-us/[a-zA-Z0-9_\-/]+)["\']', js_text))
                    ajax_calls = re.findall(r'(?:url|action)\s*[:=]\s*["\']([^"\']{5,80})["\']', js_text)
                    print(f"[JS SOURCE {js_url}] length={len(js_text)} chars")
                    if found_paths:
                        print(f"[JS SOURCE PATHS] {sorted(found_paths)[:30]}")
                    if ajax_calls:
                        print(f"[JS SOURCE URL/ACTION FIELDS] {ajax_calls[:30]}")
                    if not found_paths and not ajax_calls:
                        print(f"[JS SOURCE SNIPPET] {js_text[:400]!r}")
                except Exception as e:
                    print(f"[JS SOURCE FAILED] {js_url}: {type(e).__name__}: {e}")

            await browser.close()

        print(f"=== NETWORK CAPTURE: {len(captured)} TOTAL candidate responses ===")
        for status, ct, url in captured[:40]:
            print(f"[CAPTURED] {status} | {ct} | {url}")
    except Exception as e:
        print(f"=== NETWORK CAPTURE FAILED: {type(e).__name__}: {e} ===")
    print("=== END NETWORK CAPTURE ===")

@app.on_event("startup")
async def _start_bidspotter_scan_loop():
    if os.environ.get("TEST_CAPTURE_NETWORK"):
        asyncio.create_task(_debug_capture_network())
    elif os.environ.get("TEST_FIND_LOT_COUNT"):
        asyncio.create_task(_debug_find_lot_count_display())
    elif os.environ.get("TEST_CATALOG_SLUG") or os.environ.get("TEST_CATALOG_URLS"):
        # Isolated single-catalog mode -- daily loop stays OFF so there is
        # exactly ONE Bright Data browser session, no competition.
        asyncio.create_task(_debug_single_catalog_test())
    else:
        asyncio.create_task(_daily_bidspotter_scan_loop())

@app.api_route("/api/updates/trigger-scan", methods=["GET", "POST"])
async def api_trigger_scan():
    """Manual, on-demand trigger for debugging -- runs the exact same scan
    the background loop runs, immediately, and returns exactly what
    happened (including the real error, if any) directly in the response."""
    supabase_client, business_id = _require_config()
    status_row = await _run_bidspotter_scan_for_business(supabase_client, business_id)

@app.api_route("/api/updates/backfill-locations", methods=["GET", "POST"])
async def api_backfill_locations():
    """One-off, on-demand backfill for catalogs already sitting in
    bidspotter_auto_catalog_lots with missing state/zip/country -- fixes
    them WITHOUT touching the main scan loop or re-pulling any lots.
    Completely separate from the background loop: safe to call anytime,
    does not compete with or interrupt whatever the main loop is doing.
    For each affected catalog: fetches ONLY the catalog root page (cheap,
    no search-submit, no pagination) to get real state/zip/country.
    - If it turns out to be non-US: DELETES those rows entirely (this data
      should never have been staged -- the original non-US skip couldn't
      fire because location capture had failed and returned nothing).
    - If real US location is found: UPDATEs all rows for that catalog_url.
    - If it still can't be determined after 3 tries: leaves rows as-is and
      reports it, does not guess."""
    from playwright.async_api import async_playwright
    supabase_client, business_id = _require_config()
    wss_url = os.environ.get("BRIGHT_DATA_BROWSER_WSS")
    if not wss_url:
        return {"error": "BRIGHT_DATA_BROWSER_WSS not set"}

    rows = supabase_client.table("bidspotter_auto_catalog_lots") \
        .select("catalog_url") \
        .eq("business_id", business_id) \
        .is_("state", "null") \
        .is_("country", "null") \
        .execute()
    catalog_urls = sorted(set(r["catalog_url"] for r in rows.data))
    results = []

    def _itemprop_from(html: str, name: str) -> str:
        m = re.search(rf'<[^>]*itemprop="{name}"[^>]*\scontent="([^"]*)"', html)
        if not m:
            m = re.search(rf'<[^>]*itemprop="{name}"[^>]*>([^<]*)<', html)
        val = (m.group(1).strip() if m else "")
        return val if val and val != "." else ""

    def _rebuild_full_url(stored_url: str) -> str:
        # Stored catalog_url has https:// and slashes stripped -- rebuild a
        # real fetchable URL: httpswww.bidspotter.comen-us... ->
        # https://www.bidspotter.com/en-us/...
        u = stored_url.replace("httpswww.bidspotter.comen-us", "https://www.bidspotter.com/en-us/", 1)
        u = re.sub(r'(auction-catalogues)([a-z0-9]+)(catalogue-id-)', r'\1/\2/\3', u)
        return u

    try:
        async with async_playwright() as pw:
            for stored_url in catalog_urls:
                full_url = _rebuild_full_url(stored_url)
                state = zip_code = country = None
                for attempt in range(1, 4):
                    browser = None
                    try:
                        browser = await pw.chromium.connect_over_cdp(wss_url, timeout=120000)
                        page = await browser.new_page()
                        await page.goto(full_url, timeout=120000, wait_until="domcontentloaded")
                        await page.wait_for_timeout(2500)
                        html = await page.content()
                        state = _itemprop_from(html, "addressRegion") or None
                        zip_code = _itemprop_from(html, "postalCode") or None
                        country = _itemprop_from(html, "addressCountry") or None
                        if state or zip_code or country:
                            break
                    except Exception as e:
                        print(f"Backfill attempt {attempt}/3 failed for {full_url}: {type(e).__name__}: {e}")
                    finally:
                        if browser is not None:
                            try:
                                await browser.close()
                            except Exception:
                                pass

                if country and country.lower() not in ("united states", "usa", "us"):
                    del_result = supabase_client.table("bidspotter_auto_catalog_lots") \
                        .delete().eq("business_id", business_id).eq("catalog_url", stored_url).execute()
                    results.append({"catalog_url": stored_url, "action": "DELETED (non-US)", "country": country})
                elif state or zip_code or country:
                    supabase_client.table("bidspotter_auto_catalog_lots") \
                        .update({"state": state, "zip_code": zip_code, "country": country}) \
                        .eq("business_id", business_id).eq("catalog_url", stored_url).execute()
                    results.append({"catalog_url": stored_url, "action": "UPDATED", "state": state, "zip_code": zip_code, "country": country})
                else:
                    results.append({"catalog_url": stored_url, "action": "STILL UNKNOWN after 3 tries"})
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "partial_results": results}

    return {"catalogs_processed": len(catalog_urls), "results": results}

    return status_row

import threading
_cache_lock = threading.Lock()
_cache_store = {}  # key -> (value, expires_at_monotonic)

def _cached(key: str, ttl_seconds: float, compute_fn):
    """Simple shared TTL cache for the read endpoints below. REAL FIX for
    the outage today: every single page load/poll (from every open tab,
    every user) was hitting the database fresh and re-sorting/re-paginating
    the same rows from scratch -- 'extract vs refresh' as it should be:
    compute once, cache it, serve the cached copy to everyone until it's
    actually stale, instead of recomputing on every single request. The
    underlying data only changes when a VA uploads something or the
    background jobs write new rows, not multiple times a second, so a
    short TTL is more than fresh enough. A cache miss right after
    expiry can race across threads and compute twice -- fine, still a
    tiny fraction of the load this replaces."""
    now = time.monotonic()
    with _cache_lock:
        entry = _cache_store.get(key)
        if entry is not None and entry[1] > now:
            return entry[0]
    value = compute_fn()
    with _cache_lock:
        _cache_store[key] = (value, now + ttl_seconds)
    return value

def _invalidate_cache(prefix: str = None):
    """Call after any write that should be reflected immediately (uploads,
    Job 1/2/3 writes) instead of waiting out the TTL. prefix=None clears
    everything."""
    with _cache_lock:
        if prefix is None:
            _cache_store.clear()
        else:
            for k in [k for k in _cache_store if k.startswith(prefix)]:
                del _cache_store[k]

_READ_CACHE_TTL = 30  # seconds -- data only changes on upload/background-job writes, not multiple times a second


@app.get("/api/updates-to-make")
def api_updates_to_make():
    """Powers the 'Updates to Make' box -- unresolved new catalogs and
    reactivated (was-blank, now-has-content) catalogs, newest flagged first.
    Excludes kind='blank' -- those are confirmed to have zero lots on
    BidSpotter right now, so there's nothing actionable for a VA to do yet;
    they show in /api/confirmed-blank instead so they stay visible rather
    than just disappearing."""
    supabase_client, business_id = _require_config()
    def _compute():
        return (supabase_client.table("catalog_updates_queue").select("*")
                .eq("business_id", business_id).eq("resolved", False).neq("kind", "blank")
                .order("first_flagged_at", desc=True).execute().data or [])
    rows = _cached(f"updates-to-make:{business_id}", _READ_CACHE_TTL, _compute)
    return {"updates": rows}


@app.get("/api/confirmed-blank")
def api_confirmed_blank():
    """Catalogs Job 1 has directly verified have zero lots posted on
    BidSpotter right now (kind='blank') -- kept visible here instead of
    being silently deleted, so there's a real record of what was checked.
    Not actionable for a VA (nothing to upload yet), but transparent. Will
    naturally move to Updates-to-Make on its own the next time Job 1 or 2
    re-checks it and finds real content."""
    supabase_client, business_id = _require_config()
    def _compute():
        return (supabase_client.table("catalog_updates_queue").select("*")
                .eq("business_id", business_id).eq("resolved", False).eq("kind", "blank")
                .order("first_flagged_at", desc=True).execute().data or [])
    rows = _cached(f"confirmed-blank:{business_id}", _READ_CACHE_TTL, _compute)
    return {"blank": rows}


@app.get("/api/catalogs")
def api_catalogs():
    supabase_client, business_id = _require_config()
    def _compute():
        return _fetch_all_paginated(lambda: supabase_client.table("auction_catalogs").select("*").eq("business_id", business_id))
    rows = _cached(f"catalogs:{business_id}", _READ_CACHE_TTL, _compute)
    return {"catalogs": rows}


@app.get("/api/lots")
def api_lots(catalog_url: str = None):
    supabase_client, business_id = _require_config()
    def build_query():
        q = supabase_client.table("bidspotter_catalog_lots").select("*").eq("business_id", business_id)
        if catalog_url:
            q = q.eq("catalog_url", catalog_url)
        return q.order("last_seen_at", desc=True)
    rows = _cached(f"lots:{business_id}:{catalog_url or ''}", _READ_CACHE_TTL, lambda: _fetch_all_paginated(build_query))
    return {"lots": rows}

@app.get("/api/bright-lots")
def api_bright_lots(catalog_url: str = None):
    """Purely the automated Bright Data pull's own data -- a separate view
    so it can be compared directly against the VA's manually-uploaded
    bidspotter_catalog_lots data above, without the two ever mixing."""
    supabase_client, business_id = _require_config()
    def build_query():
        q = supabase_client.table("bidspotter_auto_catalog_lots").select("*").eq("business_id", business_id)
        if catalog_url:
            q = q.eq("catalog_url", catalog_url)
        return q.order("last_seen_at", desc=True)
    rows = _cached(f"bright-lots:{business_id}:{catalog_url or ''}", _READ_CACHE_TTL, lambda: _fetch_all_paginated(build_query))
    return {"lots": rows}


def _create_upload_record(supabase_client, business_id: str, filename: str, contents: bytes, title: str = "", explicit_catalog_url: str = "") -> dict:
    """Durable, SYNCHRONOUS step run immediately when a file is received --
    BEFORE it ever touches the in-memory processing queue. Real bug this
    fixes: uploads used to only get logged/stored once the queue worker
    actually got around to them, which meant a redeploy (container
    restart wipes the in-memory queue instantly, no trace) could silently
    lose an uploaded file completely -- the client got a fake 'queued: true'
    success response, and there was nothing anywhere to show the upload
    ever existed. Now the file is safely in storage and logged as
    'processing' the moment it's received; the queue only decides WHEN the
    (recoverable) parsing work happens, not WHETHER the upload is durable.

    explicit_catalog_url: a SECOND real bug fix. Without this, catalog_url
    was always derived from the uploaded FILENAME -- which only resolves
    the matching queue/blank-list entry if the file happens to be named
    exactly like BidSpotter's mangled URL string. Any normal filename (the
    auctioneer's own PDF name, anything human-readable) silently fails to
    match, and the item never clears -- confirmed live, this is why
    uploads kept not disappearing from the list. When the frontend knows
    which catalog a PDF is for (uploading from a specific row in Updates
    to Make / Needs Update), it now passes that real catalog_url through
    directly instead of leaving it to a filename guess."""
    title = title.strip() or filename
    raw_name = filename.rsplit(".", 1)[0] if filename else str(uuid.uuid4())
    catalog_key = re.sub(r'[^A-Za-z0-9._-]', '_', raw_name)
    catalog_url = explicit_catalog_url.strip() if explicit_catalog_url and explicit_catalog_url.strip() else catalog_key

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
        if log_id:
            supabase_client.table("auction_pdf_uploads").update({"storage_path": storage_path}).eq("id", log_id).execute()
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

    return {"log_id": log_id, "catalog_url": catalog_url, "catalog_key": catalog_key, "storage_path": storage_path}


def _process_one_pdf(supabase_client, business_id: str, filename: str, contents: bytes,
                       title: str = "", auctioneer: str = "", end_date: str = "", state: str = "", zip_code: str = "",
                       existing_record: dict = None) -> dict:
    """Does everything for one catalog PDF: extracts text, pulls every lot
    out via Gemini (the sole, authoritative parser -- see
    _extract_lots_via_gemini), ingests into bidspotter_catalog_lots/
    auction_catalogs, and updates the upload log with the real outcome.
    Shared by both the single-file upload endpoint and the zip-batch
    endpoint, so a zip upload behaves identically to uploading each file
    one at a time -- same parsing, same empty-catalog handling, same
    was_ever_empty flag, same logging.

    existing_record: pass the dict from _create_upload_record if the
    durable log+storage step already ran (the normal path now) -- skips
    redoing it. If not passed, falls back to doing it here (kept for any
    other caller that hasn't been updated yet)."""
    title = title.strip() or filename
    if existing_record:
        log_id = existing_record["log_id"]
        catalog_url = existing_record["catalog_url"]
        storage_path = existing_record["storage_path"]
    else:
        rec = _create_upload_record(supabase_client, business_id, filename, contents, title)
        log_id = rec["log_id"]
        catalog_url = rec["catalog_url"]
        storage_path = rec["storage_path"]

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
    """The file is saved to storage and logged as 'processing' SYNCHRONOUSLY
    here, before anything touches the queue -- if the container restarts
    while this file is still waiting its turn, the file and its record
    survive intact and get picked up by the recovery pass on next startup,
    instead of vanishing with a fake 'queued: true' success response and no
    trace anywhere (a real incident, not hypothetical -- this exact thing
    happened during tonight's redeploys). Only the actual Gemini parsing
    step (the slow, recoverable part) waits on _pdf_processing_queue."""
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
    explicit_catalog_url = (form.get("catalog_url") or "").strip()

    contents = await file.read()
    record = _create_upload_record(supabase_client, business_id, file.filename, contents, title, explicit_catalog_url)

    def _run_and_backfill():
        result = _process_one_pdf(supabase_client, business_id, file.filename, contents,
                                   title, auctioneer, end_date, state, zip_code, existing_record=record)
        if result.get("needs_backfill") and result.get("raw_text"):
            _backfill_catalog_metadata(supabase_client, business_id, result["catalog_url"],
                                        result["raw_text"], file.filename, state, zip_code, end_date)
        _invalidate_cache(f"catalogs:{business_id}")
        _invalidate_cache(f"pdf-uploads:{business_id}")
        _invalidate_cache(f"needs-update:{business_id}")
        _invalidate_cache(f"updates-to-make:{business_id}")

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
    for i, (filename, contents, record) in enumerate(pdf_entries):
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
            future = ex.submit(_process_one_pdf, supabase_client, business_id, filename, contents,
                                "", "", "", "", "", record)
            try:
                result = future.result(timeout=1500)
            except concurrent.futures.TimeoutError:
                ex.shutdown(wait=False)
                print(f"Zip batch: {filename} exceeded the 25-minute hard watchdog timeout -- "
                      f"giving up on this file and moving on (the stuck attempt may still finish "
                      f"on its own later and overwrite this with a real result)")
                try:
                    # UPDATE the existing durable record (created up front),
                    # not a fresh insert -- a fresh insert here would leave
                    # the original 'processing' row orphaned forever
                    # alongside a second error row for the same file.
                    if record.get("log_id"):
                        supabase_client.table("auction_pdf_uploads").update({
                            "status": "error",
                            "error_message": "Hard watchdog timeout: this file's processing did not "
                                              "complete within 25 minutes, so it was skipped to let the "
                                              "rest of the batch continue. Try uploading this one file "
                                              "again on its own.",
                        }).eq("id", record["log_id"]).execute()
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

    # Same durability fix as the single-PDF endpoint, same real incident
    # class this prevents -- a zip batch can be hundreds of files, so this
    # matters even more here: every file is logged + stored NOW, synchron-
    # ously, before any of them touch the queue. A restart mid-batch used
    # to silently drop every file that hadn't been reached yet; now they're
    # all durable immediately and the startup recovery pass picks up
    # whichever ones the restart caught mid-processing.
    entries_with_records = []
    for filename, contents in pdf_entries:
        record = _create_upload_record(supabase_client, business_id, filename, contents)
        entries_with_records.append((filename, contents, record))

    _pdf_processing_queue.put_nowait(lambda: _process_zip_batch(supabase_client, business_id, entries_with_records))
    return {"ok": True, "queued": len(pdf_entries)}


@app.get("/api/pdf-uploads")
def api_pdf_uploads():
    supabase_client, business_id = _require_config()
    def _compute():
        return _fetch_all_paginated(lambda: supabase_client.table("auction_pdf_uploads").select("*")
                                     .eq("business_id", business_id).order("uploaded_at", desc=True))
    uploads = _cached(f"pdf-uploads:{business_id}", _READ_CACHE_TTL, _compute)
    return {"uploads": uploads}


@app.get("/api/needs-update")
def api_needs_update():
    supabase_client, business_id = _require_config()
    def _compute():
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

        # REAL BUG FIXED HERE, same class as Job 1's known_urls bug: this only
        # ever checked auction_catalogs (built from manual VA uploads) -- it had
        # zero awareness that the AUTOMATED pull might have already landed real
        # lots for the same catalog in bidspotter_auto_catalog_lots. A catalog
        # the automation successfully pulled would sit on this "needs update"
        # list forever, since nothing here ever looked at the other table.
        try:
            auto_rows = _fetch_all_paginated(lambda: supabase_client.table("bidspotter_auto_catalog_lots").select("catalog_url").eq("business_id", business_id))
            has_real_lots_local = has_real_lots | {r["catalog_url"] for r in auto_rows}
        except Exception as e:
            print(f"api_needs_update: failed to check automated staging table (non-fatal): {e}")
            has_real_lots_local = has_real_lots

        needs_update = [u for u in latest_by_catalog.values()
                         if u.get("status") == "empty" and u.get("catalog_url") not in has_real_lots_local]
        for u in needs_update:
            u["end_date"] = u.get("end_date") or end_date_by_catalog.get(u.get("catalog_url"))
        needs_update.sort(key=lambda u: u.get("uploaded_at") or "", reverse=True)
        return needs_update

    needs_update = _cached(f"needs-update:{business_id}", _READ_CACHE_TTL, _compute)
    return {"needs_update": needs_update}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
