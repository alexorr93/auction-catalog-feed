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
import uuid
from datetime import datetime, timezone

import fitz  # PyMuPDF
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from supabase import create_client

app = FastAPI()
templates = Jinja2Templates(directory="templates")
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

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

def _extract_lots_via_gemini(raw_text: str, filename: str, on_chunk_lots=None) -> list:
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
    only the immediate persistence attempt failed, not the extraction."""
    import google.generativeai as genai
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        return []
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    chunk_size = 80000
    chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size)] or [raw_text]

    seen_lot_numbers = set()
    combined = []
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
        try:
            # Real bug fixed here: this had no timeout at all -- if the
            # Gemini call hangs for any reason (network issue, throttling,
            # anything), there was nothing to make it give up, so the whole
            # background task (and the catalog it was processing) could sit
            # stuck indefinitely with zero visible error. 120s is generous
            # for a single ~80k-char chunk but still finite -- a timeout
            # here is caught below like any other per-chunk failure and
            # just skips to the next chunk instead of hanging the catalog.
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
        except Exception as e:
            print(f"Gemini lot extraction failed for {filename} (chunk {chunk_i + 1}/{len(chunks)}): {e}")
            # Keep going with whatever other chunks succeed, rather than
            # losing the whole catalog over one bad chunk
            continue

    return combined


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
        print(f"PDF storage warning (upload still proceeds): {e}")
        storage_path = None

    try:
        doc = fitz.open(stream=contents, filetype="pdf")
        raw_text = ""
        for page in doc:
            raw_text += page.get_text() + "\n"
        doc.close()

        if not raw_text.strip():
            raise ValueError("No text found in this PDF (may be scanned images with no text layer -- not supported yet)")

        # Always extract via Gemini (chunked for large catalogs, so nothing
        # gets truncated) -- no heuristic deciding whether a shortcut can be
        # trusted, because every version of that heuristic tried so far had
        # a real blind spot (a catalog whose lot lines don't start with a
        # digit at all would report the regex as "complete" even while
        # missing every single lot). Reads what's actually in the PDF every
        # time instead of guessing when that's necessary.
        lots = _extract_lots_via_gemini(raw_text, filename)

        if not lots:
            if log_id:
                supabase_client.table("auction_pdf_uploads").update({
                    "status": "empty", "storage_path": storage_path, "parsed_lot_count": 0,
                }).eq("id", log_id).execute()
            return {"ok": True, "filename": filename, "lots_parsed": 0, "empty": True, "catalog_url": catalog_url}

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

        lots = _extract_lots_via_gemini(raw_text, filename, on_chunk_lots=_on_chunk)

        if not lots:
            if log_id:
                supabase_client.table("auction_pdf_uploads").update({
                    "status": "empty", "storage_path": storage_path, "parsed_lot_count": 0,
                }).eq("id", log_id).execute()
            return {"ok": True, "filename": filename, "lots_parsed": 0, "empty": True, "catalog_url": catalog_url}

        # written_count tracks what actually landed via the per-chunk
        # callback, which is the real durable number -- if any single
        # chunk's immediate write failed (rare, logged, kept in the final
        # `lots` list as a fallback rather than lost), written_count and
        # len(lots) can differ slightly; written_count is what's actually
        # in the database right now, so it's what gets reported and logged.
        if log_id:
            supabase_client.table("auction_pdf_uploads").update({
                "status": "success", "storage_path": storage_path, "parsed_lot_count": written_count,
            }).eq("id", log_id).execute()

        needs_backfill = not (state and zip_code and end_date)

        return {"ok": True, "filename": filename, "lots_parsed": written_count, "catalog_url": catalog_url,
                "needs_backfill": needs_backfill, "raw_text": raw_text if needs_backfill else None}

    except Exception as e:
        if log_id:
            supabase_client.table("auction_pdf_uploads").update({
                "status": "error", "storage_path": storage_path, "error_message": str(e),
            }).eq("id", log_id).execute()
        return {"ok": False, "filename": filename, "error": str(e)}


@app.post("/api/upload-pdf")
async def upload_pdf(request: Request, background_tasks: BackgroundTasks):
    """Real bug fixed here: this used to await _process_one_pdf directly,
    holding the HTTP request open for the entire duration -- fine for a
    small catalog, but now that Gemini is the sole lot extractor (chunked,
    possibly several sequential API calls for a large catalog), a real
    700+ lot PDF can take long enough that Railway's proxy or the browser
    itself kills the connection mid-request ("Failed to fetch", even though
    the upload was actually still working server-side). The zip-upload
    endpoint already solved this by running in the background and letting
    the Upload Log fill in live -- this now does the exact same thing for
    a single file, instead of blocking the request on it."""
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

    background_tasks.add_task(_run_and_backfill)
    return {"ok": True, "queued": True, "filename": file.filename}


def _process_zip_batch(supabase_client, business_id: str, pdf_entries: list) -> None:
    """Runs in the background (in its own thread, via BackgroundTasks -- doesn't
    block other requests while this runs). Processes every PDF found in the zip
    one at a time, same logic as a single upload for each. Progress is visible
    the whole time through the existing Upload Log / Needs Update / Catalogs
    views -- no separate progress UI needed, since each file logs itself as it
    completes, exactly like uploading them one by one would."""
    print(f"Zip batch: starting {len(pdf_entries)} catalog(s)")
    for i, (filename, contents) in enumerate(pdf_entries):
        try:
            result = _process_one_pdf(supabase_client, business_id, filename, contents)
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
async def upload_zip(request: Request, background_tasks: BackgroundTasks):
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

    background_tasks.add_task(_process_zip_batch, supabase_client, business_id, pdf_entries)
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
    catalog_rows = _fetch_all_paginated(lambda: supabase_client.table("auction_catalogs").select("catalog_url,lot_count")
                                          .eq("business_id", business_id))
    has_real_lots = {c["catalog_url"] for c in catalog_rows if (c.get("lot_count") or 0) > 0}

    needs_update = [u for u in latest_by_catalog.values()
                     if u.get("status") == "empty" and u.get("catalog_url") not in has_real_lots]
    needs_update.sort(key=lambda u: u.get("uploaded_at") or "", reverse=True)
    return {"needs_update": needs_update}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
