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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# This app serves exactly one business's feed -- there's no login to derive
# that from, so it's resolved once at startup from a known email instead of
# a raw UUID being hardcoded here (keeps this file portable/readable without
# needing to already know an opaque id).
OWNER_EMAIL = os.getenv("OWNER_EMAIL", "precisionindustrialmail@gmail.com")
_business_row = supabase.table("businesses").select("id").eq("email", OWNER_EMAIL).single().execute()
BUSINESS_ID = _business_row.data["id"]
print(f"Auction Catalog Feed serving business_id={BUSINESS_ID} ({OWNER_EMAIL})")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Parsing / extraction -- ported as-is from the main Lister app's Auction
# Monitor feature (auction-monitor-dev branch), same logic, same behavior.
# ---------------------------------------------------------------------------

def _parse_print_catalog_lots(raw_text: str) -> list:
    """Parses a BidSpotter 'Print Catalog' export's Lot/Description lines into
    structured rows. The regex only needs the lot number to be the first
    token on a line, so cleanly-ordered catalog text (one lot per line)
    parses reliably."""
    rows = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+[A-Za-z]?)\s+(.+)$', line)
        if m:
            rows.append({"lot_number": m.group(1), "description": m.group(2)[:2000]})
    return rows


def _extract_lots_via_gemini(raw_text: str, filename: str) -> list:
    """Fallback for catalog PDFs whose layout doesn't match the regex parser
    (built for BidSpotter's own Print Catalog text format) -- asks Gemini to
    pull lot number + description pairs out of arbitrary catalog text
    instead. Only called when the fast, free regex pass finds too few lots
    to trust."""
    import google.generativeai as genai
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        return []
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""This is raw text extracted from an auction catalog PDF named "{filename}".
Pull out every individual lot as a lot number and its description. Auction catalogs
list lots sequentially, usually as "LOT ###" or "Lot ###:" or similar, followed by a
description of the item(s) in that lot. Skip page headers, footers, terms & conditions,
and anything that isn't an actual lot listing.

Return ONLY a JSON array, no other text, in this exact shape:
[{{"lot_number": "123", "description": "..."}}, ...]

Text:
{raw_text[:100000]}"""
    try:
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
        return [{"lot_number": str(r.get("lot_number", "")).strip(), "description": (r.get("description") or "")[:2000]}
                for r in parsed if r.get("lot_number") and r.get("description")]
    except Exception as e:
        print(f"Gemini lot extraction failed for {filename}: {e}")
        return []


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
        resp = model.generate_content(prompt)
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


def _ingest_one_catalog(catalog_url: str, raw_text: str, meta: dict) -> dict:
    """Writes into bidspotter_catalog_lots (the dedicated table for this feed --
    deliberately separate from auction_lots, which belongs to a different,
    unrelated feature in the main app) plus auction_catalogs for the
    catalog-level summary row."""
    lots = _parse_print_catalog_lots(raw_text)
    if not lots:
        return {"catalog_url": catalog_url, "parsed": 0}

    now_iso = datetime.now(timezone.utc).isoformat()
    for lot in lots:
        record = {
            "business_id": BUSINESS_ID,
            "catalog_url": catalog_url,
            "lot_number": lot["lot_number"],
            "description": lot["description"],
            "last_seen_at": now_iso,
            "state": meta.get("state") or None,
            "zip_code": meta.get("zip_code") or None,
            "date": meta.get("end_date") or None,
        }
        existing = supabase.table("bidspotter_catalog_lots").select("id")\
            .eq("business_id", BUSINESS_ID).eq("catalog_url", catalog_url).eq("lot_number", lot["lot_number"]).limit(1).execute()
        if existing.data:
            supabase.table("bidspotter_catalog_lots").update(record).eq("id", existing.data[0]["id"]).execute()
        else:
            supabase.table("bidspotter_catalog_lots").insert(record).execute()

    catalog_fields = {k: meta[k] for k in ("title", "auctioneer", "end_date", "state") if meta.get(k)}
    catalog_existing = supabase.table("auction_catalogs").select("id")\
        .eq("business_id", BUSINESS_ID).eq("catalog_url", catalog_url).limit(1).execute()
    catalog_fields.update({"lot_count": len(lots), "lot_count_is_estimate": False, "last_checked_at": now_iso})
    if catalog_existing.data:
        supabase.table("auction_catalogs").update(catalog_fields).eq("id", catalog_existing.data[0]["id"]).execute()
    else:
        catalog_fields.update({"business_id": BUSINESS_ID, "source": "upload", "catalog_url": catalog_url, "first_seen_at": now_iso})
        supabase.table("auction_catalogs").insert(catalog_fields).execute()

    return {"catalog_url": catalog_url, "parsed": len(lots)}


def _backfill_catalog_metadata(catalog_url: str, raw_text: str, filename: str,
                                 state: str, zip_code: str, end_date: str) -> None:
    """Runs AFTER the upload response has already gone out, as a background
    task -- keeps uploads fast. Fills in whichever of state/zip/date the
    form left blank, a few seconds after the VA already sees 'success'."""
    try:
        auto_meta = _extract_catalog_metadata_via_gemini(raw_text, filename)
        patch = {}
        if not state and auto_meta.get("state"):
            patch["state"] = auto_meta["state"]
        if not zip_code and auto_meta.get("zip_code"):
            patch["zip_code"] = auto_meta["zip_code"]
        if not end_date and auto_meta.get("end_date"):
            patch["date"] = auto_meta["end_date"]
        if patch:
            supabase.table("bidspotter_catalog_lots").update(patch)\
                .eq("business_id", BUSINESS_ID).eq("catalog_url", catalog_url).execute()
    except Exception as e:
        print(f"Background metadata backfill failed for {filename} (lots themselves are unaffected): {e}")


# ---------------------------------------------------------------------------
# Routes -- everything here is intentionally public, no auth of any kind.
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("feed.html", {"request": request})


@app.get("/api/catalogs")
async def api_catalogs():
    rows = []
    start = 0
    while True:
        page = supabase.table("auction_catalogs").select("*")\
            .eq("business_id", BUSINESS_ID).range(start, start + 999).execute().data or []
        rows.extend(page)
        if len(page) < 1000:
            break
        start += 1000
    return {"catalogs": rows}


@app.get("/api/lots")
async def api_lots(catalog_url: str = None):
    q = supabase.table("bidspotter_catalog_lots").select("*").eq("business_id", BUSINESS_ID)
    if catalog_url:
        q = q.eq("catalog_url", catalog_url)
    rows = []
    start = 0
    while True:
        page = q.order("last_seen_at", desc=True).range(start, start + 999).execute().data or []
        rows.extend(page)
        if len(page) < 1000:
            break
        start += 1000
    return {"lots": rows}


@app.post("/api/upload-pdf")
async def upload_pdf(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    file = form.get("file")
    if not file or not hasattr(file, "read"):
        raise HTTPException(400, "file is required")
    title = (form.get("title") or "").strip() or file.filename
    auctioneer = (form.get("auctioneer") or "").strip()
    end_date = (form.get("end_date") or "").strip()
    state = (form.get("state") or "").strip()
    zip_code = (form.get("zip_code") or "").strip()

    contents = await file.read()
    raw_name = file.filename.rsplit(".", 1)[0] if file.filename else str(uuid.uuid4())
    catalog_key = re.sub(r'[^A-Za-z0-9._-]', '_', raw_name)
    catalog_url = catalog_key

    log_row = {
        "business_id": BUSINESS_ID, "filename": file.filename, "status": "processing",
        "catalog_url": catalog_url, "catalog_title": title,
    }
    log_res = supabase.table("auction_pdf_uploads").insert(log_row).execute()
    log_id = log_res.data[0]["id"] if log_res.data else None

    storage_path = None
    try:
        storage_path = f"{catalog_key}.pdf"
        supabase.storage.from_("auction-pdfs").upload(
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

        lots = _parse_print_catalog_lots(raw_text)
        if len(lots) < 3:
            gemini_lots = _extract_lots_via_gemini(raw_text, file.filename)
            if len(gemini_lots) > len(lots):
                lots = gemini_lots

        if not lots:
            # Not a parse failure -- the auction exists but has no lots posted
            # yet. Its own distinct status, not lumped in with real errors.
            if log_id:
                supabase.table("auction_pdf_uploads").update({
                    "status": "empty", "storage_path": storage_path, "parsed_lot_count": 0,
                }).eq("id", log_id).execute()
            return {"ok": True, "lots_parsed": 0, "empty": True, "catalog_url": catalog_url}

        meta = {"title": title, "auctioneer": auctioneer, "end_date": end_date, "state": state, "zip_code": zip_code}
        result = _ingest_one_catalog(catalog_url, "\n".join(f"{l['lot_number']} {l['description']}" for l in lots), meta)

        if log_id:
            supabase.table("auction_pdf_uploads").update({
                "status": "success", "storage_path": storage_path, "parsed_lot_count": result.get("parsed", 0),
            }).eq("id", log_id).execute()

        if not (state and zip_code and end_date):
            background_tasks.add_task(_backfill_catalog_metadata, catalog_url, raw_text, file.filename, state, zip_code, end_date)

        return {"ok": True, "lots_parsed": result.get("parsed", 0), "catalog_url": catalog_url}

    except Exception as e:
        if log_id:
            supabase.table("auction_pdf_uploads").update({
                "status": "error", "storage_path": storage_path, "error_message": str(e),
            }).eq("id", log_id).execute()
        raise HTTPException(500, str(e))


@app.get("/api/pdf-uploads")
async def api_pdf_uploads():
    res = supabase.table("auction_pdf_uploads").select("*").eq("business_id", BUSINESS_ID)\
        .order("uploaded_at", desc=True).limit(500).execute()
    return {"uploads": res.data or []}


@app.get("/api/needs-update")
async def api_needs_update():
    all_uploads = supabase.table("auction_pdf_uploads").select("*").eq("business_id", BUSINESS_ID)\
        .order("uploaded_at", desc=True).limit(500).execute().data or []
    latest_by_catalog = {}
    for u in all_uploads:
        key = u.get("catalog_url")
        if not key:
            continue
        existing = latest_by_catalog.get(key)
        if not existing or (u.get("uploaded_at") or "") > (existing.get("uploaded_at") or ""):
            latest_by_catalog[key] = u
    needs_update = [u for u in latest_by_catalog.values() if u.get("status") == "empty"]
    needs_update.sort(key=lambda u: u.get("uploaded_at") or "", reverse=True)
    return {"needs_update": needs_update}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
