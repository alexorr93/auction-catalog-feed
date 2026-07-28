# Auction Catalog Feed

A standalone, fully public app for the BidSpotter auction catalog feed —
upload catalog PDFs, browse catalogs and lots, no login required. Built
deliberately as its own codebase, separate from the main Lister app: it
has no Settings page, no eBay/Shopify credentials, and no other Lister
code at all, so this URL can be shared with anyone without exposing
anything sensitive.

It reads and writes the same Supabase tables the main app's Auction
Monitor feature already uses (`bidspotter_catalog_lots`, `auction_catalogs`,
`auction_pdf_uploads`) — this is a second, public front door onto the same
data, not a separate copy of it.

## Deploying on Railway

1. Create a new Railway service, pointed at this repo (branch: `main`).
2. Set these environment variables on the service:
   - `SUPABASE_URL` — same value as the main Lister app uses
   - `SUPABASE_KEY` — same value as the main Lister app uses
   - `GEMINI_API_KEY` — same value as the main Lister app uses (used for the
     lot/metadata auto-extraction fallback on non-BidSpotter PDF layouts)
   - `OWNER_EMAIL` — optional, defaults to `precisionindustrialmail@gmail.com`.
     This is the one business's data this app serves — there's no login here
     to derive it from, so it's resolved once at startup instead.
3. Railway should auto-detect this as a Python app (`requirements.txt` +
   `main.py`) and run it with the standard Python buildpack. No Dockerfile
   needed.
4. Once deployed, the whole app is public at that Railway URL — no login
   page, nothing gated. Share the URL with anyone who needs to view or
   add to the catalog feed.

## What's intentionally NOT in this codebase

No auth, no user accounts, no Settings page, no eBay API keys, no Shopify
keys, no Intake/Fast Scan/Upload/Inventory/Financials/Analytics — none of
the rest of Lister. This app can only ever read/write the auction catalog
feed tables, nothing else, by design.
