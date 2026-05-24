#!/usr/bin/env python3
"""
Farmers Atelier Complete Scraper
Scrapes all products from farmersatelier.com via the Shopify products.json API,
computes image & text embeddings using google/siglip-base-patch16-384 (768-dim),
and uploads everything to Supabase with batch upsert, change detection,
stale product cleanup, and staggered embedding generation.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from hashlib import md5
from io import BytesIO
from typing import Optional

import requests
import torch
from bs4 import BeautifulSoup
from PIL import Image
from supabase import create_client, Client
from transformers import AutoProcessor, AutoModel

# ─── Configuration ───────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get(
    "SUPABASE_URL",
    "https://yqawmzggcgpeyaaynrjk.supabase.co",
)
SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4",
)
SOURCE = "scraper-farmersatelier"
BRAND = "Farmers Atelier"
SECOND_HAND = False
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "google/siglip-base-patch16-384")
EMBEDDING_DIM = 768
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
EUR_TO_USD = float(os.environ.get("EUR_TO_USD", "1.08"))
BASE_URL = "https://farmersatelier.com"
BATCH_SIZE = 50
STAGGER_DELAY_S = 0.5
MAX_RETRIES = 3
STALE_MISS_THRESHOLD = 2
FAILED_LOG = "failed_products.log"

# ─── Category detection ──────────────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    "Knitwear": ["knit"],
    "Hoodies": ["hoodie"],
    "T-Shirts": ["tee", "t-shirt", "t shirt"],
    "Sweaters": ["sweater"],
    "Beanies": ["beanie"],
    "Caps": ["cap"],
    "Bags": ["bag"],
    "Longsleeves": ["long sleeve", "longsleeve"],
    "Shirts": ["shirt"],
}

SIZE_SORT_KEY = {s: i for i, s in enumerate(["XS", "S", "M", "L", "XL", "2XL", "3XL", "4XL"])}


def detect_category(title: str, description: str) -> Optional[str]:
    tl = title.lower()
    dl = (description or "").lower()
    found = []
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", tl) or re.search(r"\b" + re.escape(kw) + r"\b", dl):
                found.append(cat)
                break
    return ", ".join(found) if found else None


def clean_html(html_text: str) -> str:
    if not html_text:
        return ""
    return BeautifulSoup(html_text, "html.parser").get_text(separator=" ", strip=True)


def format_price(price_eur: float) -> str:
    return f"{price_eur:.2f}EUR, {price_eur * EUR_TO_USD:.2f}USD"


def format_sale_price(price_eur: float) -> str:
    return f"{price_eur:.2f}EUR"


def sort_sizes(sizes: list[str]) -> list[str]:
    return sorted(set(sizes), key=lambda x: SIZE_SORT_KEY.get(x, 99))


# ─── Content fingerprint ────────────────────────────────────────────────────

def compute_fingerprint(title: str, description: str, min_price: float,
                        sale_price: Optional[float], image_url: Optional[str],
                        additional_images: list[str], sizes: list[str],
                        variants_available: list[bool]) -> str:
    data = {
        "title": title,
        "description": description[:500],
        "min_price": min_price,
        "sale_price": sale_price,
        "image_url": image_url,
        "additional_images": additional_images,
        "sizes": sorted(sizes),
        "variants_available": variants_available,
    }
    return md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


# ─── Embedding Model ─────────────────────────────────────────────────────────

class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        print(f"  Loading model {model_name} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        print(f"  Model loaded (768-dim)")

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> list[float]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return outputs.pooler_output.cpu().numpy().flatten().tolist()

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        inputs = self.processor(text=text, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(self.device)
        outputs = self.model.get_text_features(**inputs)
        return outputs.pooler_output.cpu().numpy().flatten().tolist()


# ─── Image Downloader ────────────────────────────────────────────────────────

def download_image(url: str, max_retries: int = 3) -> Optional[Image.Image]:
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    [WARN] Image download failed: {e}")
    return None


# ─── Shopify API Fetcher ─────────────────────────────────────────────────────

def fetch_all_products() -> list[dict]:
    all_products = []
    page = 1
    limit = 250
    while True:
        url = f"{BASE_URL}/products.json?limit={limit}&page={page}"
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            products = data.get("products", [])
            if not products:
                break
            all_products.extend(products)
            if len(products) < limit:
                break
            page += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"[ERROR] Failed to fetch page {page}: {e}")
            break
    return all_products


# ─── Supabase Client ─────────────────────────────────────────────────────────

class SupabaseClient:
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)
        self.table = "products"

    def fetch_existing(self) -> list[dict]:
        result = self.client.table(self.table).select("*").eq("source", SOURCE).execute()
        return result.data

    def batch_upsert(self, rows: list[dict]) -> bool:
        try:
            self.client.table(self.table).upsert(rows, on_conflict="source,product_url").execute()
            return True
        except Exception as e:
            raise e

    def update_metadata(self, product_url: str, metadata: dict) -> bool:
        try:
            self.client.table(self.table).update({"metadata": json.dumps(metadata)}).eq("source", SOURCE).eq("product_url", product_url).execute()
            return True
        except Exception as e:
            print(f"    [WARN] Failed to update metadata for {product_url}: {e}")
            return False

    def delete_products(self, urls: list[str]) -> int:
        if not urls:
            return 0
        try:
            result = self.client.table(self.table).delete().eq("source", SOURCE).in_("product_url", urls).execute()
            return len(result.data)
        except Exception as e:
            print(f"    [WARN] Batch delete failed ({len(urls)} items): {e}")
            count = 0
            for url in urls:
                try:
                    r = self.client.table(self.table).delete().eq("source", SOURCE).eq("product_url", url).execute()
                    count += len(r.data)
                except Exception:
                    pass
            return count


# ─── Product processing ──────────────────────────────────────────────────────

def build_text_for_embedding(title: str, description: str, category: Optional[str],
                              gender: Optional[str], price: str, sale: Optional[str],
                              sizes: list[str]) -> str:
    parts = [f"Title: {title}"]
    if description:
        parts.append(f"Description: {description[:500]}")
    if category:
        parts.append(f"Category: {category}")
    if gender:
        parts.append(f"Gender: {gender}")
    parts.append(f"Price: {price}")
    if sale:
        parts.append(f"Sale: {sale}")
    if sizes:
        parts.append(f"Sizes: {', '.join(sizes)}")
    parts.append(f"Brand: {BRAND}")
    return " | ".join(parts)


def build_product_row(product: dict, metadata_extra: Optional[dict] = None) -> Optional[dict]:
    title = product.get("title", "")
    handle = product.get("handle", "")
    product_id = product.get("id")
    body_html = product.get("body_html", "")
    description = clean_html(body_html)
    images = product.get("images", [])
    variants = product.get("variants", [])
    options = product.get("options", [])

    product_url = f"{BASE_URL}/products/{handle}"
    image_url = images[0]["src"] if images else None
    additional_images = [img["src"] for img in images[1:]] if len(images) > 1 else []

    if not image_url:
        return None

    sizes = []
    min_price = None
    sale_price = None
    variants_available = []
    for v in variants:
        price = float(v["price"])
        sizes.append(v["title"])
        variants_available.append(v.get("available", False))
        if min_price is None or price < min_price:
            min_price = price
        if v.get("compare_at_price") and float(v["compare_at_price"]) > price:
            if sale_price is None or price < sale_price:
                sale_price = price

    if min_price is None:
        return None

    price_str = format_price(min_price)
    sale_str = format_sale_price(sale_price) if sale_price else None
    size_str = ", ".join(sort_sizes(sizes))
    category = detect_category(title, description)
    gender = "unisex"

    fingerprint = compute_fingerprint(
        title=title,
        description=description,
        min_price=min_price,
        sale_price=sale_price,
        image_url=image_url,
        additional_images=additional_images,
        sizes=list(set(sizes)),
        variants_available=variants_available,
    )

    metadata = {
        "shopify_id": product_id,
        "handle": handle,
        "fingerprint": fingerprint,
        "total_images": len(images),
        "variants": [
            {"id": v["id"], "title": v["title"], "sku": v["sku"],
             "price": v["price"], "compare_at_price": v.get("compare_at_price"),
             "available": v["available"]}
            for v in variants
        ],
        "options": [{"name": o["name"], "values": o["values"]} for o in options],
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    return {
        "id": f"{SOURCE}_{product_id}",
        "source": SOURCE,
        "product_url": product_url,
        "image_url": image_url,
        "brand": BRAND,
        "title": title,
        "description": description,
        "category": category,
        "gender": gender,
        "size": size_str,
        "second_hand": SECOND_HAND,
        "price": price_str,
        "sale": sale_str,
        "additional_images": " , ".join(additional_images) if additional_images else None,
        "metadata": json.dumps(metadata),
        "image_embedding": None,
        "info_embedding": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def log_failed(product_url: str, reason: str):
    with open(FAILED_LOG, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {product_url} | {reason}\n")


# ─── Main Scraper ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Farmers Atelier Product Scraper")
    parser.add_argument("--max-products", type=int, default=None, help="Max products (default: all)")
    parser.add_argument("--no-skip", action="store_true", help="Re-process all even if unchanged")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    print("=" * 60)
    print("  Farmers Atelier Scraper")
    print("=" * 60)

    # ── Connect ──────────────────────────────────────────────────────────────
    print("\n[1/5] Connecting to Supabase...")
    db = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)
    print("  Connected")

    # ── Fetch API products ───────────────────────────────────────────────────
    print("\n[2/5] Fetching products from Shopify API...")
    api_products = fetch_all_products()
    print(f"  Found {len(api_products)} products")

    # ── Fetch existing DB products ───────────────────────────────────────────
    print("\n[3/5] Fetching existing products from database...")
    existing_list = db.fetch_existing()
    existing_by_url = {r["product_url"]: r for r in existing_list}
    print(f"  Found {len(existing_list)} existing records")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    model = EmbeddingModel()

    # ── Process ──────────────────────────────────────────────────────────────
    print("\n[5/5] Processing products...")
    print(f"  Batch size: {args.batch_size}")
    print()

    stats = {"new": 0, "updated": 0, "unchanged": 0, "skipped_no_image": 0, "failed": 0}

    failed_logged = []

    # Build fingerprint map from existing DB metadata
    existing_fingerprints = {}
    for r in existing_list:
        try:
            meta = json.loads(r.get("metadata") or "{}")
        except (json.JSONDecodeError, TypeError):
            meta = {}
        existing_fingerprints[r["product_url"]] = meta.get("fingerprint")

    now_ts = datetime.now(timezone.utc).isoformat()
    # Build full seen_urls from ALL API products regardless of --max-products
    all_seen_urls = {f"{BASE_URL}/products/{p['handle']}" for p in api_products}

    upsert_batch = []

    for idx, product in enumerate(api_products):
        if args.max_products and idx >= args.max_products:
            break

        handle = product.get("handle", "")
        product_url = f"{BASE_URL}/products/{handle}"
        title = product.get("title", "")
        image_url = (product.get("images") or [{}])[0].get("src") if product.get("images") else None

        existing = existing_by_url.get(product_url)
        existing_fp = existing_fingerprints.get(product_url)

        row = build_product_row(product, metadata_extra={"missed_count": 0, "last_seen_at": now_ts})
        if row is None:
            print(f"  [{idx+1}/{len(api_products)}] {title} — no image, skipping")
            stats["skipped_no_image"] += 1
            failed_logged.append((product_url, "no_image"))
            continue

        new_fp = json.loads(row["metadata"]).get("fingerprint")

        is_new = existing is None
        fingerprint_changed = existing_fp is not None and existing_fp != new_fp
        image_changed = existing is not None and image_url and existing.get("image_url") and existing["image_url"] != image_url
        needs_update = is_new or fingerprint_changed or image_changed or args.no_skip

        if not needs_update:
            stats["unchanged"] += 1
            continue

        # Compute embeddings only for new or image-changed products
        needs_embed = is_new or image_changed

        action = "NEW" if is_new else ("IMAGE CHANGED" if image_changed else "DATA CHANGED")
        print(f"  [{idx+1}/{len(api_products)}] {action}: {title}")

        if needs_embed:
            try:
                img = download_image(image_url)
                if img:
                    row["image_embedding"] = model.embed_image(img)

                text_for_embedding = build_text_for_embedding(
                    title=title,
                    description=row.get("description", ""),
                    category=row.get("category"),
                    gender=row.get("gender"),
                    price=row.get("price", ""),
                    sale=row.get("sale"),
                    sizes=row.get("size", "").split(", ") if row.get("size") else [],
                )
                row["info_embedding"] = model.embed_text(text_for_embedding)
                time.sleep(STAGGER_DELAY_S)
            except Exception as e:
                print(f"    [ERROR] Embedding failed: {e}")
                stats["failed"] += 1
                failed_logged.append((product_url, f"embedding_error: {e}"))
                log_failed(product_url, f"embedding_error: {e}")
                continue
        else:
            # Data changed but image is same — carry over existing embeddings
            row["image_embedding"] = existing.get("image_embedding")
            row["info_embedding"] = existing.get("info_embedding")
            row["created_at"] = existing.get("created_at", now_ts)

        if is_new:
            stats["new"] += 1
        else:
            stats["updated"] += 1

        upsert_batch.append(row)

    # ── Batch upsert ─────────────────────────────────────────────────────────
    print(f"\n  Upserting {len(upsert_batch)} products in batches of {args.batch_size}...")

    upserted_count = 0
    for i in range(0, len(upsert_batch), args.batch_size):
        batch = upsert_batch[i:i + args.batch_size]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                db.batch_upsert(batch)
                upserted_count += len(batch)
                print(f"    Batch {i//args.batch_size + 1}: {len(batch)} products upserted")
                break
            except Exception as e:
                print(f"    Batch {i//args.batch_size + 1} failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
                else:
                    for r in batch:
                        failed_logged.append((r["product_url"], "batch_upsert_failed"))
                        log_failed(r["product_url"], "batch_upsert_failed after 3 retries")

    # ── Update metadata for unchanged seen products ──────────────────────────
    # These weren't upserted, but need missed_count reset for stale tracking
    upserted_urls = {r["product_url"] for r in upsert_batch}
    unchanged_seen = all_seen_urls - upserted_urls
    if unchanged_seen:
        print(f"\n  Updating staleness tracking for {len(unchanged_seen)} unchanged products...")
    for url in unchanged_seen:
        r = existing_by_url.get(url)
        if not r:
            continue
        try:
            meta = json.loads(r.get("metadata") or "{}")
        except (json.JSONDecodeError, TypeError):
            meta = {}
        meta["missed_count"] = 0
        meta["last_seen_at"] = now_ts
        db.update_metadata(url, meta)

    # ── Handle stale products ────────────────────────────────────────────────
    deleted = 0
    # Safety: only run stale cleanup if API returned a substantial result
    if len(api_products) < 10:
        print(f"\n  Skipping stale check (only {len(api_products)} products from API — likely a partial fetch)")
    else:
        print(f"\n  Checking for stale products...")
        to_delete = []
        to_increment = []

        for r in existing_list:
            url = r["product_url"]
            if url in all_seen_urls:
                continue
            try:
                meta = json.loads(r.get("metadata") or "{}")
            except (json.JSONDecodeError, TypeError):
                meta = {}
            missed = meta.get("missed_count", 0)
            if missed >= STALE_MISS_THRESHOLD - 1:
                to_delete.append(url)
            else:
                to_increment.append(url)

        deleted = db.delete_products(to_delete)
        for url in to_increment:
            r = existing_by_url.get(url)
            if not r:
                continue
            try:
                meta = json.loads(r.get("metadata") or "{}")
            except (json.JSONDecodeError, TypeError):
                meta = {}
            meta["missed_count"] = meta.get("missed_count", 0) + 1
            meta["last_seen_at"] = now_ts
            db.update_metadata(url, meta)

    stats["deleted"] = deleted

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  RUN SUMMARY")
    print("=" * 60)
    print(f"  Total products from API:       {len(api_products)}")
    print(f"  New products added:            {stats['new']}")
    print(f"  Products updated:              {stats['updated']}")
    print(f"  Products unchanged (skipped):  {stats['unchanged']}")
    print(f"  Products deleted (stale):      {stats['deleted']}")
    print(f"  Skipped (no image):            {stats['skipped_no_image']}")
    print(f"  Failed:                        {stats['failed']}")
    print("=" * 60)

    if failed_logged:
        print(f"\n  {len(failed_logged)} failures logged to {FAILED_LOG}")

    if stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
