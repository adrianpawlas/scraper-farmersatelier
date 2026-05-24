#!/usr/bin/env python3
"""
Farmers Atelier Complete Scraper
Scrapes all products from farmersatelier.com via the Shopify products.json API,
computes image & text embeddings using google/siglip-base-patch16-384 (768-dim),
and uploads everything to Supabase.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from typing import Optional

import requests
import torch
from bs4 import BeautifulSoup
from PIL import Image
from supabase import create_client, Client
from transformers import AutoProcessor, AutoModel

# ─── Configuration ───────────────────────────────────────────────────────────
# Credentials from environment variables (with fallback for local use)

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

def detect_category(title: str, description: str) -> Optional[str]:
    title_lower = title.lower()
    desc_lower = (description or "").lower()
    found = []
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, title_lower) or re.search(pattern, desc_lower):
                found.append(cat)
                break
    if not found:
        return None
    return ", ".join(found)


def clean_html(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def format_price(price_eur: float) -> str:
    prices = []
    prices.append(f"{price_eur:.2f}EUR")
    prices.append(f"{price_eur * EUR_TO_USD:.2f}USD")
    return ", ".join(prices)


def format_sale_price(price_eur: float) -> str:
    return f"{price_eur:.2f}EUR"


def build_text_for_embedding(title: str, description: str, category: Optional[str],
                              gender: Optional[str], price: str, sale: Optional[str],
                              sizes: list) -> str:
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


# ─── Embedding Model ─────────────────────────────────────────────────────────

class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        print(f"[INFO] Loading embedding model {model_name} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        print(f"[INFO] Model loaded. Embedding dim: {EMBEDDING_DIM}")

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> list[float]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        embedding = outputs.pooler_output.cpu().numpy().flatten().tolist()
        return embedding

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        inputs = self.processor(
            text=text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        embedding = outputs.pooler_output.cpu().numpy().flatten().tolist()
        return embedding


# ─── Image Downloader ────────────────────────────────────────────────────────

def download_image(url: str, max_retries: int = 3) -> Optional[Image.Image]:
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return img
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"[WARN] Failed to download image {url}: {e}")
    return None


# ─── Shopify API Fetcher ─────────────────────────────────────────────────────

def fetch_all_products() -> list[dict]:
    all_products = []
    page = 1
    limit = 250

    while True:
        url = f"{BASE_URL}/products.json?limit={limit}&page={page}"
        print(f"[INFO] Fetching products page {page}...")
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

    print(f"[INFO] Total products fetched: {len(all_products)}")
    return all_products


# ─── Supabase Uploader ───────────────────────────────────────────────────────

class SupabaseUploader:
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)
        self.table = "products"

    def upsert_product(self, product_data: dict) -> bool:
        try:
            source = product_data["source"]
            product_url = product_data["product_url"]
            data = self.client.table(self.table).upsert(
                product_data,
                on_conflict="source,product_url"
            ).execute()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to upsert {product_data.get('title', '?')}: {e}")
            return False

    def product_exists(self, source: str, product_url: str) -> bool:
        try:
            result = self.client.table(self.table).select("id").eq("source", source).eq("product_url", product_url).execute()
            return len(result.data) > 0
        except Exception:
            return False


# ─── Main Scraper ────────────────────────────────────────────────────────────

def scrape_products(embedding_model: EmbeddingModel, uploader: SupabaseUploader,
                    skip_existing: bool = True, max_products: Optional[int] = None):
    products = fetch_all_products()

    if max_products:
        products = products[:max_products]

    stats = {"total": len(products), "success": 0, "skipped": 0, "failed": 0}

    for idx, product in enumerate(products):
        title = product.get("title", "")
        handle = product.get("handle", "")
        product_id = product.get("id")
        vendor = product.get("vendor", "")
        body_html = product.get("body_html", "")
        description = clean_html(body_html)
        images = product.get("images", [])
        variants = product.get("variants", [])
        options = product.get("options", [])

        product_url = f"{BASE_URL}/products/{handle}"
        image_url = images[0]["src"] if images else None
        additional_images = [img["src"] for img in images[1:]] if len(images) > 1 else []

        if not image_url:
            print(f"[SKIP] {title} - no images")
            stats["skipped"] += 1
            continue

        if skip_existing and uploader.product_exists(SOURCE, product_url):
            print(f"[SKIP] {title} - already exists")
            stats["skipped"] += 1
            continue

        # Extract sizes and prices
        sizes = []
        min_price = None
        max_price = None
        sale_price = None

        for v in variants:
            price = float(v["price"])
            sizes.append(v["title"])
            if min_price is None or price < min_price:
                min_price = price
            if max_price is None or price > max_price:
                max_price = price
            if v.get("compare_at_price") and float(v["compare_at_price"]) > price:
                if sale_price is None or price < sale_price:
                    sale_price = price

        if min_price is None:
            print(f"[SKIP] {title} - no price")
            stats["skipped"] += 1
            continue

        price_str = format_price(min_price)
        sale_str = format_sale_price(sale_price) if sale_price else None
        size_str = ", ".join(sorted(set(sizes), key=lambda x: ["S", "M", "L", "XL", "2XL", "XS", "3XL"].index(x) if x in ["S", "M", "L", "XL", "2XL", "XS", "3XL"] else 99))

        # Detect category; all products are unisex for this brand
        category = detect_category(title, description)
        gender = "unisex"

        # Build metadata
        metadata = {
            "shopify_id": product_id,
            "handle": handle,
            "vendor": vendor,
            "variants": [
                {
                    "id": v["id"],
                    "title": v["title"],
                    "sku": v["sku"],
                    "price": v["price"],
                    "compare_at_price": v.get("compare_at_price"),
                    "available": v["available"],
                }
                for v in variants
            ],
            "options": [{"name": o["name"], "values": o["values"]} for o in options],
            "total_images": len(images),
        }

        # Build text for info embedding (SIGLIP max 64 tokens, keep it concise)
        text_for_embedding = build_text_for_embedding(
            title=title,
            description=description,
            category=category,
            gender=gender,
            price=price_str,
            sale=sale_str,
            sizes=list(set(sizes)),
        )

        # Compute embeddings
        print(f"[{idx+1}/{stats['total']}] Processing: {title}")
        image_embedding = None
        info_embedding = None

        try:
            img = download_image(image_url)
            if img:
                image_embedding = embedding_model.embed_image(img)
                print(f"  Image embedding computed ({len(image_embedding)} dim)")
            else:
                print("  [WARN] Could not download image for embedding")

            info_embedding = embedding_model.embed_text(text_for_embedding)
            print(f"  Info embedding computed ({len(info_embedding)} dim)")
        except Exception as e:
            print(f"  [ERROR] Embedding failed: {e}")
            stats["failed"] += 1
            continue

        # Prepare row for Supabase (vectors as JSON arrays for pgvector/PostgREST)
        row = {
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
            "image_embedding": image_embedding,
            "info_embedding": info_embedding,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # PostgREST requires pgvector values as JSON-compatible lists (not strings)
        # supabase-py should handle this, but ensure they're plain lists
        if image_embedding is not None and not isinstance(image_embedding, list):
            row["image_embedding"] = json.loads(image_embedding) if isinstance(image_embedding, str) else list(image_embedding)
        if info_embedding is not None and not isinstance(info_embedding, list):
            row["info_embedding"] = json.loads(info_embedding) if isinstance(info_embedding, str) else list(info_embedding)

        # Upload to Supabase
        if uploader.upsert_product(row):
            stats["success"] += 1
            print(f"  ✓ Uploaded to Supabase")
        else:
            stats["failed"] += 1

        # Small delay to avoid rate limiting
        time.sleep(1)

    return stats


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Farmers Atelier Product Scraper")
    parser.add_argument(
        "--max-products",
        type=int,
        default=None,
        help="Maximum number of products to process (default: all)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-process all products even if they already exist",
    )
    parser.add_argument(
        "--force-embed",
        action="store_true",
        help="Recompute embeddings and update existing records",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Farmers Atelier Complete Scraper")
    print("=" * 60)

    # Initialize Supabase
    print("\n[1/4] Connecting to Supabase...")
    uploader = SupabaseUploader(SUPABASE_URL, SUPABASE_KEY)
    print("[OK] Connected to Supabase")

    # Initialize embedding model
    print(f"\n[2/4] Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    embedding_model = EmbeddingModel()
    print("[OK] Model loaded")

    # Run scraper
    print("\n[3/4] Fetching products and computing embeddings...")
    stats = scrape_products(
        embedding_model=embedding_model,
        uploader=uploader,
        skip_existing=not args.no_skip,
        max_products=args.max_products,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"  Total products found: {stats['total']}")
    print(f"  Successfully uploaded: {stats['success']}")
    print(f"  Skipped (already exist): {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    print("=" * 60)

    if stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
