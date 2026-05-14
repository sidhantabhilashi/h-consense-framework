"""
download_movielens.py
---------------------
Downloads and extracts the MovieLens 100K dataset into data/raw/.

Usage:
    python 01_data/download_movielens.py

Outputs:
    data/raw/ml-100k/u.data      <- ratings (user, item, rating, timestamp)
    data/raw/ml-100k/u.item      <- movie metadata
    data/raw/ml-100k/u.user      <- user metadata

Safe to re-run: skips download if files already exist.
"""

# !! MUST be first — fixes macOS OpenMP crash before any library loads.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import zipfile
import requests

# Allow importing config from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def log(msg):
    """Simple timestamped console logger."""
    print(f"[download] {msg}", flush=True)


def download_file(url, dest_path):
    """
    Downloads a file from url to dest_path with a progress indicator.
    Raises an exception with a clear message if download fails.
    """
    log(f"Downloading from: {url}")
    log(f"Saving to:        {dest_path}")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()  # raises HTTPError for bad status codes
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            "[ERROR] Cannot connect to the internet. "
            "Check your network connection and try again."
        )
    except requests.exceptions.HTTPError as e:
        raise SystemExit(f"[ERROR] HTTP error while downloading: {e}")
    except requests.exceptions.Timeout:
        raise SystemExit("[ERROR] Download timed out after 60s. Try again.")

    total_bytes = int(response.headers.get("content-length", 0))
    downloaded  = 0

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_bytes > 0:
                pct = downloaded / total_bytes * 100
                print(f"\r[download] Progress: {pct:.1f}%  ", end="", flush=True)
    print()  # newline after progress
    log(f"Download complete. File size: {downloaded / 1024:.1f} KB")


def extract_zip(zip_path, extract_to):
    """
    Extracts a zip file to extract_to directory.
    Raises a clear error if the zip is corrupted.
    """
    log(f"Extracting {zip_path} -> {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # List contents before extracting so user can see what's inside
            names = zf.namelist()
            log(f"Zip contains {len(names)} files. Extracting...")
            zf.extractall(extract_to)
    except zipfile.BadZipFile:
        raise SystemExit(
            f"[ERROR] The downloaded file is not a valid zip: {zip_path}\n"
            "       Delete it and run this script again."
        )
    log("Extraction complete.")


def verify_required_files(ml_dir):
    """
    Checks that the key files we need actually exist after extraction.
    Prints a clear error listing what is missing.
    """
    required = ["u.data", "u.item", "u.user"]
    missing  = [f for f in required if not os.path.isfile(os.path.join(ml_dir, f))]

    if missing:
        raise SystemExit(
            f"[ERROR] Extraction succeeded but required files are missing: {missing}\n"
            f"       Expected them inside: {ml_dir}\n"
            f"       Contents found: {os.listdir(ml_dir)}"
        )

    log("Verified required files exist:")
    for f in required:
        full = os.path.join(ml_dir, f)
        size = os.path.getsize(full)
        log(f"  {f:15s}  {size / 1024:.1f} KB")


def main():
    log("=" * 55)
    log("MovieLens 100K — Download Script")
    log("=" * 55)

    # ── Step 1: Create directories ──────────────────────────────────
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    log(f"Raw data dir: {config.RAW_DATA_DIR}")

    zip_path = os.path.join(config.RAW_DATA_DIR, config.MOVIELENS_ZIP)
    ml_dir   = os.path.join(config.RAW_DATA_DIR, config.MOVIELENS_DIR)

    # ── Step 2: Check if already downloaded ──────────────────────────
    udata_path = os.path.join(ml_dir, config.RATINGS_FILE)
    if os.path.isfile(udata_path):
        log(f"Dataset already exists at: {udata_path}")
        log("Skipping download. Delete data/raw/ml-100k/ to re-download.")
        verify_required_files(ml_dir)
        log("All good! Ready for preprocessing.")
        return

    # ── Step 3: Download ─────────────────────────────────────────────
    download_file(config.MOVIELENS_URL, zip_path)

    # ── Step 4: Extract ──────────────────────────────────────────────
    extract_zip(zip_path, config.RAW_DATA_DIR)

    # ── Step 5: Verify ──────────────────────────────────────────────
    verify_required_files(ml_dir)

    # ── Step 6: Cleanup zip (optional, saves ~5MB) ──────────────────────
    os.remove(zip_path)
    log(f"Removed zip file: {zip_path}")

    log("=" * 55)
    log("SUCCESS: Dataset ready. Next step:")
    log("  python 01_data/preprocess.py")
    log("=" * 55)


if __name__ == "__main__":
    main()
