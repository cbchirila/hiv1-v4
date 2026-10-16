#!/usr/bin/env python3
"""Download all files from a SureChEMBL bulk-data directory.

Requires tqdm (pip install tqdm). Supports retries and HTTP Range resume.
"""

from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen
import re
import sys
import time

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("Missing dependency: tqdm. Install it using: python -m pip install tqdm")

SOURCE_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBL/bulk_data/2026-10-06/"
DESTINATION = Path(__file__).resolve().parent / "002 - surechembl prefetch"
CHUNK_SIZE = 8 * 1024 * 1024
MAX_RETRIES = 5
TIMEOUT = 120
USER_AGENT = "SureChEMBL-prefetch/1.0"


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def list_files():
    request = Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=TIMEOUT) as response:
        html = response.read().decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(html)
    base = urlparse(SOURCE_URL)
    files = {}
    for href in parser.links:
        url = urljoin(SOURCE_URL, href)
        parsed = urlparse(url)
        if (parsed.scheme != base.scheme or parsed.netloc != base.netloc
                or not parsed.path.startswith(base.path) or parsed.path.endswith("/")):
            continue
        relative = unquote(parsed.path[len(base.path):])
        # Only files immediately within the listed directory; no traversal.
        if not relative or "/" in relative or relative in (".", ".."):
            continue
        files[relative] = url
    return sorted(files.items())


def format_size(size):
    if size is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024


def remote_size(url):
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
        with urlopen(req, timeout=TIMEOUT) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None


def download(name, url):
    destination = DESTINATION / name
    partial = DESTINATION / (name + ".part")
    expected = remote_size(url)

    if destination.exists() and expected is not None and destination.stat().st_size == expected:
        print(f"[SKIP] {name} (already complete)", flush=True)
        return True
    if destination.exists():
        # Treat files with unknown/wrong size as incomplete, without deleting data.
        if not partial.exists():
            destination.rename(partial)
        else:
            destination.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if expected is not None and offset == expected:
                partial.replace(destination)
                print(f"[OK] {name} ({format_size(expected)})", flush=True)
                return True
            if expected is not None and offset > expected:
                partial.unlink()
                offset = 0

            headers = {"User-Agent": USER_AGENT}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            req = Request(url, headers=headers)
            with urlopen(req, timeout=TIMEOUT) as response:
                status = response.status
                if offset and status != 206:
                    # Server ignored Range. Restart safely instead of appending duplicate bytes.
                    offset = 0
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
                    if not match or int(match.group(1)) != offset:
                        raise IOError("Invalid HTTP Range response")
                    if match.group(3) != "*":
                        expected = int(match.group(3))
                elif expected is None:
                    length = response.headers.get("Content-Length")
                    expected = int(length) if length else None

                mode = "ab" if offset else "wb"
                downloaded = offset
                with partial.open(mode) as output, tqdm(
                    total=expected,
                    initial=offset,
                    desc=name,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    dynamic_ncols=True,
                    leave=True,
                ) as progress:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        progress.update(len(chunk))

            if expected is not None and downloaded != expected:
                raise IOError(f"Incomplete file: {downloaded} of {expected} bytes")
            partial.replace(destination)
            print(f"[OK] {name}", flush=True)
            return True
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            print(f"\n[ERROR] {name}, attempt {attempt}/{MAX_RETRIES}: {exc}", flush=True)
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 30))
    return False


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    print(f"Source: {SOURCE_URL}\nDestination: {DESTINATION}\n", flush=True)
    try:
        files = list_files()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"Could not retrieve the file list: {exc}", file=sys.stderr)
        return 1
    if not files:
        print("No files were found.", file=sys.stderr)
        return 1
    print(f"Found {len(files)} files.\n", flush=True)
    failed = [name for name, url in files if not download(name, url)]
    if failed:
        print("\nFailed downloads: " + ", ".join(failed), file=sys.stderr)
        return 1
    print("\nAll files have been downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
