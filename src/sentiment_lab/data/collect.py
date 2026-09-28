"""Crawl the publisher's HTML, discover its archive, and stream the reviews.

This is a public dataset acquisition crawler, not a live IMDb review crawler.
The archive is read in place: no untrusted tar paths are extracted to disk.
"""
import hashlib
import logging
import re
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import Paths
from ..io import read_json, write_json

SOURCE = "https://ai.stanford.edu/~amaas/data/sentiment/"
AGENT = "CourseSentimentLab/1.0 (educational public dataset acquisition)"
LOGGER = logging.getLogger(__name__)
REVIEW_PATH = re.compile(r"aclImdb/(train|test)/(pos|neg)/(\d+)_(\d+)\.txt$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def session() -> requests.Session:
    client = requests.Session()
    client.headers["User-Agent"] = AGENT
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    client.mount("https://", HTTPAdapter(max_retries=retries))
    return client


def check_robots(client: requests.Session, url: str) -> dict:
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    response = client.get(robots_url, timeout=(15, 60))
    if response.status_code == 404:
        return {"url": robots_url, "status": 404, "allowed": True}
    response.raise_for_status()
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    if not parser.can_fetch("CourseSentimentLab", url):
        raise PermissionError(f"robots.txt disallows collection: {url}")
    return {"url": robots_url, "status": response.status_code, "allowed": True}


def discover_archive(page: str, base_url: str = SOURCE) -> str:
    """Extract the real download link instead of assuming a hardcoded path."""
    soup = BeautifulSoup(page, "html.parser")
    for link in soup.select("a[href]"):
        url = urljoin(base_url, link["href"])
        if urlparse(url).path.endswith("/aclImdb_v1.tar.gz"):
            if urlparse(url).hostname != urlparse(base_url).hostname:
                raise ValueError("Unexpected off-site archive URL")
            return url
    raise ValueError("Publisher page has no aclImdb_v1.tar.gz link")


def download(client: requests.Session, url: str, target: Path) -> None:
    """Download bounded HTTP byte ranges; retry failed bodies, not just headers."""
    temporary = target.with_suffix(".part")
    metadata = client.head(url, timeout=(15, 60))
    metadata.raise_for_status()
    total = int(metadata.headers["Content-Length"])
    chunk_size = 8 * 1024 * 1024
    position = 0
    etag = metadata.headers.get("ETag")
    try:
        with temporary.open("wb") as stream:
            while position < total:
                end = min(position + chunk_size, total) - 1
                headers = {"Range": f"bytes={position}-{end}"}
                if etag:
                    headers["If-Range"] = etag
                for attempt in range(3):
                    try:
                        stream.seek(position)
                        stream.truncate()
                        with client.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
                            response.raise_for_status()
                            if response.status_code == 206:
                                expected_range = f"bytes {position}-{end}/{total}"
                                if response.headers.get("Content-Range") != expected_range:
                                    raise ValueError("Unexpected Content-Range; archive may have changed")
                                expected = end - position + 1
                            elif response.status_code == 200 and position == 0:
                                expected = total  # Server may ignore Range on the first request.
                            else:
                                raise ValueError("Server stopped honoring byte ranges")
                            received = 0
                            for block in response.iter_content(1024 * 1024):
                                stream.write(block)
                                received += len(block)
                            if received != expected:
                                raise IOError("Incomplete archive segment")
                        position += received
                        LOGGER.info("Downloaded %.0f / %.0f MiB", position / 1024**2, total / 1024**2)
                        break
                    except (requests.RequestException, IOError):
                        if attempt == 2:
                            raise
                        LOGGER.warning("Retrying segment at byte %s", position)
                        time.sleep(2 ** attempt)
                time.sleep(0.2)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def read_reviews(archive: Path) -> pd.DataFrame:
    rows = []
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            match = REVIEW_PATH.fullmatch(member.name)
            if not match or not member.isfile():
                continue
            split, sentiment, review_id, rating = match.groups()
            if member.size > 2 * 1024 * 1024:
                raise ValueError(f"Review unexpectedly large: {member.name}")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable review: {member.name}")
            rows.append({"id": f"{split}/{sentiment}/{review_id}", "split": split,
                         "label": int(sentiment == "pos"), "rating": int(rating),
                         "text": stream.read().decode("utf-8", errors="strict")})
    frame = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    counts = frame.groupby(["split", "label"]).size().to_dict()
    if len(frame) != 50000 or set(counts.values()) != {12500} or len(counts) != 4:
        raise ValueError(f"Unexpected original dataset size/balance: {counts}")
    return frame


def collect(paths: Paths, offline: bool = False) -> pd.DataFrame:
    paths.raw.mkdir(parents=True, exist_ok=True)
    archive = paths.raw / "aclImdb_v1.tar.gz"
    provenance_path = paths.raw / "provenance.json"
    if archive.exists() and provenance_path.exists():
        provenance = read_json(provenance_path)
        if sha256(archive) != provenance["sha256"]:
            raise ValueError("Cached archive checksum mismatch; remove it and reacquire")
        LOGGER.info("Verified cached archive; network access skipped")
    else:
        if offline:
            raise FileNotFoundError("Offline mode needs the archive and provenance.json; run collect first")
        with session() as client:
            robots = check_robots(client, SOURCE)
            response = client.get(SOURCE, timeout=(15, 60))
            response.raise_for_status()
            (paths.raw / "source_page.html").write_text(response.text, encoding="utf-8")
            url = discover_archive(response.text)
            archive_robots = check_robots(client, url)
            time.sleep(1)  # Be polite: this crawler needs only a handful of requests.
            download(client, url, archive)
        provenance = {"source_page": SOURCE, "archive_url": url,
                      "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                      "sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
                      "robots": [robots, archive_robots], "user_agent": AGENT,
                      "acquisition": "publisher HTML discovery + public benchmark archive",
                      "original_labeled_reviews": 50000, "excluded_unlabeled_reviews": 50000}
        write_json(provenance_path, provenance)
    reviews = read_reviews(archive)
    LOGGER.info("Read %s labeled reviews", len(reviews))
    return reviews
