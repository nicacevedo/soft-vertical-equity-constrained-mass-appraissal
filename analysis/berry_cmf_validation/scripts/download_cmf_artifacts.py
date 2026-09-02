#!/usr/bin/env python3
"""Download CMF/Berry local-replication artifacts; never alter raw bytes.

Writes analysis/berry_cmf_validation/file_manifest.csv.
Stores files under data/berry_cmf/raw/<jurisdiction>/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[3]
ANALYSIS = REPO / "analysis" / "berry_cmf_validation"
RAW_ROOT = REPO / "data" / "berry_cmf" / "raw"
LOG_DIR = ANALYSIS / "logs"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
COOKIEJAR = LOG_DIR / "box_cookies.txt"
ITEM_RE = re.compile(
    r'\{"typedID":"(?P<tid>[^"]+)","type":"(?P<typ>file|folder)","id":(?P<id>\d+),'
    r'.*?"name":"(?P<name>[^"]+)","itemSize":(?P<sz>\d+)',
    re.S,
)

# Primary jurisdictions: download all listed Box shares and CMF site files.
# Secondary: code/docs only unless file is small.
PRIMARY_BOX = [
    # Detroit
    {"jurisdiction": "detroit_mi", "shared": "eva6qqtubtlhhiei0old3vy2yoyy1kx5",
     "page": "https://propertytaxproject.uchicago.edu/detroit-raw-code/",
     "gov": "City of Detroit Office of the Assessor", "designation": "report",
     "priority": "primary"},
    {"jurisdiction": "detroit_mi", "shared": "q3mi0r3xcm8u4wncp0e842qi9q6grgyd",
     "page": "https://propertytaxproject.uchicago.edu/detroit-raw-code/",
     "gov": "n/a", "designation": "replication_code", "priority": "primary"},
    {"jurisdiction": "detroit_mi", "shared": "qzz9nz9l81m1vku1q6luqzmxvdw9q9wb",
     "page": "https://propertytaxproject.uchicago.edu/detroit-raw-code/",
     "gov": "City of Detroit Office of the Assessor",
     "designation": "replication_data", "priority": "primary"},
    # Philadelphia
    {"jurisdiction": "philadelphia_pa", "shared": "320haoiyghjreigksljv1c4xw7u6lhnb",
     "page": "https://propertytaxproject.uchicago.edu/philadelphia-raw-data-code/",
     "gov": "City of Philadelphia Office of Property Assessment (inferred)",
     "designation": "raw_public_dataset", "priority": "primary"},
    {"jurisdiction": "philadelphia_pa", "shared": "rom2t66ys8pxs5afqpd6xat8rayoc2sq",
     "page": "https://propertytaxproject.uchicago.edu/philadelphia-code/",
     "gov": "City of Philadelphia Office of Property Assessment (inferred)",
     "designation": "cleaned_public_dataset", "priority": "primary"},
    # Orleans
    {"jurisdiction": "orleans_la", "shared": "hz5rv02dpgw61e0qp3kvbyz0je1omi6v",
     "page": "https://propertytaxproject.uchicago.edu/orleans-parish-raw-code/",
     "gov": "Orleans Parish Assessor (public record request in folder)",
     "designation": "replication_materials", "priority": "primary"},
    {"jurisdiction": "orleans_la", "shared": "k69tibdzp6u0lbbq20w6m1ugtmn6z7lr",
     "page": "https://propertytaxproject.uchicago.edu/orleans-parish-raw-code/",
     "gov": "n/a", "designation": "replication_code", "priority": "primary"},
    {"jurisdiction": "orleans_la", "shared": "7wg581vmzragpao2y9raemscmthbnwn5",
     "page": "https://propertytaxproject.uchicago.edu/orleans-parish-raw-code/",
     "gov": "n/a", "designation": "replication_code", "priority": "primary"},
    # Franklin / Columbus
    {"jurisdiction": "franklin_oh", "shared": "2jn1707wbpxdd98m1lke6igkqmdvg5t3",
     "page": "https://propertytaxproject.uchicago.edu/columnbs-raw-data-code/",
     "gov": "Franklin County Auditor", "designation": "raw_public_dataset",
     "priority": "primary"},
    # St. Louis
    {"jurisdiction": "st_louis_county_mo", "shared": "rabph6sd546szpwe6likep763hv8g3v3",
     "page": "https://propertytaxproject.uchicago.edu/stlouis-county-raw-data-code/",
     "gov": "St. Louis County Assessor", "designation": "raw_public_dataset",
     "priority": "primary"},
    {"jurisdiction": "st_louis_county_mo", "shared": "3ciiptmn8gcpe9ohwb7zj2fn46qc0qez",
     "page": "https://propertytaxproject.uchicago.edu/stlouis-county-raw-data-code/",
     "gov": "n/a", "designation": "report", "priority": "primary"},
    # Cook
    {"jurisdiction": "cook_il", "shared": "5j9offt7kv763i62duvhvi6hk5rrh1ok",
     "page": "https://propertytaxproject.uchicago.edu/cook-county-raw-data-code/",
     "gov": "Cook County Board of Review / CCAO residential sales extracts",
     "designation": "raw_public_dataset", "priority": "primary"},
]

# Secondary: skip files larger than this unless they are code/docs.
SECONDARY_MAX_BYTES = 25 * 1024 * 1024
SECONDARY_BOX = [
    {"jurisdiction": "nyc_ny", "shared": "5z3sociaxaeidf9pl05tsfshwnca5p6z",
     "page": "https://propertytaxproject.uchicago.edu/ny-raw-code/",
     "gov": "NYC Department of Finance (confirm in README)",
     "designation": "replication_materials", "priority": "secondary_inventory_only"},
    {"jurisdiction": "nyc_ny", "shared": "ve3x97o09r5ri7ch18rfgw9f8v6ip1bw",
     "page": "https://propertytaxproject.uchicago.edu/ny-raw-code/",
     "gov": "n/a", "designation": "replication_code", "priority": "secondary_inventory_only"},
    {"jurisdiction": "nyc_ny", "shared": "5aes0b89wj68adjluej6lywyts35shia",
     "page": "https://propertytaxproject.uchicago.edu/ny-raw-code/",
     "gov": "n/a", "designation": "replication_code", "priority": "secondary_inventory_only"},
    {"jurisdiction": "erie_ny", "shared": "5n4x5sx0t36ycvsjd1rrhxd77rzokuoj",
     "page": "https://propertytaxproject.uchicago.edu/buffalo-raw-data-code/",
     "gov": "City of Buffalo assessment/sales extracts",
     "designation": "raw_public_dataset", "priority": "secondary_inventory_only"},
    {"jurisdiction": "erie_ny", "shared": "zbc3ol4mlfbcuvy7y5y18o2rr35u4yjc",
     "page": "https://propertytaxproject.uchicago.edu/buffalo-raw-data-code/",
     "gov": "n/a", "designation": "report", "priority": "secondary_inventory_only"},
    {"jurisdiction": "clark_nv", "shared": "es7id24z9nad6hutfve38ppkfkogneni",
     "page": "https://propertytaxproject.uchicago.edu/las-vegas-raw-data-code/",
     "gov": "Clark County Assessor (inferred)",
     "designation": "raw_public_dataset", "priority": "secondary_inventory_only"},
    {"jurisdiction": "los_angeles_ca", "shared": "s5gz5t9aus1fgko4qtvo8qk0xlba6m8n",
     "page": "https://propertytaxproject.uchicago.edu/los-angeles-raw-data-code/",
     "gov": "Los Angeles County Assessor",
     "designation": "raw_public_dataset", "priority": "secondary_inventory_only"},
    {"jurisdiction": "maricopa_az", "shared": "5cg6rza2ekwo9fdy7zgxki4cku4ak21q",
     "page": "https://propertytaxproject.uchicago.edu/maricopa-county-raw-data-code/",
     "gov": "Maricopa County Assessor (PHX-named files)",
     "designation": "raw_public_dataset", "priority": "secondary_inventory_only"},
]

DIRECT_HTTP = [
    {"jurisdiction": "detroit_mi",
     "url": "https://propertytaxproject.uchicago.edu/files/2020/03/Prop-Tax-Detroit-Final-3220.pdf",
     "page": "https://propertytaxproject.uchicago.edu/papers/",
     "gov": "City of Detroit Office of the Assessor", "designation": "local_report",
     "priority": "primary", "relpath": "detroit_mi/cmf_site/Prop-Tax-Detroit-Final-3220.pdf"},
    {"jurisdiction": "detroit_mi",
     "url": "https://propertytaxproject.uchicago.edu/files/2020/03/Detroit-Parcel-Data-from-Assessor.zip",
     "page": "https://propertytaxproject.uchicago.edu/papers/",
     "gov": "City of Detroit Office of the Assessor",
     "designation": "raw_assessor_parcel_data", "priority": "primary",
     "relpath": "detroit_mi/cmf_site/Detroit-Parcel-Data-from-Assessor.zip"},
]

CODE_ONLY_EXT = {".r", ".rmd", ".txt", ".md", ".pdf", ".html", ".xlsx", ".xls"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, buf: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def curl(args: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    cmd = ["curl", "-L", "--retry", "5", "--retry-delay", "3", "--connect-timeout", "30",
           "--max-time", str(timeout), "-A", UA, "-c", str(COOKIEJAR), "-b", str(COOKIEJAR)] + args
    return subprocess.run(cmd, capture_output=True)


def curl_download(url: str, dest: Path, timeout: int = 0) -> Tuple[int, str, str]:
    """Resume-safe download. Returns (http_code, final_url, stderr)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # If a complete dest exists, do not rewrite.
    extra = ["-C", "-", "-o", str(tmp), "-w", "%{http_code} %{url_effective}", "-D", str(tmp) + ".hdr"]
    if timeout:
        # override default max-time for large files: 0 means no curl --max-time in our helper
        pass
    max_time = ["--max-time", str(timeout)] if timeout else ["--max-time", "0"]
    cmd = ["curl", "-L", "--retry", "8", "--retry-delay", "4", "--connect-timeout", "30",
           *max_time, "-A", UA, "-c", str(COOKIEJAR), "-b", str(COOKIEJAR), *extra, url]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    parts = out.split(" ", 1)
    code = int(parts[0]) if parts and parts[0].isdigit() else proc.returncode
    final = parts[1] if len(parts) > 1 else url
    if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(dest)
        hdr = Path(str(tmp) + ".hdr")
        if hdr.exists():
            hdr.replace(dest.with_suffix(dest.suffix + ".headers"))
    return code, final, proc.stderr[-2000:] if proc.stderr else ""


def parse_box_items(html: str) -> List[dict]:
    seen = set()
    items = []
    for m in ITEM_RE.finditer(html):
        key = (m["id"], m["name"])
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "typed_id": m["tid"],
            "type": m["typ"],
            "id": m["id"],
            "name": m["name"],
            "size": int(m["sz"]),
        })
    return items


def box_get_html(url: str) -> str:
    dest = LOG_DIR / "box_pages" / ("live_" + hashlib.sha256(url.encode()).hexdigest()[:16] + ".html")
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = curl(["-o", str(dest), url], timeout=60)
    if dest.exists():
        return dest.read_text(errors="replace")
    raise RuntimeError(f"failed to fetch {url}: {proc.stderr[:400]!r}")


def box_list_recursive(shared: str, folder_id: Optional[str], rel: Path) -> List[dict]:
    if folder_id:
        url = f"https://uchicago.app.box.com/s/{shared}/folder/{folder_id}"
    else:
        url = f"https://uchicago.app.box.com/s/{shared}"
    html = box_get_html(url)
    items = parse_box_items(html)
    out = []
    for it in items:
        safe = it["name"].replace("/", "_").replace("\\", "_")
        child_rel = rel / safe
        rec = dict(it, relpath=str(child_rel), shared=shared, parent_folder_id=folder_id or "")
        out.append(rec)
        if it["type"] == "folder":
            out.extend(box_list_recursive(shared, it["id"], child_rel))
    return out


def should_download(item: dict, priority: str) -> Tuple[bool, str]:
    if item["type"] != "file":
        return False, "folder_listing_only"
    if priority == "primary":
        return True, "primary_all"
    ext = Path(item["name"]).suffix.lower()
    if ext in CODE_ONLY_EXT:
        return True, "secondary_code_or_docs"
    if item["size"] <= SECONDARY_MAX_BYTES:
        return True, "secondary_small_file"
    return False, "secondary_inventory_skip_large"


def archive_members(path: Path) -> Optional[str]:
    suf = path.suffix.lower()
    name = path.name.lower()
    try:
        if suf == ".zip" or zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                rows = [f"{i.filename}\t{i.file_size}\t{i.compress_size}" for i in z.infolist()]
            return "\n".join(rows)
    except Exception as e:
        return f"ARCHIVE_LIST_FAILED:{e}"
    return None


def mime_of(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    if mt:
        return mt
    proc = subprocess.run(["file", "-b", "--mime-type", str(path)], capture_output=True, text=True)
    return (proc.stdout or "application/octet-stream").strip()


def compression_status(path: Path, members: Optional[str]) -> str:
    suf = path.suffix.lower()
    if members:
        return "archive_zip" if suf == ".zip" or "zip" in mime_of(path) else "archive"
    if suf in {".gz", ".bz2", ".xz", ".7z", ".zip", ".tar"}:
        return suf.lstrip(".")
    return "uncompressed"


def append_manifest(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "jurisdiction", "source_url", "final_resolved_url", "retrieval_timestamp_utc",
        "filename", "relpath", "byte_size", "sha256", "mime_type", "compression_archive_status",
        "raw_processed_designation", "accompanying_source_page", "original_government_source",
        "priority_set", "download_status", "skip_reason", "expected_size_from_box",
        "archive_members", "box_shared_name", "box_item_id", "http_status",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def download_box_share(spec: dict, manifest_path: Path) -> None:
    shared = spec["shared"]
    jur = spec["jurisdiction"]
    print(f"\n=== BOX {jur} {shared} ===", flush=True)
    try:
        items = box_list_recursive(shared, None, Path("box") / shared)
    except Exception as e:
        append_manifest([{
            "jurisdiction": jur, "source_url": f"https://uchicago.box.com/s/{shared}",
            "final_resolved_url": "", "retrieval_timestamp_utc": utc_now(),
            "filename": "", "relpath": "", "byte_size": "", "sha256": "",
            "mime_type": "", "compression_archive_status": "",
            "raw_processed_designation": spec["designation"],
            "accompanying_source_page": spec["page"],
            "original_government_source": spec["gov"],
            "priority_set": spec["priority"], "download_status": "LIST_FAILED",
            "skip_reason": str(e), "expected_size_from_box": "",
            "archive_members": "", "box_shared_name": shared, "box_item_id": "",
            "http_status": "",
        }])
        print("LIST_FAILED", e, flush=True)
        return

    listing_path = LOG_DIR / "box_listings" / f"{jur}_{shared}.json"
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    listing_path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    rows = []
    for it in items:
        want, reason = should_download(it, spec["priority"])
        dest = RAW_ROOT / jur / it["relpath"]
        source_url = (
            f"https://uchicago.app.box.com/index.php?rm=box_download_shared_file"
            f"&shared_name={shared}&file_id=f_{it['id']}"
            if it["type"] == "file"
            else f"https://uchicago.app.box.com/s/{shared}/folder/{it['id']}"
        )
        rec = {
            "jurisdiction": jur,
            "source_url": source_url,
            "final_resolved_url": "",
            "retrieval_timestamp_utc": utc_now(),
            "filename": it["name"],
            "relpath": str(Path(jur) / it["relpath"]),
            "byte_size": "",
            "sha256": "",
            "mime_type": "",
            "compression_archive_status": "",
            "raw_processed_designation": spec["designation"],
            "accompanying_source_page": spec["page"],
            "original_government_source": spec["gov"],
            "priority_set": spec["priority"],
            "download_status": "",
            "skip_reason": reason,
            "expected_size_from_box": it["size"],
            "archive_members": "",
            "box_shared_name": shared,
            "box_item_id": it["id"],
            "http_status": "",
        }
        if it["type"] == "folder":
            rec["download_status"] = "FOLDER_LISTED"
            (RAW_ROOT / jur / it["relpath"]).mkdir(parents=True, exist_ok=True)
            rows.append(rec)
            continue
        if not want:
            rec["download_status"] = "SKIPPED"
            rows.append(rec)
            print(f"  SKIP {it['name']} ({it['size']} bytes) {reason}", flush=True)
            continue
        if dest.exists() and dest.stat().st_size == it["size"]:
            rec["download_status"] = "ALREADY_PRESENT"
            rec["byte_size"] = dest.stat().st_size
            rec["sha256"] = sha256_file(dest)
            rec["mime_type"] = mime_of(dest)
            members = archive_members(dest)
            rec["archive_members"] = members or ""
            rec["compression_archive_status"] = compression_status(dest, members)
            rec["final_resolved_url"] = source_url
            rows.append(rec)
            print(f"  HAVE {it['name']}", flush=True)
            continue
        print(f"  GET  {it['name']} expect={it['size']}", flush=True)
        # large files: no max-time
        timeout = 0 if it["size"] > 50_000_000 else 600
        code, final, err = curl_download(source_url, dest, timeout=timeout)
        rec["http_status"] = code
        rec["final_resolved_url"] = final
        rec["retrieval_timestamp_utc"] = utc_now()
        if dest.exists() and dest.stat().st_size > 0:
            rec["byte_size"] = dest.stat().st_size
            rec["sha256"] = sha256_file(dest)
            rec["mime_type"] = mime_of(dest)
            members = archive_members(dest)
            rec["archive_members"] = members or ""
            rec["compression_archive_status"] = compression_status(dest, members)
            rec["download_status"] = "DOWNLOADED"
            if it["size"] and dest.stat().st_size != it["size"]:
                rec["download_status"] = "SIZE_MISMATCH"
                rec["skip_reason"] = f"got {dest.stat().st_size} expected {it['size']}"
        else:
            rec["download_status"] = "FAILED"
            rec["skip_reason"] = err[:500]
            print("  FAIL", it["name"], code, err[:200], flush=True)
        rows.append(rec)
    append_manifest(rows, manifest_path)


def download_direct(spec: dict, manifest_path: Path) -> None:
    dest = RAW_ROOT / spec["relpath"]
    print(f"\n=== HTTP {spec['url']} ===", flush=True)
    # HEAD first
    proc = curl(["-I", spec["url"]], timeout=60)
    headers = (proc.stdout or b"").decode("utf-8", "replace")
    (LOG_DIR / "http_heads").mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "http_heads" / Path(spec["relpath"]).name).write_text(headers)
    clen = ""
    for line in headers.splitlines():
        if line.lower().startswith("content-length:"):
            clen = line.split(":", 1)[1].strip()
    rec = {
        "jurisdiction": spec["jurisdiction"],
        "source_url": spec["url"],
        "final_resolved_url": spec["url"],
        "retrieval_timestamp_utc": utc_now(),
        "filename": Path(spec["relpath"]).name,
        "relpath": spec["relpath"],
        "byte_size": "",
        "sha256": "",
        "mime_type": "",
        "compression_archive_status": "",
        "raw_processed_designation": spec["designation"],
        "accompanying_source_page": spec["page"],
        "original_government_source": spec["gov"],
        "priority_set": spec["priority"],
        "download_status": "",
        "skip_reason": f"head_content_length={clen}",
        "expected_size_from_box": clen,
        "archive_members": "",
        "box_shared_name": "",
        "box_item_id": "",
        "http_status": "",
    }
    expected = int(clen) if clen.isdigit() else None
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        rec["download_status"] = "ALREADY_PRESENT"
        rec["byte_size"] = dest.stat().st_size
        rec["sha256"] = sha256_file(dest)
        rec["mime_type"] = mime_of(dest)
        members = archive_members(dest)
        rec["archive_members"] = members or ""
        rec["compression_archive_status"] = compression_status(dest, members)
        append_manifest([rec], manifest_path)
        return
    code, final, err = curl_download(spec["url"], dest, timeout=600)
    rec["http_status"] = code
    rec["final_resolved_url"] = final
    rec["retrieval_timestamp_utc"] = utc_now()
    if dest.exists() and dest.stat().st_size > 0:
        rec["byte_size"] = dest.stat().st_size
        rec["sha256"] = sha256_file(dest)
        rec["mime_type"] = mime_of(dest)
        members = archive_members(dest)
        rec["archive_members"] = members or ""
        rec["compression_archive_status"] = compression_status(dest, members)
        rec["download_status"] = "DOWNLOADED"
    else:
        rec["download_status"] = "FAILED"
        rec["skip_reason"] = err[:500]
    append_manifest([rec], manifest_path)


def clone_cmfproperty(manifest_path: Path) -> None:
    dest = RAW_ROOT / "_shared" / "cmfproperty"
    url = "https://github.com/cmf-uchicago/cmfproperty.git"
    print("\n=== GIT", url, "===", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").exists():
        subprocess.run(["git", "-C", str(dest), "fetch", "--all"], check=False)
    else:
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=False)
    sha = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True)
    rec = {
        "jurisdiction": "ALL",
        "source_url": url,
        "final_resolved_url": url,
        "retrieval_timestamp_utc": utc_now(),
        "filename": "cmfproperty",
        "relpath": "_shared/cmfproperty",
        "byte_size": "",
        "sha256": (sha.stdout or "").strip(),
        "mime_type": "git_repository",
        "compression_archive_status": "uncompressed",
        "raw_processed_designation": "analytic_code",
        "accompanying_source_page": "https://propertytaxproject.uchicago.edu/user-data/",
        "original_government_source": "n/a",
        "priority_set": "hub",
        "download_status": "CLONED" if (dest / ".git").exists() else "FAILED",
        "skip_reason": "git_head_in_sha256_field",
        "expected_size_from_box": "",
        "archive_members": "",
        "box_shared_name": "",
        "box_item_id": "",
        "http_status": "",
    }
    append_manifest([rec], manifest_path)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    COOKIEJAR.parent.mkdir(parents=True, exist_ok=True)
    manifest = ANALYSIS / "file_manifest.csv"
    if manifest.exists():
        # keep a copy but start a fresh write for this run by renaming
        bak = ANALYSIS / f"file_manifest.prev_{int(time.time())}.csv"
        manifest.rename(bak)
        print("rotated previous manifest to", bak, flush=True)

    clone_cmfproperty(manifest)
    for spec in DIRECT_HTTP:
        download_direct(spec, manifest)
    for spec in PRIMARY_BOX:
        download_box_share(spec, manifest)
    for spec in SECONDARY_BOX:
        download_box_share(spec, manifest)
    print("\nDONE manifest=", manifest, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
