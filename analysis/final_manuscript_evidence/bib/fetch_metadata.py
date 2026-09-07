#!/usr/bin/env python3
"""ONE-TIME authoritative metadata fetch for the four uncitable historical entries.

This is the ONLY file in the Tier-B0 area that performs network I/O, and it
lives outside code/ deliberately: the build scripts and the test suite must stay
deterministic and offline. They read the cache this script writes; they never
call it. tests/test_b0_isolation.py asserts that no module under code/ or tests/
imports it or performs network I/O of its own.

Sources are authoritative only: Crossref by DOI where a DOI is known, and a
Crossref bibliographic query where it is not. Nothing is transcribed by hand,
nothing is inferred, and no DOI is invented. Each response is stored verbatim
with its request URL, a UTC retrieval timestamp and the sha256 of the bytes, so
every staged field can be traced to the exact response it came from. This
mirrors the precedent set by P1's provenance/ed2_source_cache.

Usage:  python bib/fetch_metadata.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "metadata_cache"
UA = "TierB0-manuscript-evidence/1.0 (mailto:nacevedo@mit.edu)"

# Transcription base: paper/references_major_revision_additions.txt.
# The DOI (where present) comes from that file; everything else is fetched.
TARGETS = [
    {"citekey": "PaglinFogarty1972", "doi": "10.1086/NTJ41791839"},
    {"citekey": "Cheng1974", "doi": None,
     "query": "Property Taxation, Assessment Performance and Its Measurement"},
    {"citekey": "Edelstein1979", "doi": "10.2307/2330450"},
    {"citekey": "SundermanEtAl1990", "doi": "10.1080/10835547.1990.12090625"},
]


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read()


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    index = {}
    for t in TARGETS:
        key = t["citekey"]
        if t["doi"]:
            url = "https://api.crossref.org/works/" + urllib.parse.quote(t["doi"])
            kind = "CROSSREF_DOI"
        else:
            url = ("https://api.crossref.org/works?rows=5&query.bibliographic="
                   + urllib.parse.quote(t["query"]))
            kind = "CROSSREF_BIBLIOGRAPHIC_QUERY"
        print(f"fetching {key} <- {url}")
        try:
            raw = get(url)
        except Exception as e:
            print(f"  FAILED: {e}. Leaving {key} without a cached response; every "
                  "field it would have supplied stays UNVERIFIED_FLAGGED.")
            index[key] = {"lookup_kind": kind, "request_url": url,
                          "status": "FETCH_FAILED", "error": str(e)}
            continue
        path = CACHE / f"{key}.json"
        path.write_bytes(raw)
        index[key] = {
            "lookup_kind": kind,
            "request_url": url,
            "doi_from_repo_txt": t["doi"],
            "response_file": path.name,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "response_bytes": len(raw),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "status": "OK",
        }
        print(f"  ok  {len(raw)} bytes  sha256 {index[key]['response_sha256'][:16]}...")

    idx = CACHE / "RETRIEVAL_INDEX.json"
    idx.write_text(json.dumps({
        "schema_version": 1,
        "what": "Verbatim authoritative metadata responses for the four "
                "historical entries that paper_v17_option1.tex cannot cite.",
        "discipline": "Authoritative sources only. Nothing transcribed by hand, "
                      "nothing inferred, no DOI invented. Fields the responses "
                      "do not confirm stay UNVERIFIED_FLAGGED.",
        "generated_by": "analysis/final_manuscript_evidence/bib/fetch_metadata.py",
        "note": "Run manually. No build script or test performs network I/O.",
        "entries": index,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {idx}")
    return 0 if all(v["status"] == "OK" for v in index.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
