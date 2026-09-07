#!/usr/bin/env python3
"""Stage the four uncitable historical BibTeX entries, with per-field provenance.

paper_v17_option1.tex:30-31 loads only references.bib and
references_additions.bib, so the four historical entries transcribed in
paper/references_major_revision_additions.txt cannot be cited at all: they never
print. Tier B0 stages validated, ready-to-paste BibTeX under bib/ and specifies
the fix. NO FILE UNDER paper/ IS EDITED.

Provenance discipline, per field:
  IN_REPO_TXT        the value comes from paper/references_major_revision_additions.txt
  CROSSREF_DOI       the value comes from the cached authoritative response
  UNVERIFIED_FLAGGED nothing authoritative confirms it; emitted as an explicit
                     % UNVERIFIED_FLAGGED marker in the .bib

and an independent agreement verdict against the cached response:
  EXACT / NORMALIZED_MATCH / CROSSREF_SILENT / CONFLICT

This script is OFFLINE and deterministic. It reads bib/metadata_cache/, which
bib/fetch_metadata.py wrote once; it never performs network I/O itself. Nothing
is inferred and no DOI is invented: the Crossref bibliographic query for
Cheng1974 returned no matching record, so its fields carry the in-repo
transcription with an explicit unverified marker rather than a fabricated DOI.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import b0_common as c

CACHE = c.BIB / "metadata_cache"
BIB_OUT = c.BIB / "staged_historical_entries.bib"
PROV_OUT = c.BIB / "STAGED_BIB_PROVENANCE.md"
SRC_TXT = c.PAPER / "references_major_revision_additions.txt"

FIELDS = ("author", "title", "journal", "year", "volume", "number", "pages", "doi")


def parse_repo_bibtex(text: str) -> dict:
    """Parse the hand-transcribed entries in references_major_revision_additions.txt."""
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*(?:,|$)", body, re.S):
            fields[fm.group(1).strip().lower()] = " ".join(fm.group(2).split())
        out[key] = {"entry_type": kind, "fields": fields}
    return out


def crossref_record(key: str):
    """The cached authoritative record for `key`, or None if there is no match."""
    path = CACHE / f"{key}.json"
    if not path.exists():
        return None
    msg = json.loads(path.read_text(encoding="utf-8"))["message"]
    if "items" in msg:                      # bibliographic query, not a DOI hit
        return None
    return msg


def cr_fields(msg: dict) -> dict:
    """Project a Crossref record onto BibTeX field names."""
    if not msg:
        return {}
    out = {}
    if msg.get("author"):
        out["author"] = " and ".join(
            f"{a.get('family','')}, {a.get('given','')}".strip(", ")
            for a in msg["author"])
    if msg.get("title"):
        out["title"] = msg["title"][0]
    if msg.get("container-title"):
        out["journal"] = msg["container-title"][0]
    for src in ("published-print", "published-online", "issued"):
        dp = (msg.get(src) or {}).get("date-parts") or []
        if dp and dp[0]:
            out["year"] = str(dp[0][0])
            break
    if msg.get("volume"):
        out["volume"] = str(msg["volume"])
    if msg.get("issue"):
        out["number"] = str(msg["issue"])
    if msg.get("page"):
        out["pages"] = str(msg["page"])
    if msg.get("DOI"):
        out["doi"] = msg["DOI"]
    return out


def norm(s: str) -> str:
    """Fold case, accents, punctuation and separators for an agreement verdict."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("--", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def norm_authors(s: str) -> str:
    """Compare author lists on surnames plus first initials only.

    Crossref frequently omits middle initials that the in-repo transcription
    carries (Fogarty, Michael P.; Sunderman, Mark A.), so a surname-plus-initial
    comparison is the honest test of agreement. The richer in-repo form is what
    gets staged; the difference is recorded as CROSSREF_SILENT on that detail.
    """
    people = []
    for part in re.split(r"\s+and\s+", s):
        part = part.strip()
        if not part:
            continue
        if "," in part:
            fam, given = part.split(",", 1)
        else:
            bits = part.split()
            fam, given = bits[-1], " ".join(bits[:-1])
        people.append(norm(fam) + (norm(given)[:1] if given.strip() else ""))
    return "|".join(people)


def verdict(field: str, repo_val: str, cr_val: str) -> str:
    if not cr_val:
        return "CROSSREF_SILENT"
    if not repo_val:
        return "CROSSREF_SILENT" if not cr_val else "EXACT"
    if repo_val == cr_val:
        return "EXACT"
    if field == "author":
        return ("NORMALIZED_MATCH"
                if norm_authors(repo_val) == norm_authors(cr_val) else "CONFLICT")
    if field == "pages":
        # Crossref often records only the START page (Edelstein: "753"). A
        # start-page match does NOT verify the end page, so a range against a
        # single page is CROSSREF_SILENT on the range -- which routes the field
        # to UNVERIFIED_FLAGGED -- rather than a normalized match.
        rs = re.split(r"-+", repo_val)[0].strip()
        cs = re.split(r"-+", cr_val)[0].strip()
        if rs != cs:
            return "CONFLICT"
        repo_is_range = bool(re.search(r"-+\s*\d", repo_val))
        cr_is_range = bool(re.search(r"-+\s*\d", cr_val))
        if repo_is_range and not cr_is_range:
            return "CROSSREF_SILENT"
        return "EXACT" if norm(repo_val) == norm(cr_val) else "NORMALIZED_MATCH"
    if norm(repo_val) == norm(cr_val):
        return "NORMALIZED_MATCH"
    return "CONFLICT"


def _detail(field: str, rv: str, cv: str, v: str) -> str:
    """Say precisely what differs, so a normalized match is not a black box."""
    if v == "EXACT":
        return ""
    if v == "CROSSREF_SILENT" and field == "pages" and cv:
        return (f"start page {cv} confirmed; the end page in the staged range "
                f"{rv} is NOT confirmed by any authoritative source")
    if v == "CROSSREF_SILENT":
        return "the cached authoritative response does not carry this field"
    if v == "NORMALIZED_MATCH" and field == "author":
        return ("same authors; the staged form keeps middle initials that the "
                "authoritative response omits")
    if v == "NORMALIZED_MATCH":
        return (f"differs only after normalization (case, accents, punctuation, "
                f"leading article): authoritative form is {cv!r}")
    if v == "CONFLICT":
        return f"substantive disagreement: authoritative form is {cv!r}"
    return ""


def main() -> int:
    repo = parse_repo_bibtex(c.read_text(SRC_TXT))
    idx = json.loads((CACHE / "RETRIEVAL_INDEX.json").read_text(encoding="utf-8"))

    entries, prov, problems = [], [], []
    for key in ("PaglinFogarty1972", "Cheng1974", "Edelstein1979",
                "SundermanEtAl1990"):
        if key not in repo:
            problems.append(f"{key} absent from {c.rel(SRC_TXT)}")
            continue
        rf = repo[key]["fields"]
        msg = crossref_record(key)
        cf = cr_fields(msg)
        retrieval = idx["entries"].get(key, {})

        lines = [f"@{repo[key]['entry_type']}{{{key},"]
        rows = []
        for f in FIELDS:
            rv, cv = rf.get(f, ""), cf.get(f, "")
            v = verdict(f, rv, cv)
            if rv:
                staged, source = rv, ("CROSSREF_DOI" if v == "EXACT"
                                      else "IN_REPO_TXT")
            elif cv:
                staged, source = cv, "CROSSREF_DOI"
            else:
                staged, source = "", "UNVERIFIED_FLAGGED"
            # a value only the hand transcription supports is not authoritative
            if staged and source == "IN_REPO_TXT" and v == "CROSSREF_SILENT":
                source = "UNVERIFIED_FLAGGED"
            if v == "CONFLICT":
                source = "UNVERIFIED_FLAGGED"

            note = _detail(f, rv, cv, v)
            if staged:
                lines.append(f"  {f:8} = {{{staged}}},")
                if source == "UNVERIFIED_FLAGGED":
                    lines.append(f"  % UNVERIFIED_FLAGGED: {f} -- "
                                 + (note or f"{v.lower()} against the cached "
                                            "authoritative response")
                                 + "; verify before pasting")
            else:
                lines.append(f"  % UNVERIFIED_FLAGGED: {f} -- no value in "
                             f"{c.rel(SRC_TXT)} and none in the cached response")
            rows.append({"field": f, "staged_value": staged, "source": source,
                         "crossref_agreement": v, "repo_txt_value": rv,
                         "crossref_value": cv,
                         "detail": _detail(f, rv, cv, v)})
            if source not in c.BIB_FIELD_SOURCES:
                problems.append(f"{key}.{f}: bad source {source}")
            if v not in c.BIB_AGREEMENT:
                problems.append(f"{key}.{f}: bad agreement {v}")

        lines[-1] = lines[-1].rstrip(",") if lines[-1].endswith(",") else lines[-1]
        lines.append("}")
        entries.append("\n".join(lines))
        prov.append({"citekey": key, "retrieval": retrieval, "rows": rows,
                     "crossref_matched": msg is not None})

    header = f"""% Staged BibTeX for the four historical references that
% paper/paper_v17_option1.tex currently CANNOT cite.
%
% Why they are uncitable: lines 30-31 of the manuscript load only
% references.bib and references_additions.bib. The entries transcribed in
% {c.rel(SRC_TXT)}
% are in no loaded file, so \\cite of any of them prints nothing.
%
% The fix, specified in MANUSCRIPT_REVISION_SPEC.md and NOT applied here (no
% file under paper/ is edited by Tier B0):
%   1. add a third \\addbibresource for the file these entries live in, and
%   2. actually \\cite them where the historical vertical-equity literature is
%      discussed.
%
% Provenance: every field below is either confirmed against a cached
% authoritative Crossref response or carries an explicit % UNVERIFIED_FLAGGED
% marker. Per-field detail is in bib/STAGED_BIB_PROVENANCE.md. Nothing is
% inferred and no DOI is invented.
"""
    c.write_text(BIB_OUT, header + "\n" + "\n\n".join(entries) + "\n")
    c.write_text(PROV_OUT, render_prov(prov))

    print(f"wrote {c.rel(BIB_OUT)}")
    print(f"wrote {c.rel(PROV_OUT)}")
    for p in prov:
        n_flag = sum(1 for r in p["rows"] if r["source"] == "UNVERIFIED_FLAGGED")
        print(f"  {p['citekey']:20} crossref_matched={str(p['crossref_matched']):5} "
              f"unverified_fields={n_flag}")
    if problems:
        print(f"\n  PROBLEMS ({len(problems)}):")
        for x in problems:
            print("   -", x)
    return 1 if problems else 0


def render_prov(prov: list) -> str:
    L = ["# Staged bibliography -- per-field provenance", "",
         "The four entries below cannot currently be cited from "
         "`paper/paper_v17_option1.tex`: lines 30-31 load only `references.bib` "
         "and `references_additions.bib`, and these entries live in "
         "`paper/references_major_revision_additions.txt`, which no "
         "`\\addbibresource` loads. So `\\cite` of any of them prints nothing.",
         "",
         "`bib/staged_historical_entries.bib` holds ready-to-paste BibTeX. "
         "**No file under `paper/` was edited.**", "",
         "## Discipline", "",
         "- Authoritative sources only: Crossref by DOI, or a Crossref "
         "bibliographic query where no DOI is known.",
         "- Nothing transcribed by hand into a field that an authoritative "
         "response could confirm; nothing inferred; **no DOI invented**.",
         "- `source` says where the staged value came from; "
         "`crossref_agreement` is an independent verdict on whether the cached "
         "response confirms it.",
         "- A value that only the in-repo transcription supports is marked "
         "`UNVERIFIED_FLAGGED`, not `IN_REPO_TXT`, because the transcription is "
         "not an authoritative source. `references_additions.bib` sets the "
         "standard the writer should meet: its metadata was obtained by DOI "
         "content negotiation against doi.org rather than typed by hand.",
         "- The fetch is a one-time step in `bib/fetch_metadata.py`. Neither the "
         "build scripts nor the test suite performs network I/O; they read the "
         "cached responses in `bib/metadata_cache/`.", ""]
    for p in prov:
        r = p["retrieval"]
        L += [f"## `{p['citekey']}`", ""]
        if r:
            L += [f"- lookup: **{r.get('lookup_kind','')}**",
                  f"- request: `{r.get('request_url','')}`",
                  f"- retrieved: {r.get('retrieved_utc','')}",
                  f"- response: `{r.get('response_file','')}` "
                  f"({r.get('response_bytes','?')} bytes, sha256 "
                  f"`{str(r.get('response_sha256',''))[:32]}...`)",
                  f"- authoritative record matched: "
                  f"**{'yes' if p['crossref_matched'] else 'NO'}**", ""]
        if not p["crossref_matched"]:
            L += ["> The bibliographic query returned no record for this work. "
                  "Every field therefore rests on the in-repo transcription "
                  "alone and is marked `UNVERIFIED_FLAGGED`. No DOI is supplied, "
                  "because none was found and none may be invented.", ""]
        L += ["| field | staged value | source | crossref agreement | note |",
              "|---|---|---|---|---|"]
        for row in p["rows"]:
            sv = row["staged_value"] or "*(none)*"
            note = row["detail"] or "—"
            L.append(f"| `{row['field']}` | {sv} | `{row['source']}` | "
                     f"`{row['crossref_agreement']}` | {note} |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
