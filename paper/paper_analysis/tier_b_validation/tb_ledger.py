#!/usr/bin/env python3
"""The Tier-B provenance ledger -- an INPUT TO BE VERIFIED, not an authority.

The frozen Tier-B0 numeric map indexes only the *baseline* manuscript's tokens.
The Tier-B writing pass adds manuscript-facing numbers the baseline never
printed (matched-beta, the centered-spread comparator, temporal robustness, the
ED2 counts, the Cell-C rows). Those need a second resolution path -- and a
second place to write a number down is a second place for a wrong number to
live, unless the writing-down is *checked*.

So a ledger entry proves nothing by itself. For every entry the validator:

  1. reopens the frozen artifact under ``analysis/``;
  2. verifies its sha256 -- against the file AND against
     ``FINAL_EVIDENCE_MANIFEST.json``, so an entry cannot pin an artifact the
     frozen index does not know;
  3. executes the selector in ``SELECTOR_GRAMMAR.md`` form;
  4. re-extracts ``raw_value`` and requires byte-exact agreement with the
     literal recorded in the entry;
  5. applies ``value_transform`` from the closed vocabulary;
  6. applies ``rounding_rule`` from the closed vocabulary;
  7. compares the result with ``rendered_value``, and -- when the entry declares
     a manuscript anchor -- with what the ``.tex`` actually prints there.

Every step routes through the frozen ``b0_common`` implementation, so values are
carried as literal decimal TEXT and converted with ``decimal.Decimal``. Binary
float is never used: ``rounding_rule`` uses ``ROUND_HALF_UP``, not Python's
banker's ``round()``, so 0.025 displays as 0.03 rather than 0.02, and
``times_100`` is an exact decimal ``scaleb`` rather than a multiplication.

The two stages stay separate on purpose. ``rounding_rule`` never carries a unit
conversion: MAPE is a fraction in the artifact and a percent in the manuscript,
while VEI and COD are already percent. Conflating the stages is how a percent
gets multiplied twice, or not at all.

At B1.0 the ledger has no manuscript entries -- B1.1 through B1.4 add no printed
number. ``spec/ledger_selftest.yaml`` therefore exercises the machinery against
frozen values that the baseline manuscript already prints, and ``selftest()``
additionally proves each step is load-bearing by mutating it and requiring a
rejection. A verifier that only ever passes is not verifying anything.
"""
from __future__ import annotations

import sys

# Set before ANY project import: running the gate must not leave a stray
# __pycache__ anywhere, least of all under analysis/.
sys.dont_write_bytecode = True

import copy
import re

import tb_common as tb

REQUIRED_FIELDS = ("artifact_path", "artifact_sha256", "selector", "raw_value",
                   "value_transform", "rounding_rule", "rendered_value")

# Addressing and semantics carried alongside the seven verified fields.
OPTIONAL_FIELDS = ("entry_id", "artifact_kind", "column", "metric",
                   "manuscript_anchor", "latex_label", "counting_unit",
                   "reference_cell", "comparison_purpose",
                   "primary_or_sensitivity", "guidance_status",
                   "attained_status", "added_by_stage", "role", "note")

LEDGER_FILE = tb.TB_LEDGER / "tier_b_numeric_ledger.yaml"
SELFTEST_FILE = tb.TB_SPEC / "ledger_selftest.yaml"


class LedgerError(RuntimeError):
    """A ledger entry failed one of the seven verification steps."""


def load(path=LEDGER_FILE) -> list:
    doc = tb.read_yaml(path) or {}
    return list(doc.get("entries") or [])


def _manifest_sha(art: str):
    man = tb.frozen_manifest()
    rec = man["artifacts"].get(art)
    if rec is None:
        return None, f"artifact is not in FINAL_EVIDENCE_MANIFEST.json: {art}"
    if not rec.get("present_in_worktree", False):
        return None, (f"artifact is recorded present_in_worktree=false and may "
                      f"not be read: {art}")
    return rec["sha256"], None


def verify_entry(entry: dict, tex=None, require_manuscript: bool = True) -> dict:
    """Run the seven steps. Returns a per-step record; raises nothing."""
    c, _ = tb.frozen_b0()
    eid = entry.get("entry_id") or entry.get("metric") or "(unnamed)"
    steps, problems = {}, []

    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    unknown = [k for k in entry
               if k not in REQUIRED_FIELDS and k not in OPTIONAL_FIELDS]
    if missing:
        problems.append(f"{eid}: missing required field(s) {missing}")
    if unknown:
        problems.append(f"{eid}: unknown field(s) {unknown}")
    if missing:
        return {"entry_id": eid, "ok": False, "steps": steps,
                "problems": problems}

    art = entry["artifact_path"]
    path = tb.REPO / art

    # 1. reopen the frozen artifact
    if not str(path.resolve()).startswith(str(tb.ANALYSIS.resolve())):
        problems.append(f"{eid}: artifact_path must be under analysis/: {art}")
    if not path.exists():
        problems.append(f"{eid}: artifact does not exist: {art}")
        return {"entry_id": eid, "ok": False, "steps": steps,
                "problems": problems}
    steps["1_reopened"] = art

    # 2. verify sha256 against the file and against the frozen manifest
    actual = tb.sha256_file(path)
    steps["2_sha256_file"] = actual
    if actual != entry["artifact_sha256"]:
        problems.append(f"{eid}: sha256 mismatch: file {actual} != ledger "
                        f"{entry['artifact_sha256']}")
    man_sha, man_err = _manifest_sha(art)
    steps["2_sha256_manifest"] = man_sha
    if man_err:
        problems.append(f"{eid}: {man_err}")
    elif man_sha != actual:
        problems.append(f"{eid}: sha256 disagrees with the frozen manifest: "
                        f"file {actual} != manifest {man_sha}")

    kind = entry.get("artifact_kind") or path.suffix.lstrip(".").lower()
    kind = {"yml": "yaml", "md": "markdown"}.get(kind, kind)

    # 3. execute the selector, 4. recover the raw value
    try:
        col = entry.get("column") or entry.get("metric") or ""
        raw = c.resolve_selector(path, kind, entry["selector"], col)
        raw = "" if raw is None else str(raw)
        steps["3_selector"] = entry["selector"]
        steps["4_raw_value"] = raw
    except Exception as e:
        problems.append(f"{eid}: selector failed: {type(e).__name__}: {e}")
        return {"entry_id": eid, "ok": False, "steps": steps,
                "problems": problems}

    if raw != str(entry["raw_value"]):
        problems.append(f"{eid}: raw_value is not byte-exact: artifact {raw!r} "
                        f"!= ledger {entry['raw_value']!r}")

    # 5. transform, 6. rounding  (both exact Decimal, from the closed vocabularies)
    tr, rr = entry["value_transform"], entry["rounding_rule"]
    if not c.valid_transform(tr):
        problems.append(f"{eid}: value_transform {tr!r} is outside the closed vocabulary")
    if not c.valid_rounding(rr):
        problems.append(f"{eid}: rounding_rule {rr!r} is outside the closed vocabulary")
    disp = None
    if c.valid_transform(tr) and c.valid_rounding(rr):
        try:
            transformed = c.apply_transform(raw, tr)
            steps["5_transformed"] = transformed
            disp = c.apply_rounding(transformed, rr)
            steps["6_rounded"] = disp
        except Exception as e:
            problems.append(f"{eid}: value pipeline failed: {type(e).__name__}: {e}")

    # 7. compare with rendered_value, and with what the .tex prints
    rendered = str(entry["rendered_value"])
    steps["7_rendered_value"] = rendered
    if disp is not None:
        rd = rendered.replace("$", "").replace("\\", "").strip()
        if disp != rd and disp != rendered:
            problems.append(f"{eid}: rendered_value {rendered!r} is not the "
                            f"pipeline output {disp!r} (raw {raw!r}, {tr} + {rr})")

    anchor = entry.get("manuscript_anchor")
    if require_manuscript and anchor:
        tex = tex or tb.live_tex()
        found = _rendered_value_is_printed(tex, anchor, rendered)
        steps["7_printed_at_anchor"] = found
        if not found:
            problems.append(f"{eid}: rendered_value {rendered!r} does not appear "
                            f"in the ACTIVE build at anchor {anchor!r}")

    return {"entry_id": eid, "ok": not problems, "steps": steps,
            "problems": problems}


def _rendered_value_is_printed(tex, anchor: str, rendered: str) -> bool:
    """Is this value printed in the ACTIVE build under this anchor?

    Anchors, not line numbers: every baseline line number is stale the moment
    B1.1 deletes a table, which SELECTOR_GRAMMAR.md section 4 makes explicit.

    The search is over the SOURCE, not over the coverage audit's token
    population, and that difference matters. The frozen audit masks math
    environments -- deliberately, so that exponents and indices are not counted
    as results -- so a value written as ``$40.04$`` is invisible to it. It is
    still printed, so the ledger must see it: this is the check that a value the
    audit cannot reach is nevertheless tied to a frozen artifact.
    """
    variants = {rendered}
    plain = rendered.replace("$", "").replace("\\", "").strip()
    variants |= {plain, plain.replace(",", ""), plain.lstrip("-"),
                 plain.replace(",", "").lstrip("-")}
    for v in sorted(variants, key=len, reverse=True):
        if not v:
            continue
        # a numeric value must match as a whole token, not inside a longer one
        # A trailing '.' is a sentence period, not part of the number, so it must
        # NOT block a match -- otherwise a value at the end of a sentence reads
        # as absent. A trailing digit still does block one, which is what stops
        # "0.01" from matching inside "0.0118".
        pat = (re.escape(v) if re.search(r"[A-Za-z_]", v)
               else r"(?<![0-9A-Za-z.,])" + re.escape(v) + r"(?![0-9A-Za-z,])")
        for m in re.finditer(pat, tex.src):
            if tex.bucket_at(m.start()) != "ACTIVE":
                continue
            if tex.nearest_label(m.start()) == anchor:
                return True
    return False


def verify_all(tex=None, path=LEDGER_FILE, require_manuscript=True) -> dict:
    entries = load(path)
    tex = tex or tb.live_tex()
    results = [verify_entry(e, tex, require_manuscript) for e in entries]
    ids = [r["entry_id"] for r in results]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    problems = [p for r in results for p in r["problems"]]
    if dup:
        problems.append(f"duplicate ledger entry_id(s): {dup}")
    # A/C role guard, the same one the frozen maps carry.
    for e in entries:
        rc, cp = e.get("reference_cell"), e.get("comparison_purpose")
        if rc == "A" and cp == "PENALTY_ISOLATING":
            problems.append(f"{e.get('entry_id')}: reference_cell A may not carry "
                            f"comparison_purpose PENALTY_ISOLATING")
        if rc == "C" and cp == "WORKFLOW_BENCHMARK":
            problems.append(f"{e.get('entry_id')}: reference_cell C may not carry "
                            f"comparison_purpose WORKFLOW_BENCHMARK")
    return {"n_entries": len(entries), "results": results, "problems": problems}


# --------------------------------------------------------------------------
# Self-test: prove every step is load-bearing, by breaking it.
# --------------------------------------------------------------------------
def selftest() -> list:
    """Positive and negative tests of the seven-step verifier."""
    problems = []
    entries = load(SELFTEST_FILE)
    if not entries:
        return ["ledger self-test has no entries; the verifier is unproven"]
    tex = tb.live_tex()

    for e in entries:
        r = verify_entry(e, tex, require_manuscript=True)
        if not r["ok"]:
            problems += [f"selftest POSITIVE failed: {p}" for p in r["problems"]]

    base = entries[0]

    def _must_reject(label, mutate):
        m = copy.deepcopy(base)
        mutate(m)
        r = verify_entry(m, tex, require_manuscript=True)
        if r["ok"]:
            problems.append(f"selftest NEGATIVE '{label}' was ACCEPTED; step is "
                            f"not load-bearing")

    _must_reject("wrong artifact sha256",
                 lambda m: m.update(artifact_sha256="0" * 64))
    _must_reject("selector that matches nothing",
                 lambda m: m.update(selector=m["selector"] + ";evaluation=nope"))
    _must_reject("raw_value silently edited",
                 lambda m: m.update(raw_value="0.123456789"))
    _must_reject("transform outside the vocabulary",
                 lambda m: m.update(value_transform="times_1000"))
    _must_reject("rounding outside the vocabulary",
                 lambda m: m.update(rounding_rule="banker:3"))
    _must_reject("rendered_value not the pipeline output",
                 lambda m: m.update(rendered_value="1.234"))
    _must_reject("value not actually printed at the declared anchor",
                 lambda m: m.update(manuscript_anchor="tab:feature_groups"))

    # ROUND_HALF_UP, not banker's rounding: 0.025 -> 0.03 at two places.
    c, _ = tb.frozen_b0()
    if c.apply_rounding("0.025", "half_up:2") != "0.03":
        problems.append("rounding is not ROUND_HALF_UP: 0.025 must display 0.03")
    if c.apply_transform("0.2120088150100442", "times_100") != "21.20088150100442":
        problems.append("times_100 is not an exact decimal scaleb")
    return problems


if __name__ == "__main__":
    import json
    print(json.dumps(verify_all(), indent=2, sort_keys=True, default=str)[:4000])
    print("SELFTEST PROBLEMS:", selftest())
