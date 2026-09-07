# Staged bibliography -- per-field provenance

The four entries below cannot currently be cited from `paper/paper_v17_option1.tex`: lines 30-31 load only `references.bib` and `references_additions.bib`, and these entries live in `paper/references_major_revision_additions.txt`, which no `\addbibresource` loads. So `\cite` of any of them prints nothing.

`bib/staged_historical_entries.bib` holds ready-to-paste BibTeX. **No file under `paper/` was edited.**

## Discipline

- Authoritative sources only: Crossref by DOI, or a Crossref bibliographic query where no DOI is known.
- Nothing transcribed by hand into a field that an authoritative response could confirm; nothing inferred; **no DOI invented**.
- `source` says where the staged value came from; `crossref_agreement` is an independent verdict on whether the cached response confirms it.
- A value that only the in-repo transcription supports is marked `UNVERIFIED_FLAGGED`, not `IN_REPO_TXT`, because the transcription is not an authoritative source. `references_additions.bib` sets the standard the writer should meet: its metadata was obtained by DOI content negotiation against doi.org rather than typed by hand.
- The fetch is a one-time step in `bib/fetch_metadata.py`. Neither the build scripts nor the test suite performs network I/O; they read the cached responses in `bib/metadata_cache/`.

## `PaglinFogarty1972`

- lookup: **CROSSREF_DOI**
- request: `https://api.crossref.org/works/10.1086/NTJ41791839`
- retrieved: 2026-09-07T08:12:24+00:00
- response: `PaglinFogarty1972.json` (2088 bytes, sha256 `425fa63614f5e020705459c92bbdbcc0...`)
- authoritative record matched: **yes**

| field | staged value | source | crossref agreement | note |
|---|---|---|---|---|
| `author` | Paglin, Morton and Fogarty, Michael P. | `IN_REPO_TXT` | `NORMALIZED_MATCH` | same authors; the staged form keeps middle initials that the authoritative response omits |
| `title` | Equity and the Property Tax: A New Conceptual Focus | `IN_REPO_TXT` | `NORMALIZED_MATCH` | differs only after normalization (case, accents, punctuation, leading article): authoritative form is 'EQUITY AND THE PROPERTY TAX: A NEW CONCEPTUAL FOCUS' |
| `journal` | National Tax Journal | `CROSSREF_DOI` | `EXACT` | — |
| `year` | 1972 | `CROSSREF_DOI` | `EXACT` | — |
| `volume` | 25 | `CROSSREF_DOI` | `EXACT` | — |
| `number` | 4 | `CROSSREF_DOI` | `EXACT` | — |
| `pages` | 557--565 | `CROSSREF_DOI` | `EXACT` | — |
| `doi` | 10.1086/NTJ41791839 | `IN_REPO_TXT` | `NORMALIZED_MATCH` | differs only after normalization (case, accents, punctuation, leading article): authoritative form is '10.1086/ntj41791839' |

## `Cheng1974`

- lookup: **CROSSREF_BIBLIOGRAPHIC_QUERY**
- request: `https://api.crossref.org/works?rows=5&query.bibliographic=Property%20Taxation%2C%20Assessment%20Performance%20and%20Its%20Measurement`
- retrieved: 2026-09-07T08:12:25+00:00
- response: `Cheng1974.json` (16927 bytes, sha256 `bebaef973d47359f9134c48e53c4048b...`)
- authoritative record matched: **NO**

> The bibliographic query returned no record for this work. Every field therefore rests on the in-repo transcription alone and is marked `UNVERIFIED_FLAGGED`. No DOI is supplied, because none was found and none may be invented.

| field | staged value | source | crossref agreement | note |
|---|---|---|---|---|
| `author` | Cheng, Pao Lun | `UNVERIFIED_FLAGGED` | `CROSSREF_SILENT` | the cached authoritative response does not carry this field |
| `title` | Property Taxation, Assessment Performance and Its Measurement | `UNVERIFIED_FLAGGED` | `CROSSREF_SILENT` | the cached authoritative response does not carry this field |
| `journal` | Public Finance | `UNVERIFIED_FLAGGED` | `CROSSREF_SILENT` | the cached authoritative response does not carry this field |
| `year` | 1974 | `UNVERIFIED_FLAGGED` | `CROSSREF_SILENT` | the cached authoritative response does not carry this field |
| `volume` | 29 | `UNVERIFIED_FLAGGED` | `CROSSREF_SILENT` | the cached authoritative response does not carry this field |
| `number` | 3 | `UNVERIFIED_FLAGGED` | `CROSSREF_SILENT` | the cached authoritative response does not carry this field |
| `pages` | 268--284 | `UNVERIFIED_FLAGGED` | `CROSSREF_SILENT` | the cached authoritative response does not carry this field |
| `doi` | *(none)* | `UNVERIFIED_FLAGGED` | `CROSSREF_SILENT` | the cached authoritative response does not carry this field |

## `Edelstein1979`

- lookup: **CROSSREF_DOI**
- request: `https://api.crossref.org/works/10.2307/2330450`
- retrieved: 2026-09-07T08:12:25+00:00
- response: `Edelstein1979.json` (1536 bytes, sha256 `32b597c92e9c5a9cf9bcb8a41258b736...`)
- authoritative record matched: **yes**

| field | staged value | source | crossref agreement | note |
|---|---|---|---|---|
| `author` | Edelstein, Robert H. | `CROSSREF_DOI` | `EXACT` | — |
| `title` | An Appraisal of Residential Property Tax Regressivity | `CROSSREF_DOI` | `EXACT` | — |
| `journal` | Journal of Financial and Quantitative Analysis | `IN_REPO_TXT` | `NORMALIZED_MATCH` | differs only after normalization (case, accents, punctuation, leading article): authoritative form is 'The Journal of Financial and Quantitative Analysis' |
| `year` | 1979 | `CROSSREF_DOI` | `EXACT` | — |
| `volume` | 14 | `CROSSREF_DOI` | `EXACT` | — |
| `number` | 4 | `CROSSREF_DOI` | `EXACT` | — |
| `pages` | 753--768 | `UNVERIFIED_FLAGGED` | `CROSSREF_SILENT` | start page 753 confirmed; the end page in the staged range 753--768 is NOT confirmed by any authoritative source |
| `doi` | 10.2307/2330450 | `CROSSREF_DOI` | `EXACT` | — |

## `SundermanEtAl1990`

- lookup: **CROSSREF_DOI**
- request: `https://api.crossref.org/works/10.1080/10835547.1990.12090625`
- retrieved: 2026-09-07T08:12:25+00:00
- response: `SundermanEtAl1990.json` (5423 bytes, sha256 `fe5ea3a8dcb4863b95ed3a73d6f462c0...`)
- authoritative record matched: **yes**

| field | staged value | source | crossref agreement | note |
|---|---|---|---|---|
| `author` | Sunderman, Mark A. and Birch, John W. and Cannaday, Roger E. and Hamilton, Thomas W. | `IN_REPO_TXT` | `NORMALIZED_MATCH` | same authors; the staged form keeps middle initials that the authoritative response omits |
| `title` | Testing for Vertical Inequity in Property Tax Systems | `CROSSREF_DOI` | `EXACT` | — |
| `journal` | Journal of Real Estate Research | `CROSSREF_DOI` | `EXACT` | — |
| `year` | 1990 | `CROSSREF_DOI` | `EXACT` | — |
| `volume` | 5 | `CROSSREF_DOI` | `EXACT` | — |
| `number` | 3 | `CROSSREF_DOI` | `EXACT` | — |
| `pages` | 319--334 | `CROSSREF_DOI` | `EXACT` | — |
| `doi` | 10.1080/10835547.1990.12090625 | `CROSSREF_DOI` | `EXACT` | — |

