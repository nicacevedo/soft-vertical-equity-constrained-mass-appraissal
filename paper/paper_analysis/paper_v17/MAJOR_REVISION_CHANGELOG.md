# Major-Revision Changelog — 2026-09-05

## Reader-facing changes implemented

- Reframed the manuscript as a **CCAO-centered applied-methods and diagnostic paper**.
- Replaced the broad fairness/taxation title with a price-related vertical-equity title and CCAO research-workflow subtitle.
- Rewrote the abstract around the narrow observed-sale estimand, Direct–Surrogate mechanism distinction, complete temporal paths, and explicit non-selection/non-fairness boundary.
- Rewrote the Introduction into a linear CCAO story: baseline tension → train-time translation → path design → contributions → claim boundaries.
- Reorganized Related Work and added the historical vertical-equity bridge from classical log-log assessment models to the current covariance target.
- Added historical bibliography entries for Paglin–Fogarty (1972), Cheng (1974), Edelstein (1979), and Sunderman et al. (1990).
- Renamed the method section to emphasize regularization of a **price-related first-order pattern** rather than a general “regressivity correction.”
- Distinguished the mathematical Direct objective from the empirical **Direct-diagonal** LightGBM implementation.
- Added the required centered-spread post-hoc comparator design implied by the Direct fixed-space corollary.
- Added a domain-specific comparator slot as P1 robustness.
- Reduced the main metric section to a clear hierarchy: `R²`/MAE, PRB/VEI, `beta_log`/`Delta_NL`; moved the full complementary definitions/guidance to the appendix.
- Corrected COD/COV guidance language and kept the May 2026 IAAO revision explicitly labeled as an exposure draft.
- Replaced the baseline main table with a compact four-metric motivation table.
- Removed numeric boldface/asterisk “winner” formatting from appendix path tables pending paired uncertainty.
- Demoted candidate-region/guardrail machinery from the main contribution to a historical exploratory appendix diagnostic.
- Rewrote the Results around complete paths, first-order attenuation, nonlinear failure modes, accuracy–equity trajectories, and temporal instability of raw penalty coordinates.
- Rewrote Discussion/Conclusions around three defensible conclusions and explicit CCAO operational boundaries.
- Removed legacy six-county ATTOM results from active inferential support; the normalized external benchmark appears only as secondary status/robustness material.
- Rewrote reproducibility language to require a final canonical artifact manifest rather than promise a future package.
- Removed all tracked-revision macros and visible old/new text from the revised source.

## Empirical items intentionally left as hidden TODOs

P0: native/custom zero-penalty parity + full path rerun; centered-spread comparator; paired uncertainty; CCAO data/provenance/scope freeze; final table/figure regeneration; implementation scaling audit; final artifact manifest; abstract quantitative freeze.

P1: VEI bootstrap recomputation; MKI tie handling; `Delta_NL` sensitivity; same-date/repeated-PIN leakage audit; retransformation/smearing sensitivity; limited learner-capacity sensitivity; CCAO subgroup/worst-group analysis; sold-versus-unsold transport; one domain-specific comparator; mechanism figure regeneration; temporal uncertainty; external stress-test integration only after its audit closes.

## Build verification

The revised TeX was compiled twice in an isolated syntax-check environment. It has balanced environments, no duplicate labels, no unresolved internal cross-references, and no LaTeX fatal errors. That syntax-check build intentionally used dummy versions of `references.bib` / `references_additions.bib` and did not contain the repository figure directory, so it is **not** a submission PDF. The reader-facing PDF should be regenerated only inside the canonical repository after the P0/P1 empirical TODOs and figure assets are frozen.
