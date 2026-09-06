# Claude Code P0 Handoff

Baseline commit:
b878b00886584d8d81402a77fd19b93599204dd9

## Canonical manuscript

paper/paper_v17_option1.tex
paper/paper_v17_option1.pdf

These are the only current manuscript baselines.

paper_v17_option2, paper_v15, paper_v12, paper_v6, and all older
manuscripts are historical only. Do not use them as edit baselines.

## Authority order

1. Repository code, configurations, data provenance, and executed outputs
   are authoritative for implementation facts.

2. ADDITIONAL CONTEXT/P0_EXECUTION_SPEC.md
   is the authoritative work order for this pass.

3. ADDITIONAL CONTEXT/FINAL_SCIENTIFIC_AUDIT.md
   provides the detailed referee rationale and evidence.

4. ADDITIONAL CONTEXT/CHATGPT_ASSESSMENT_OF_CLAUDE_AUDIT.md
   contains later adjudication and corrections to the audit.

When the audit and the assessment disagree on an implementation-level
claim, verify it against the repository rather than choosing either by
authority.

## Scope of this pass

P0 scientific validation only.

Do not rewrite:
- Abstract
- Introduction
- Results
- Discussion
- Conclusion
- title
- scientific claims

until the P0 experiments are complete.

Create all new work additively under:

analysis/p0_major_revision_validation/

Do not overwrite frozen prior analyses.

## First action

Start read-only.

Map:
- executed Direct and Surrogate implementations;
- canonical 994-tree configuration;
- rho=0 controls and any existing parity rerun;
- seven rolling-origin splits;
- held-out split;
- 2025 forward split;
- existing frozen rho paths;
- exact scripts/configs/jobs that produced them.

Return a proposed execution plan before editing code or launching jobs.