# Frozen 2025 forward evaluation

Written at 2026-09-04T22:37:52Z.
Candidate regions, baselines, grids, and the forward freeze were not modified from 2025 outcomes.

## Evaluation layers

No independent pre-2025 TEST split exists. Layers: `CV_FOLD`, `CV_OOF` (fold-mean of frozen 2018–2024 metrics; **not** an independent test), `FORWARD_2025`.

## Rubric

```
FORWARD_SUPPORTS_CV: 2025 equity correction has the same sign as CV at activity
and at the 25% A_beta anchor; 50% A_beta remains attained or still reduces
|beta|; no reversal of PRB/beta toward worse regressivity; Delta_NMSE has the
same sign as CV (or remains near zero).
FORWARD_PARTIAL: some frozen anchors transfer, others do not, with no wholesale
reversal of the correction mechanism.
FORWARD_WEAKENS_CV: correction remains same-signed but is materially smaller, or
predictive cost is materially larger, at the frozen activity/25%/50% coordinates.
FORWARD_REVERSAL: at a frozen activity or 25%/50% anchor, 2025 PRB or beta_log
moves away from the ideal relative to that split's own baseline.
```

## Sample sizes

| jurisdiction | n_train | n_eval_2025 | Var_train(log price) |
|---|---:|---:|---:|
| wayne | 164695 | 17347 | 0.3962 |
| philadelphia | 63151 | 2921 | 0.3988 |
| st_louis_county | 131321 | 13719 | 0.5594 |
| allegheny | 110681 | 12113 | 0.4249 |
| maricopa | 553132 | 47512 | 0.3200 |
| king | 177707 | 17812 | 0.3366 |
| miami_dade | 140853 | 11909 | 0.5393 |
| middlesex | 74121 | 9222 | 0.3294 |
| cook | 278479 | 24993 | 0.4593 |

## Baseline 2025 metrics (rho = 0)

| jurisdiction | family | R2_price | NMSE | PRB | beta_log | PRD | MKI | VEI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| wayne | direct | 0.833 | 0.197 | -0.110 | -0.205 | 1.089 | 0.863 | -35.022 |
| wayne | surrogate | 0.832 | 0.197 | -0.111 | -0.206 | 1.090 | 0.863 | -34.304 |
| philadelphia | direct | 0.749 | 0.281 | -0.198 | -0.305 | 1.116 | 0.768 | -40.890 |
| philadelphia | surrogate | 0.747 | 0.280 | -0.202 | -0.308 | 1.119 | 0.761 | -41.927 |
| st_louis_county | direct | 0.812 | 0.151 | -0.053 | -0.136 | 1.097 | 0.902 | -28.865 |
| st_louis_county | surrogate | 0.814 | 0.151 | -0.053 | -0.137 | 1.097 | 0.901 | -28.324 |
| allegheny | direct | 0.717 | 0.235 | -0.113 | -0.224 | 1.109 | 0.859 | -39.871 |
| allegheny | surrogate | 0.716 | 0.235 | -0.115 | -0.226 | 1.110 | 0.858 | -39.911 |
| maricopa | direct | 0.297 | 0.160 | -0.060 | -0.176 | 1.094 | 0.835 | -14.787 |
| maricopa | surrogate | 0.298 | 0.160 | -0.059 | -0.176 | 1.095 | 0.834 | -14.579 |
| king | direct | 0.750 | 0.170 | -0.077 | -0.173 | 1.070 | 0.864 | -6.773 |
| king | surrogate | 0.752 | 0.171 | -0.077 | -0.173 | 1.071 | 0.864 | -7.623 |
| miami_dade | direct | 0.725 | 0.165 | -0.047 | -0.159 | 1.114 | 0.883 | -13.471 |
| miami_dade | surrogate | 0.718 | 0.165 | -0.047 | -0.159 | 1.114 | 0.884 | -13.362 |
| middlesex | direct | 0.253 | 0.209 | -0.106 | -0.204 | 1.093 | 0.844 | -6.840 |
| middlesex | surrogate | 0.261 | 0.207 | -0.107 | -0.204 | 1.093 | 0.844 | -7.731 |
| cook | direct | 0.830 | 0.170 | -0.109 | -0.184 | 1.088 | 0.887 | -27.756 |
| cook | surrogate | 0.831 | 0.170 | -0.109 | -0.184 | 1.088 | 0.887 | -28.486 |

## Jurisdiction × family findings

### Wayne County, MI (`wayne`)
**direct** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.205 PRB=-0.110
- activity ρ̃=0.3873 A_beta_2025=0.095 Delta_NMSE_2025=-0.000 (CV 0.001)
- guardrail ρ̃=0.8157 A_beta_2025=0.176 Delta_NMSE_2025=0.001
- 25% A_beta ρ̃=2.3250 A_beta_2025=0.332 Delta_NMSE=0.004
- 50% A_beta ρ̃=100.7700 A_beta_2025=0.663 Delta_NMSE=0.037

**surrogate** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.206 PRB=-0.111
- activity ρ̃=0.0873 A_beta_2025=0.052 Delta_NMSE_2025=0.003 (CV 0.005)
- guardrail ρ̃=0.3873 A_beta_2025=0.194 Delta_NMSE_2025=0.006
- 25% A_beta ρ̃=1.4541 A_beta_2025=0.396 Delta_NMSE=0.025

### Philadelphia County, PA (`philadelphia`)
**direct** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.305 PRB=-0.198
- activity ρ̃=0.2669 A_beta_2025=0.057 Delta_NMSE_2025=-0.003 (CV 0.002)
- guardrail ρ̃=0.8157 A_beta_2025=0.155 Delta_NMSE_2025=-0.002
- 25% A_beta ρ̃=1.6822 A_beta_2025=0.252 Delta_NMSE=0.005
- 50% A_beta ρ̃=10.9438 A_beta_2025=0.520 Delta_NMSE=0.035

**surrogate** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.308 PRB=-0.202
- activity ρ̃=0.0873 A_beta_2025=0.051 Delta_NMSE_2025=0.002 (CV 0.007)
- guardrail ρ̃=0.2669 A_beta_2025=0.127 Delta_NMSE_2025=0.004
- 25% A_beta ρ̃=0.8510 A_beta_2025=0.224 Delta_NMSE=0.016
- 50% A_beta ρ̃=21.1738 A_beta_2025=0.483 Delta_NMSE=0.095

### St. Louis County, MO (`st_louis_county`)
**direct** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.136 PRB=-0.053
- activity ρ̃=0.2669 A_beta_2025=0.071 Delta_NMSE_2025=0.000 (CV 0.000)
- guardrail ρ̃=23.2984 A_beta_2025=0.727 Delta_NMSE_2025=0.013
- 25% A_beta ρ̃=2.2096 A_beta_2025=0.388 Delta_NMSE=0.004
- 50% A_beta ρ̃=104.2798 A_beta_2025=0.787 Delta_NMSE=0.023

**surrogate** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.137 PRB=-0.053
- activity ρ̃=0.0873 A_beta_2025=0.089 Delta_NMSE_2025=0.002 (CV 0.001)
- guardrail ρ̃=0.3873 A_beta_2025=0.193 Delta_NMSE_2025=0.014
- 25% A_beta ρ̃=0.7806 A_beta_2025=0.338 Delta_NMSE=0.024
- 50% A_beta ρ̃=9.0756 A_beta_2025=0.643 Delta_NMSE=0.057

### Allegheny County, PA (`allegheny`)
**direct** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.224 PRB=-0.113
- activity ρ̃=0.2669 A_beta_2025=0.073 Delta_NMSE_2025=0.001 (CV 0.001)
- guardrail ρ̃=23.2984 A_beta_2025=0.682 Delta_NMSE_2025=0.040
- 25% A_beta ρ̃=1.5388 A_beta_2025=0.294 Delta_NMSE=0.008
- 50% A_beta ρ̃=10.9015 A_beta_2025=0.615 Delta_NMSE=0.031

**surrogate** — `FORWARD_SUPPORTS_CV`

- baseline 2025 beta_log=-0.226 PRB=-0.115
- activity ρ̃=0.1267 A_beta_2025=0.119 Delta_NMSE_2025=0.004 (CV 0.003)
- guardrail ρ̃=0.5621 A_beta_2025=0.290 Delta_NMSE_2025=0.021
- 25% A_beta ρ̃=0.9062 A_beta_2025=0.372 Delta_NMSE=0.033
- 50% A_beta ρ̃=7.3584 A_beta_2025=0.679 Delta_NMSE=0.110

### Maricopa County, AZ (`maricopa`)
**direct** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.176 PRB=-0.060
- activity ρ̃=0.2669 A_beta_2025=0.045 Delta_NMSE_2025=0.000 (CV -0.001)
- guardrail ρ̃=0.8157 A_beta_2025=0.113 Delta_NMSE_2025=0.001
- 25% A_beta ρ̃=1.5739 A_beta_2025=0.176 Delta_NMSE=0.003
- 50% A_beta ρ̃=31.5009 A_beta_2025=0.375 Delta_NMSE=0.013

**surrogate** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.176 PRB=-0.059
- activity ρ̃=0.0873 A_beta_2025=0.082 Delta_NMSE_2025=0.001 (CV 0.002)
- guardrail ρ̃=0.2669 A_beta_2025=0.132 Delta_NMSE_2025=0.002
- 25% A_beta ρ̃=0.7887 A_beta_2025=0.194 Delta_NMSE=0.008
- 50% A_beta ρ̃=7.3951 A_beta_2025=0.387 Delta_NMSE=0.034

### King County, WA (`king`)
**direct** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.173 PRB=-0.077
- activity ρ̃=0.1839 A_beta_2025=0.055 Delta_NMSE_2025=0.001 (CV -0.001)
- guardrail ρ̃=23.2984 A_beta_2025=0.644 Delta_NMSE_2025=0.032
- 25% A_beta ρ̃=1.8930 A_beta_2025=0.328 Delta_NMSE=0.008
- 50% A_beta ρ̃=70.6302 A_beta_2025=0.692 Delta_NMSE=0.048

**surrogate** — `FORWARD_SUPPORTS_CV`

- baseline 2025 beta_log=-0.173 PRB=-0.077
- activity ρ̃=0.0873 A_beta_2025=0.112 Delta_NMSE_2025=0.008 (CV 0.004)
- guardrail ρ̃=0.2669 A_beta_2025=0.196 Delta_NMSE_2025=0.017
- 25% A_beta ρ̃=1.0233 A_beta_2025=0.365 Delta_NMSE=0.035
- 50% A_beta ρ̃=10.1582 A_beta_2025=0.598 Delta_NMSE=0.091

### Miami-Dade County, FL (`miami_dade`)
**direct** — `FORWARD_SUPPORTS_CV`

- baseline 2025 beta_log=-0.159 PRB=-0.047
- activity ρ̃=0.3873 A_beta_2025=0.102 Delta_NMSE_2025=0.001 (CV -0.000)
- guardrail ρ̃=23.2984 A_beta_2025=0.616 Delta_NMSE_2025=0.032
- 25% A_beta ρ̃=1.0527 A_beta_2025=0.203 Delta_NMSE=0.004
- 50% A_beta ρ̃=4.6455 A_beta_2025=0.456 Delta_NMSE=0.014

**surrogate** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.159 PRB=-0.047
- activity ρ̃=0.0602 A_beta_2025=0.083 Delta_NMSE_2025=0.000 (CV 0.002)
- guardrail ρ̃=0.1267 A_beta_2025=0.141 Delta_NMSE_2025=0.002
- 25% A_beta ρ̃=0.2027 A_beta_2025=0.176 Delta_NMSE=0.003
- 50% A_beta ρ̃=1.0802 A_beta_2025=0.380 Delta_NMSE=0.021

### Middlesex County, MA (`middlesex`)
**direct** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.204 PRB=-0.106
- activity ρ̃=0.3873 A_beta_2025=0.084 Delta_NMSE_2025=-0.001 (CV 0.001)
- guardrail ρ̃=11.0617 A_beta_2025=0.595 Delta_NMSE_2025=0.022
- 25% A_beta ρ̃=3.0982 A_beta_2025=0.400 Delta_NMSE=0.007
- 50% A_beta ρ̃=56.2696 A_beta_2025=0.647 Delta_NMSE=0.089

**surrogate** — `FORWARD_SUPPORTS_CV`

- baseline 2025 beta_log=-0.204 PRB=-0.107
- activity ρ̃=0.2669 A_beta_2025=0.191 Delta_NMSE_2025=0.023 (CV 0.020)
- guardrail ρ̃=0.0602 A_beta_2025=0.072 Delta_NMSE_2025=0.005
- 25% A_beta ρ̃=1.0400 A_beta_2025=0.412 Delta_NMSE=0.066
- 50% A_beta ρ̃=101.0648 A_beta_2025=0.968 Delta_NMSE=0.264

### Cook County, IL (`cook`)
**direct** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.184 PRB=-0.109
- activity ρ̃=0.1839 A_beta_2025=0.056 Delta_NMSE_2025=-0.001 (CV 0.000)
- guardrail ρ̃=0.5621 A_beta_2025=0.152 Delta_NMSE_2025=-0.001
- 25% A_beta ρ̃=0.8068 A_beta_2025=0.199 Delta_NMSE=-0.000
- 50% A_beta ρ̃=2.4868 A_beta_2025=0.399 Delta_NMSE=0.003

**surrogate** — `FORWARD_PARTIAL`

- baseline 2025 beta_log=-0.184 PRB=-0.109
- activity ρ̃=0.0602 A_beta_2025=0.079 Delta_NMSE_2025=0.000 (CV 0.002)
- guardrail ρ̃=0.3873 A_beta_2025=0.324 Delta_NMSE_2025=0.009
- 25% A_beta ρ̃=0.2152 A_beta_2025=0.226 Delta_NMSE=0.004
- 50% A_beta ρ̃=0.7476 A_beta_2025=0.468 Delta_NMSE=0.019

## Direct common-band forward test

Frozen interval `[0.387298, 0.562080]` across 8 protocol-valid Direct jurisdictions (Allegheny held out as sensitivity).
Both endpoints practically useful in 2025 for **8/8** jurisdictions.
The interval was **not** redefined from 2025.

## Surrogate forward test

Preserved CV conclusion: `NO_NONDEGENERATE_ALL_JURISDICTION_INTERSECTION`. No 2025 search for a replacement universal band.

## Bootstrap (paired monthly-block, 200 draws)

Significance is never used to move a frozen coordinate. Percentile intervals are for frozen anchors only.

| jurisdiction | family | role | Delta_NMSE mean [2.5, 97.5] | Delta_beta_log mean [2.5, 97.5] | excludes 0 (NMSE / beta) |
|---|---|---|---|---|---|
| wayne | direct | activity | -0.0000 [-0.0006, 0.0005] | 0.0196 [0.0187, 0.0204] | False / True |
| wayne | direct | guardrail | 0.0007 [-0.0003, 0.0016] | 0.0362 [0.0356, 0.0368] | False / True |
| wayne | direct | A_beta_0.25 | 0.0046 [0.0032, 0.0060] | 0.0679 [0.0667, 0.0692] | True / True |
| wayne | direct | A_beta_0.5 | 0.0372 [0.0339, 0.0407] | 0.1361 [0.1344, 0.1375] | True / True |
| wayne | direct | direct_common_hi | 0.0004 [-0.0006, 0.0015] | 0.0267 [0.0260, 0.0277] | False / True |
| wayne | surrogate | activity | 0.0028 [0.0018, 0.0039] | 0.0107 [0.0099, 0.0115] | True / True |
| wayne | surrogate | guardrail | 0.0066 [0.0042, 0.0090] | 0.0399 [0.0384, 0.0415] | True / True |
| wayne | surrogate | A_beta_0.25 | 0.0254 [0.0208, 0.0308] | 0.0814 [0.0794, 0.0832] | True / True |
| philadelphia | direct | activity | -0.0025 [-0.0049, -0.0001] | 0.0173 [0.0155, 0.0192] | True / True |
| philadelphia | direct | guardrail | -0.0021 [-0.0056, 0.0009] | 0.0471 [0.0439, 0.0493] | False / True |
| philadelphia | direct | A_beta_0.25 | 0.0052 [0.0005, 0.0095] | 0.0766 [0.0734, 0.0795] | True / True |
| philadelphia | direct | A_beta_0.5 | 0.0352 [0.0243, 0.0468] | 0.1581 [0.1505, 0.1647] | True / True |
| philadelphia | direct | direct_common_lo | -0.0034 [-0.0057, -0.0010] | 0.0271 [0.0250, 0.0288] | True / True |
| philadelphia | direct | direct_common_hi | -0.0026 [-0.0059, 0.0002] | 0.0371 [0.0348, 0.0396] | False / True |
| philadelphia | surrogate | activity | 0.0020 [-0.0015, 0.0050] | 0.0157 [0.0124, 0.0185] | False / True |
| philadelphia | surrogate | guardrail | 0.0041 [-0.0010, 0.0100] | 0.0391 [0.0349, 0.0422] | False / True |
| philadelphia | surrogate | A_beta_0.25 | 0.0162 [0.0068, 0.0279] | 0.0689 [0.0626, 0.0745] | True / True |
| philadelphia | surrogate | A_beta_0.5 | 0.0952 [0.0749, 0.1227] | 0.1488 [0.1393, 0.1576] | True / True |
| st_louis_county | direct | activity | 0.0005 [-0.0003, 0.0013] | 0.0097 [0.0090, 0.0102] | False / True |
| st_louis_county | direct | guardrail | 0.0131 [0.0102, 0.0161] | 0.0989 [0.0972, 0.1011] | True / True |
| st_louis_county | direct | A_beta_0.25 | 0.0044 [0.0030, 0.0060] | 0.0527 [0.0515, 0.0538] | True / True |
| st_louis_county | direct | A_beta_0.5 | 0.0231 [0.0202, 0.0262] | 0.1071 [0.1043, 0.1102] | True / True |
| st_louis_county | direct | direct_common_lo | -0.0002 [-0.0012, 0.0007] | 0.0140 [0.0134, 0.0145] | False / True |
| st_louis_county | direct | direct_common_hi | 0.0003 [-0.0008, 0.0015] | 0.0199 [0.0193, 0.0206] | False / True |
| st_louis_county | surrogate | activity | 0.0021 [0.0014, 0.0031] | 0.0121 [0.0104, 0.0135] | True / True |
| st_louis_county | surrogate | guardrail | 0.0139 [0.0117, 0.0160] | 0.0263 [0.0239, 0.0282] | True / True |
| st_louis_county | surrogate | A_beta_0.25 | 0.0242 [0.0216, 0.0268] | 0.0461 [0.0436, 0.0482] | True / True |
| st_louis_county | surrogate | A_beta_0.5 | 0.0567 [0.0521, 0.0614] | 0.0876 [0.0827, 0.0914] | True / True |
| allegheny | direct | activity | 0.0014 [0.0005, 0.0023] | 0.0162 [0.0149, 0.0176] | True / True |
| allegheny | direct | guardrail | 0.0400 [0.0376, 0.0428] | 0.1526 [0.1471, 0.1575] | True / True |
| allegheny | direct | A_beta_0.25 | 0.0076 [0.0056, 0.0096] | 0.0658 [0.0642, 0.0677] | True / True |
| allegheny | direct | A_beta_0.5 | 0.0310 [0.0284, 0.0332] | 0.1375 [0.1331, 0.1416] | True / True |
| allegheny | direct | direct_common_lo | 0.0014 [0.0004, 0.0024] | 0.0222 [0.0206, 0.0238] | True / True |
| allegheny | direct | direct_common_hi | 0.0029 [0.0014, 0.0046] | 0.0300 [0.0281, 0.0320] | True / True |
| allegheny | surrogate | activity | 0.0045 [0.0028, 0.0062] | 0.0268 [0.0248, 0.0290] | True / True |
| allegheny | surrogate | guardrail | 0.0212 [0.0162, 0.0265] | 0.0656 [0.0627, 0.0683] | True / True |
| allegheny | surrogate | A_beta_0.25 | 0.0328 [0.0265, 0.0405] | 0.0843 [0.0806, 0.0879] | True / True |
| allegheny | surrogate | A_beta_0.5 | 0.1100 [0.0953, 0.1273] | 0.1537 [0.1475, 0.1606] | True / True |
| maricopa | direct | activity | 0.0004 [-0.0000, 0.0008] | 0.0079 [0.0073, 0.0084] | False / True |
| maricopa | direct | guardrail | 0.0015 [0.0009, 0.0022] | 0.0200 [0.0191, 0.0208] | True / True |
| maricopa | direct | A_beta_0.25 | 0.0029 [0.0017, 0.0042] | 0.0309 [0.0300, 0.0321] | True / True |
| maricopa | direct | A_beta_0.5 | 0.0134 [0.0111, 0.0160] | 0.0661 [0.0648, 0.0671] | True / True |
| maricopa | direct | direct_common_lo | 0.0008 [0.0004, 0.0013] | 0.0111 [0.0105, 0.0116] | True / True |
| maricopa | direct | direct_common_hi | 0.0006 [0.0001, 0.0011] | 0.0154 [0.0147, 0.0161] | True / True |
| maricopa | surrogate | activity | 0.0011 [-0.0000, 0.0023] | 0.0145 [0.0129, 0.0163] | False / True |
| maricopa | surrogate | guardrail | 0.0025 [0.0008, 0.0042] | 0.0234 [0.0218, 0.0249] | True / True |
| maricopa | surrogate | A_beta_0.25 | 0.0079 [0.0062, 0.0098] | 0.0342 [0.0320, 0.0362] | True / True |
| maricopa | surrogate | A_beta_0.5 | 0.0338 [0.0307, 0.0367] | 0.0685 [0.0662, 0.0708] | True / True |
| king | direct | activity | 0.0006 [-0.0002, 0.0013] | 0.0095 [0.0083, 0.0108] | False / True |
| king | direct | guardrail | 0.0319 [0.0276, 0.0361] | 0.1113 [0.1096, 0.1130] | True / True |
| king | direct | A_beta_0.25 | 0.0078 [0.0061, 0.0093] | 0.0568 [0.0555, 0.0578] | True / True |
| king | direct | A_beta_0.5 | 0.0483 [0.0430, 0.0531] | 0.1195 [0.1168, 0.1226] | True / True |
| king | direct | direct_common_lo | 0.0018 [0.0003, 0.0028] | 0.0174 [0.0165, 0.0182] | True / True |
| king | direct | direct_common_hi | 0.0016 [0.0005, 0.0025] | 0.0244 [0.0233, 0.0254] | True / True |
| king | surrogate | activity | 0.0083 [0.0049, 0.0127] | 0.0194 [0.0173, 0.0220] | True / True |
| king | surrogate | guardrail | 0.0166 [0.0126, 0.0217] | 0.0339 [0.0311, 0.0371] | True / True |
| king | surrogate | A_beta_0.25 | 0.0348 [0.0305, 0.0382] | 0.0631 [0.0578, 0.0693] | True / True |
| king | surrogate | A_beta_0.5 | 0.0907 [0.0847, 0.0980] | 0.1034 [0.0968, 0.1102] | True / True |
| miami_dade | direct | activity | 0.0012 [0.0003, 0.0022] | 0.0163 [0.0148, 0.0175] | True / True |
| miami_dade | direct | guardrail | 0.0319 [0.0269, 0.0368] | 0.0978 [0.0944, 0.1010] | True / True |
| miami_dade | direct | A_beta_0.25 | 0.0043 [0.0023, 0.0058] | 0.0324 [0.0309, 0.0336] | True / True |
| miami_dade | direct | A_beta_0.5 | 0.0143 [0.0107, 0.0175] | 0.0724 [0.0707, 0.0738] | True / True |
| miami_dade | direct | direct_common_hi | 0.0022 [0.0008, 0.0033] | 0.0215 [0.0205, 0.0226] | True / True |
| miami_dade | surrogate | activity | 0.0002 [-0.0011, 0.0014] | 0.0132 [0.0113, 0.0150] | False / True |
| miami_dade | surrogate | guardrail | 0.0017 [-0.0006, 0.0038] | 0.0225 [0.0203, 0.0255] | False / True |
| miami_dade | surrogate | A_beta_0.25 | 0.0032 [0.0006, 0.0053] | 0.0279 [0.0255, 0.0304] | True / True |
| miami_dade | surrogate | A_beta_0.5 | 0.0206 [0.0169, 0.0242] | 0.0603 [0.0561, 0.0651] | True / True |
| middlesex | direct | activity | -0.0011 [-0.0024, 0.0004] | 0.0173 [0.0153, 0.0190] | False / True |
| middlesex | direct | guardrail | 0.0226 [0.0178, 0.0278] | 0.1214 [0.1147, 0.1263] | True / True |
| middlesex | direct | A_beta_0.25 | 0.0070 [0.0033, 0.0104] | 0.0816 [0.0788, 0.0846] | True / True |
| middlesex | direct | A_beta_0.5 | 0.0892 [0.0804, 0.0977] | 0.1321 [0.1261, 0.1380] | True / True |
| middlesex | direct | direct_common_hi | -0.0013 [-0.0030, 0.0002] | 0.0290 [0.0274, 0.0303] | False / True |
| middlesex | surrogate | activity | 0.0229 [0.0187, 0.0267] | 0.0391 [0.0346, 0.0438] | True / True |
| middlesex | surrogate | guardrail | 0.0047 [0.0023, 0.0068] | 0.0148 [0.0115, 0.0180] | True / True |
| middlesex | surrogate | A_beta_0.25 | 0.0654 [0.0577, 0.0743] | 0.0838 [0.0774, 0.0897] | True / True |
| middlesex | surrogate | A_beta_0.5 | 0.2629 [0.2457, 0.2797] | 0.1973 [0.1855, 0.2090] | True / True |
| cook | direct | activity | -0.0006 [-0.0011, -0.0001] | 0.0103 [0.0098, 0.0109] | True / True |
| cook | direct | guardrail | -0.0009 [-0.0012, -0.0005] | 0.0279 [0.0271, 0.0287] | True / True |
| cook | direct | A_beta_0.25 | -0.0004 [-0.0008, -0.0001] | 0.0366 [0.0359, 0.0373] | True / True |
| cook | direct | A_beta_0.5 | 0.0028 [0.0020, 0.0037] | 0.0732 [0.0718, 0.0746] | True / True |
| cook | direct | direct_common_lo | -0.0006 [-0.0012, -0.0001] | 0.0201 [0.0192, 0.0207] | True / True |
| cook | surrogate | activity | 0.0004 [0.0000, 0.0008] | 0.0145 [0.0137, 0.0154] | True / True |
| cook | surrogate | guardrail | 0.0088 [0.0073, 0.0102] | 0.0597 [0.0575, 0.0624] | True / True |
| cook | surrogate | A_beta_0.25 | 0.0041 [0.0030, 0.0050] | 0.0415 [0.0397, 0.0431] | True / True |
| cook | surrogate | A_beta_0.5 | 0.0194 [0.0171, 0.0215] | 0.0861 [0.0830, 0.0901] | True / True |

## Answers to the predeclared scientific questions

1. Are all nine 2025 baseline AVMs still vertically regressive? **18/18** family-jurisdiction baselines have beta_log < -0.02: wayne/direct, wayne/surrogate, philadelphia/direct, philadelphia/surrogate, st_louis_county/direct, st_louis_county/surrogate, allegheny/direct, allegheny/surrogate, maricopa/direct, maricopa/surrogate, king/direct, king/surrogate, miami_dade/direct, miami_dade/surrogate, middlesex/direct, middlesex/surrogate, cook/direct, cook/surrogate.
2. Does normalized activity onset transfer? See activity rows above; judged per jurisdiction rather than forced.
3. Lack of portable upper guardrails: Surrogate remains without a nondegenerate all-jurisdiction intersection; Direct guardrails still span a wide range. Not redefined.
4. Frozen Direct common interval transfer: 8/8 protocol-valid jurisdictions keep both endpoints practically useful.
5. Direct vs Surrogate frontier: see `figures/paper/accuracy_mechanism_frontier_cv_vs_2025.pdf`.
6. Surrogate early limitations: Middlesex CV ordering and Miami-Dade early upper bound are tested, not rewritten.
7. 25%/50% A_beta: see `mechanism_anchor_forward.csv`.
8. Temporal stability: see `cv_to_2025_path_drift.pdf` and anchor 2025-minus-CV columns.
9. Ratio profiles: see `figures/ratio_profiles/` and `forward_ratio_profile_examples.pdf`.
10. Berry/local vs AVM: official-assessment and AVM ratios remain separate constructs; see `berry_local_vs_avm_ratio_profiles.pdf`. Wayne is not Detroit.
11. 2025 vs CV portability claims: SUPPORTS=4 PARTIAL=14 WEAKENS=0 REVERSAL=0 of 18 family-jurisdiction pairs. Numbers in the tables dominate the labels.

No 2025-derived candidate region was written.
