# Figures

Place real figure assets here. The generator looks for these paths first and
falls back to native-drawn equivalents if they are absent.

## Expected files

### `baseline_regressivity_plot.png`

- Consumed by slide 9 (`annotated_chart`).
- Source: paper Figure `fig:regressivity_motivation`
  (`Fairness Price/CCAO meeting/img/Baseline LightGBM_0_motivation.pdf` in
  the paper's asset tree).
- Content: ratio `r = P̂ / P` on the y-axis vs `log(sale price)` on the
  x-axis, baseline LightGBM on the 2023 CCAO test set, with a horizontal
  `r = 1` reference line and a LOWESS-style trend.
- Recommended export: PNG at 1600×1000 px, white background, 2× device
  scale for crisp embedding.

If this file is missing, `generate_deck.js` renders a native PPT scatter
plot with schematic (representative, not actual) data so the slide still
lands the regressivity pattern. The schematic fallback is flagged in the
slide's speaker notes when used.
