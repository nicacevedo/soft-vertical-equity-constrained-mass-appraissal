// generate_deck.js
// Reads deck.json and renders a .pptx file to the project root using PptxGenJS.
// Supports exactly 8 layout types. Any other layout_type is a hard error.

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

// -----------------------------------------------------------------------------
// Paths and constants
// -----------------------------------------------------------------------------

const PROJECT_ROOT = path.resolve(__dirname, "..");
const DECK_JSON_PATH = path.join(PROJECT_ROOT, "deck.json");
const OUTPUT_PATH = path.join(PROJECT_ROOT, "preliminaries.pptx");
const ASSETS_DIR = path.join(PROJECT_ROOT, "assets");

const SUPPORTED_LAYOUTS = [
  "title_slide",
  "opener_infographic",
  "process_flow",
  "concept_diagram",
  "comparison_table",
  "map_plus_timeline",
  "split_panel",
  "annotated_chart",
];

// Slide geometry (inches). LAYOUT_WIDE => 13.333 x 7.5.
const SLIDE = { w: 13.333, h: 7.5 };
const MARGIN = { x: 0.5, top: 0.4 };
const TITLE = { y: 0.35, h: 0.9 };
const BODY = { y: 1.45, h: 5.4 };
const FOOTER = { y: 7.0, h: 0.35 };

const COLOR = {
  text: "1A1A1A",
  muted: "5A5A5A",
  accent: "1F4E79",
  accentSoft: "D9E4F1",
  good: "2E7D32",
  bad: "C62828",
  neutralFill: "F2F4F7",
  border: "B8BEC7",
  highlightFill: "FFF4D6",
};

const FONT = { face: "Calibri" };

// -----------------------------------------------------------------------------
// Shared helpers
// -----------------------------------------------------------------------------

function addTitle(slide, text) {
  slide.addText(sanitizeText(text) || "", {
    x: MARGIN.x,
    y: TITLE.y,
    w: SLIDE.w - 2 * MARGIN.x,
    h: TITLE.h,
    fontFace: FONT.face,
    fontSize: 26,
    bold: true,
    color: COLOR.text,
    valign: "top",
  });
}

function addFooter(slide, slideNumber, totalSlides) {
  slide.addText(`Preliminaries · ${slideNumber} / ${totalSlides}`, {
    x: MARGIN.x,
    y: FOOTER.y,
    w: SLIDE.w - 2 * MARGIN.x,
    h: FOOTER.h,
    fontFace: FONT.face,
    fontSize: 10,
    color: COLOR.muted,
    align: "right",
  });
}

function addSpeakerNotes(slide, slideSpec) {
  const parts = [];
  if (slideSpec.notes) parts.push(sanitizeText(slideSpec.notes));
  const refs = slideSpec.source_refs || [];
  if (refs.length > 0) {
    const refLines = refs.map(
      (r) => `- ${r.file || ""}${r.locator ? " :: " + r.locator : ""}${r.cite ? " [" + r.cite + "]" : ""}`,
    );
    parts.push("Sources:\n" + refLines.join("\n"));
  }
  if (parts.length > 0) slide.addNotes(parts.join("\n\n"));
}

// Strip common LaTeX tokens out of text that will be rendered as plain PPT text.
// This is the global guard against raw-LaTeX leakage on slides.
const LATEX_REPLACEMENTS = [
  [/\\widehat\s*\{?\s*P\s*\}?/g, "P̂"],
  [/\\widehat\s*\{?\s*Y\s*\}?/g, "Ŷ"],
  [/\\hat\s*\{?\s*P\s*\}?/g, "P̂"],
  [/\\hat\s*\{?\s*Y\s*\}?/g, "Ŷ"],
  [/\\approx\b/g, "≈"],
  [/\\times\b/g, "×"],
  [/\\cdot\b/g, "·"],
  [/\\leq\b/g, "≤"],
  [/\\geq\b/g, "≥"],
  [/\\to\b/g, "→"],
  [/\\rightarrow\b/g, "→"],
  [/\\alpha\b/g, "α"],
  [/\\beta\b/g, "β"],
  [/\\rho\b/g, "ρ"],
  [/\\text\s*\{([^{}]*)\}/g, "$1"],
  [/\\mathrm\s*\{([^{}]*)\}/g, "$1"],
  [/\\mathbf\s*\{([^{}]*)\}/g, "$1"],
];

function sanitizeText(input) {
  if (input == null) return input;
  if (typeof input !== "string") return input;
  let out = input;
  for (const [re, rep] of LATEX_REPLACEMENTS) out = out.replace(re, rep);
  // Strip surviving leading backslashes on tokens and stray braces.
  out = out.replace(/\\([A-Za-z]+)/g, "$1");
  out = out.replace(/[{}]/g, "");
  // Collapse runs of whitespace introduced by stripping.
  out = out.replace(/\s{2,}/g, " ").trim();
  return out;
}

function textBlocksByRole(slideSpec, role) {
  return (slideSpec.text_blocks || [])
    .filter((b) => b.role === role)
    .map((b) => sanitizeText(b.text));
}

function firstText(slideSpec, role) {
  const items = textBlocksByRole(slideSpec, role);
  return items.length > 0 ? items[0] : null;
}

// -----------------------------------------------------------------------------
// Layout: title_slide
// -----------------------------------------------------------------------------

function renderTitleSlide(slide, slideSpec) {
  // This layout manages its own title area; skip the default addTitle band.
  const v = slideSpec.visual || {};
  const authors = v.authors || [];
  const totalW = SLIDE.w - 2 * MARGIN.x;

  slide.addShape("rect", {
    x: 0,
    y: 0,
    w: SLIDE.w,
    h: SLIDE.h,
    fill: { color: "FFFFFF" },
    line: { color: "FFFFFF", width: 0 },
  });

  // Thin accent band at the top.
  slide.addShape("rect", {
    x: 0,
    y: 0,
    w: SLIDE.w,
    h: 0.15,
    fill: { color: COLOR.accent },
    line: { color: COLOR.accent, width: 0 },
  });

  // Main title.
  slide.addText(sanitizeText(slideSpec.title) || "", {
    x: MARGIN.x,
    y: SLIDE.h * 0.32,
    w: totalW,
    h: 1.4,
    fontFace: FONT.face,
    fontSize: 36,
    bold: true,
    color: COLOR.text,
    align: "center",
    valign: "middle",
  });

  const subtitle = firstText(slideSpec, "subtitle");
  if (subtitle) {
    slide.addText(subtitle, {
      x: MARGIN.x,
      y: SLIDE.h * 0.32 + 1.3,
      w: totalW,
      h: 0.6,
      fontFace: FONT.face,
      fontSize: 20,
      italic: true,
      color: COLOR.accent,
      align: "center",
    });
  }

  if (authors.length > 0) {
    const blockY = SLIDE.h * 0.62;
    const rowH = 0.55;
    authors.forEach((a, i) => {
      slide.addText(
        [
          { text: sanitizeText(a.name || ""), options: { bold: true, fontSize: 16, color: COLOR.text } },
          { text: "  —  ", options: { fontSize: 16, color: COLOR.muted } },
          { text: sanitizeText(a.affiliation || ""), options: { fontSize: 14, italic: true, color: COLOR.muted } },
        ],
        {
          x: MARGIN.x,
          y: blockY + i * rowH,
          w: totalW,
          h: rowH,
          fontFace: FONT.face,
          align: "center",
          valign: "middle",
        },
      );
    });
  }
}

// -----------------------------------------------------------------------------
// Layout: opener_infographic
// -----------------------------------------------------------------------------

function renderOpenerInfographic(slide, slideSpec) {
  const v = slideSpec.visual || {};
  const stats = v.stats || [];

  const leftW = (SLIDE.w - 2 * MARGIN.x) * 0.55;
  const rightX = MARGIN.x + leftW + 0.3;
  const rightW = SLIDE.w - rightX - MARGIN.x;

  const rowH = Math.min(1.4, (BODY.h - 0.4) / Math.max(stats.length, 1));
  stats.forEach((s, i) => {
    const y = BODY.y + i * rowH;
    slide.addText(
      [
        { text: s.value || "", options: { bold: true, fontSize: 40, color: COLOR.accent } },
        { text: "  " + (s.label || ""), options: { fontSize: 16, color: COLOR.text } },
      ],
      { x: MARGIN.x, y, w: leftW, h: rowH, fontFace: FONT.face, valign: "middle" },
    );
  });

  const iconRow = (v.icon_row && v.icon_row.sequence) || [];
  if (iconRow.length > 0) {
    const cellW = rightW / iconRow.length;
    const iconY = BODY.y + BODY.h * 0.3;
    const iconH = BODY.h * 0.4;
    iconRow.forEach((item, i) => {
      const x = rightX + i * cellW;
      const isArrow = /arrow/i.test(item.icon || "");
      slide.addShape(isArrow ? "rightArrow" : "roundRect", {
        x: x + cellW * 0.15,
        y: iconY,
        w: cellW * 0.7,
        h: iconH,
        fill: { color: isArrow ? COLOR.accentSoft : COLOR.neutralFill },
        line: { color: COLOR.border, width: 0.75 },
      });
      if (!isArrow) {
        slide.addText(item.label || item.icon || "", {
          x,
          y: iconY + iconH + 0.05,
          w: cellW,
          h: 0.35,
          fontFace: FONT.face,
          fontSize: 10,
          color: COLOR.muted,
          align: "center",
        });
      }
    });
  }

  const subtitle = firstText(slideSpec, "subtitle");
  if (subtitle) {
    slide.addText(subtitle, {
      x: MARGIN.x,
      y: BODY.y + BODY.h - 0.7,
      w: SLIDE.w - 2 * MARGIN.x,
      h: 0.55,
      fontFace: FONT.face,
      fontSize: 16,
      italic: true,
      color: COLOR.muted,
    });
  }
}

// -----------------------------------------------------------------------------
// Layout: process_flow
// -----------------------------------------------------------------------------

function renderProcessFlow(slide, slideSpec) {
  const v = slideSpec.visual || {};
  const steps = v.steps || [];
  if (steps.length === 0) return;

  const totalW = SLIDE.w - 2 * MARGIN.x;
  const gap = 0.2;
  const arrowW = 0.4;
  const availableW = totalW - (steps.length - 1) * (gap + arrowW);
  const boxW = availableW / steps.length;
  const boxH = 1.6;
  const boxY = BODY.y + 0.6;

  steps.forEach((step, i) => {
    const x = MARGIN.x + i * (boxW + gap + arrowW);
    const isHighlight = !!step.highlight;
    slide.addShape("roundRect", {
      x,
      y: boxY,
      w: boxW,
      h: boxH,
      fill: { color: isHighlight ? COLOR.highlightFill : COLOR.neutralFill },
      line: { color: isHighlight ? COLOR.accent : COLOR.border, width: isHighlight ? 2 : 1 },
    });
    slide.addText(String(step.index ?? i + 1), {
      x,
      y: boxY + 0.1,
      w: boxW,
      h: 0.35,
      fontFace: FONT.face,
      fontSize: 12,
      bold: true,
      color: COLOR.accent,
      align: "center",
    });
    slide.addText(step.label || "", {
      x: x + 0.1,
      y: boxY + 0.5,
      w: boxW - 0.2,
      h: boxH - 0.6,
      fontFace: FONT.face,
      fontSize: 13,
      color: COLOR.text,
      align: "center",
      valign: "middle",
    });
    if (i < steps.length - 1) {
      slide.addShape("rightArrow", {
        x: x + boxW + gap / 2,
        y: boxY + boxH / 2 - 0.2,
        w: arrowW,
        h: 0.4,
        fill: { color: COLOR.accentSoft },
        line: { color: COLOR.border, width: 0.5 },
      });
    }
  });

  const formula = firstText(slideSpec, "formula");
  if (formula) {
    slide.addText(formula, {
      x: MARGIN.x,
      y: BODY.y,
      w: totalW,
      h: 0.5,
      fontFace: FONT.face,
      fontSize: 16,
      italic: true,
      color: COLOR.accent,
      align: "center",
    });
  }

  const caption = firstText(slideSpec, "caption");
  if (caption) {
    slide.addText(caption, {
      x: MARGIN.x,
      y: boxY + boxH + 0.4,
      w: totalW,
      h: 0.6,
      fontFace: FONT.face,
      fontSize: 13,
      color: COLOR.muted,
      align: "center",
    });
  }
}

// -----------------------------------------------------------------------------
// Layout: concept_diagram
// -----------------------------------------------------------------------------

function renderConceptDiagram(slide, slideSpec) {
  const v = slideSpec.visual || {};
  const totalW = SLIDE.w - 2 * MARGIN.x;

  const gridY = BODY.y + 0.1;
  const gridH = BODY.h * 0.65;
  const gridW = Math.min(totalW, gridH * 1.6);
  const gridX = MARGIN.x + (totalW - gridW) / 2;

  const grid = v.grid || {};
  const totalCells = grid.total_cells_hint || 400;
  const highlightedFraction = grid.highlighted_fraction ?? 0.05;
  const cols = 25;
  const rows = Math.ceil(totalCells / cols);
  const cellW = gridW / cols;
  const cellH = gridH / rows;
  const nHighlighted = Math.max(1, Math.round(cols * rows * highlightedFraction));

  // Deterministic pseudo-random highlight placement.
  const highlighted = new Set();
  let seed = 7;
  while (highlighted.size < nHighlighted) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    highlighted.add(seed % (cols * rows));
  }

  for (let i = 0; i < cols * rows; i++) {
    const r = Math.floor(i / cols);
    const c = i % cols;
    const isH = highlighted.has(i);
    slide.addShape("rect", {
      x: gridX + c * cellW,
      y: gridY + r * cellH,
      w: cellW - 0.02,
      h: cellH - 0.02,
      fill: { color: isH ? COLOR.accent : COLOR.neutralFill },
      line: { color: COLOR.border, width: 0.25 },
    });
  }

  // Legend under grid.
  const legendY = gridY + gridH + 0.2;
  slide.addShape("rect", {
    x: gridX,
    y: legendY,
    w: 0.25,
    h: 0.25,
    fill: { color: COLOR.accent },
    line: { color: COLOR.border, width: 0.25 },
  });
  slide.addText(grid.highlighted_label || "observed", {
    x: gridX + 0.3,
    y: legendY - 0.05,
    w: 3,
    h: 0.35,
    fontFace: FONT.face,
    fontSize: 11,
    color: COLOR.text,
  });
  slide.addShape("rect", {
    x: gridX + 3.4,
    y: legendY,
    w: 0.25,
    h: 0.25,
    fill: { color: COLOR.neutralFill },
    line: { color: COLOR.border, width: 0.25 },
  });
  slide.addText(grid.unhighlighted_label || "unobserved", {
    x: gridX + 3.7,
    y: legendY - 0.05,
    w: 4,
    h: 0.35,
    fontFace: FONT.face,
    fontSize: 11,
    color: COLOR.text,
  });

  // Feature tag row.
  const tags = v.feature_tag_row || [];
  if (tags.length > 0) {
    const tagsY = legendY + 0.55;
    const tagW = Math.min(1.6, totalW / tags.length - 0.1);
    const tagsTotalW = tagW * tags.length + 0.1 * (tags.length - 1);
    const tagsStartX = MARGIN.x + (totalW - tagsTotalW) / 2;
    tags.forEach((t, i) => {
      slide.addShape("roundRect", {
        x: tagsStartX + i * (tagW + 0.1),
        y: tagsY,
        w: tagW,
        h: 0.4,
        fill: { color: COLOR.accentSoft },
        line: { color: COLOR.accent, width: 0.75 },
      });
      slide.addText(t, {
        x: tagsStartX + i * (tagW + 0.1),
        y: tagsY,
        w: tagW,
        h: 0.4,
        fontFace: FONT.face,
        fontSize: 11,
        color: COLOR.accent,
        align: "center",
        valign: "middle",
      });
    });
  }

  const def = firstText(slideSpec, "definition");
  if (def) {
    slide.addText(def, {
      x: MARGIN.x,
      y: BODY.y - 0.35,
      w: totalW,
      h: 0.5,
      fontFace: FONT.face,
      fontSize: 13,
      italic: true,
      color: COLOR.muted,
      align: "center",
    });
  }
}

// -----------------------------------------------------------------------------
// Layout: split_panel
// -----------------------------------------------------------------------------

// Draw a tiny house glyph centered at (cx, topY) with the given body width.
function drawHouse(slide, cx, topY, bodyW, color) {
  const bodyH = bodyW * 0.8;
  const roofH = bodyW * 0.45;
  const bodyX = cx - bodyW / 2;
  // Roof (triangle via freeform not available; use isoceles triangle shape).
  slide.addShape("triangle", {
    x: bodyX - bodyW * 0.1,
    y: topY,
    w: bodyW * 1.2,
    h: roofH,
    fill: { color: color },
    line: { color: COLOR.text, width: 0.5 },
  });
  // Body.
  slide.addShape("rect", {
    x: bodyX,
    y: topY + roofH,
    w: bodyW,
    h: bodyH,
    fill: { color: "FFFFFF" },
    line: { color: COLOR.text, width: 0.75 },
  });
  // Door.
  const doorW = bodyW * 0.28;
  const doorH = bodyH * 0.55;
  slide.addShape("rect", {
    x: cx - doorW / 2,
    y: topY + roofH + bodyH - doorH,
    w: doorW,
    h: doorH,
    fill: { color: color },
    line: { color: COLOR.text, width: 0.5 },
  });
}

// Draw a small ratio badge above a house. `color` is the outline/text color.
function drawRatioBadge(slide, cx, topY, text, color) {
  const w = 0.55;
  const h = 0.3;
  slide.addShape("roundRect", {
    x: cx - w / 2,
    y: topY,
    w,
    h,
    fill: { color: "FFFFFF" },
    line: { color, width: 1 },
  });
  slide.addText(text, {
    x: cx - w / 2,
    y: topY,
    w,
    h,
    fontFace: FONT.face,
    fontSize: 10,
    bold: true,
    color,
    align: "center",
    valign: "middle",
  });
}

// Colors for ratio badges: red if too high or too low vs 1, green if near 1.
function ratioColor(ratioStr) {
  const r = parseFloat(ratioStr);
  if (isFinite(r) && Math.abs(r - 1) <= 0.05) return COLOR.good;
  return COLOR.bad;
}

// Draw the horizontal-equity mini-sketch inside a panel.
function drawHorizontalEquityPanel(slide, panel, x, y, w, h) {
  const bad = panel.ratios_bad || ["0.82", "1.08", "1.21"];
  const good = panel.ratios_good || ["0.98", "1.01", "1.02"];
  const rowLabelH = 0.3;
  const rowH = (h - rowLabelH * 2) / 2;
  const bodyW = Math.min(rowH * 0.55, w / 6);

  function drawRow(ratios, rowTopY, label, labelColor) {
    slide.addText(label, {
      x: x + 0.1,
      y: rowTopY,
      w: w - 0.2,
      h: rowLabelH,
      fontFace: FONT.face,
      fontSize: 11,
      italic: true,
      color: labelColor,
    });
    const n = ratios.length;
    const cellW = (w - 0.4) / n;
    ratios.forEach((r, i) => {
      const cx = x + 0.2 + cellW * (i + 0.5);
      const houseTopY = rowTopY + rowLabelH + 0.35;
      drawRatioBadge(slide, cx, rowTopY + rowLabelH + 0.02, "r=" + r, ratioColor(r));
      drawHouse(slide, cx, houseTopY, bodyW, COLOR.accent);
    });
  }

  drawRow(bad, y, "Inconsistent (bad)", COLOR.bad);
  drawRow(good, y + rowLabelH + rowH, "Consistent (good)", COLOR.good);
}

// Draw the vertical-equity mini-sketch inside a panel.
function drawVerticalEquityPanel(slide, panel, x, y, w, h) {
  const tiers = panel.tiers || [
    { size: "small", ratio: "1.15" },
    { size: "med", ratio: "1.02" },
    { size: "large", ratio: "0.88" },
  ];
  const sizeMap = { small: 0.35, med: 0.55, large: 0.8 };
  const n = tiers.length;
  const cellW = (w - 0.4) / n;
  const rowTopY = y + h * 0.25;
  const houseMaxW = Math.min(1.0, cellW * 0.7);

  tiers.forEach((t, i) => {
    const cx = x + 0.2 + cellW * (i + 0.5);
    const scale = sizeMap[t.size] || 0.55;
    const bodyW = houseMaxW * scale;
    drawRatioBadge(slide, cx, rowTopY, "r=" + t.ratio, ratioColor(t.ratio));
    drawHouse(slide, cx, rowTopY + 0.35, bodyW, COLOR.accent);
  });

  // Value-axis arrow underneath.
  const arrowY = y + h - 0.55;
  slide.addShape("rightArrow", {
    x: x + 0.2,
    y: arrowY,
    w: w - 0.4,
    h: 0.25,
    fill: { color: COLOR.accentSoft },
    line: { color: COLOR.border, width: 0.5 },
  });
  slide.addText("property value →", {
    x: x + 0.2,
    y: arrowY + 0.25,
    w: w - 0.4,
    h: 0.25,
    fontFace: FONT.face,
    fontSize: 10,
    italic: true,
    color: COLOR.muted,
    align: "center",
  });
}

function renderSplitPanel(slide, slideSpec) {
  const v = slideSpec.visual || {};
  const totalW = SLIDE.w - 2 * MARGIN.x;
  const gap = 0.3;
  const panelW = (totalW - gap) / 2;
  const panelH = BODY.h * 0.75;
  const panelY = BODY.y;

  const left = v.left_panel || v.left || {};
  const right = v.right_panel || v.right || {};
  const isEquity = v.type === "equity_panels";

  [left, right].forEach((panel, i) => {
    const x = MARGIN.x + i * (panelW + gap);
    slide.addShape("roundRect", {
      x,
      y: panelY,
      w: panelW,
      h: panelH,
      fill: { color: COLOR.neutralFill },
      line: { color: COLOR.border, width: 1 },
    });
    slide.addText(sanitizeText(panel.title) || "", {
      x: x + 0.2,
      y: panelY + 0.2,
      w: panelW - 0.4,
      h: 0.5,
      fontFace: FONT.face,
      fontSize: 18,
      bold: true,
      color: COLOR.accent,
    });
    const sub = sanitizeText(panel.subtitle || panel.subtext || "");
    if (sub) {
      slide.addText(sub, {
        x: x + 0.2,
        y: panelY + 0.75,
        w: panelW - 0.4,
        h: 0.5,
        fontFace: FONT.face,
        fontSize: 12,
        italic: true,
        color: COLOR.muted,
      });
    }

    // Inner sketch area.
    const sketchX = x + 0.3;
    const sketchY = panelY + 1.35;
    const sketchW = panelW - 0.6;
    const sketchH = panelH - 2.0;

    if (isEquity && i === 0) {
      drawHorizontalEquityPanel(slide, panel, sketchX, sketchY, sketchW, sketchH);
    } else if (isEquity && i === 1) {
      drawVerticalEquityPanel(slide, panel, sketchX, sketchY, sketchW, sketchH);
    } else if (panel.sketch) {
      slide.addText("[ " + panel.sketch + " ]", {
        x: sketchX,
        y: sketchY,
        w: sketchW,
        h: sketchH,
        fontFace: FONT.face,
        fontSize: 12,
        color: COLOR.muted,
        align: "center",
        valign: "middle",
      });
    }

    if (panel.annotation) {
      slide.addText(sanitizeText(panel.annotation), {
        x: x + 0.2,
        y: panelY + panelH - 0.55,
        w: panelW - 0.4,
        h: 0.4,
        fontFace: FONT.face,
        fontSize: 12,
        bold: true,
        color: COLOR.bad,
        align: "center",
      });
    }
  });

  // Optional connector (e.g., "more flexible" arrow between two columns).
  const connector = v.connector;
  if (connector && connector.type === "arrow") {
    slide.addShape("rightArrow", {
      x: MARGIN.x + panelW - 0.2,
      y: panelY + panelH / 2 - 0.2,
      w: gap + 0.4,
      h: 0.4,
      fill: { color: COLOR.accent },
      line: { color: COLOR.accent, width: 0.5 },
    });
    if (connector.label) {
      slide.addText(connector.label, {
        x: MARGIN.x + panelW - 0.6,
        y: panelY + panelH / 2 + 0.3,
        w: gap + 1.2,
        h: 0.3,
        fontFace: FONT.face,
        fontSize: 10,
        italic: true,
        color: COLOR.accent,
        align: "center",
      });
    }
  }

  // Footer line (caption or formula shared under both panels).
  const footer = v.footer_line || firstText(slideSpec, "caption") || firstText(slideSpec, "formula");
  if (footer) {
    slide.addText(footer, {
      x: MARGIN.x,
      y: panelY + panelH + 0.25,
      w: totalW,
      h: 0.4,
      fontFace: FONT.face,
      fontSize: 13,
      italic: true,
      color: COLOR.muted,
      align: "center",
    });
  }
}

// -----------------------------------------------------------------------------
// Layout: map_plus_timeline
// -----------------------------------------------------------------------------

function renderMapPlusTimeline(slide, slideSpec) {
  const v = slideSpec.visual || {};
  const totalW = SLIDE.w - 2 * MARGIN.x;

  const mapW = totalW * 0.32;
  const mapH = BODY.h * 0.55;
  const mapX = MARGIN.x;
  const mapY = BODY.y;

  slide.addShape("roundRect", {
    x: mapX,
    y: mapY,
    w: mapW,
    h: mapH,
    fill: { color: COLOR.accentSoft },
    line: { color: COLOR.accent, width: 1 },
  });
  const mapLabel = (v.map_locator && v.map_locator.region) || "Map locator";
  slide.addText(mapLabel, {
    x: mapX,
    y: mapY + mapH / 2 - 0.25,
    w: mapW,
    h: 0.5,
    fontFace: FONT.face,
    fontSize: 14,
    bold: true,
    color: COLOR.accent,
    align: "center",
  });

  // Timeline to the right of the map.
  const tlX = mapX + mapW + 0.4;
  const tlW = totalW - mapW - 0.4;
  const tlY = mapY + 0.3;
  const tlH = 1.2;
  const blocks = (v.timeline && v.timeline.blocks) || [];
  if (blocks.length > 0) {
    const gap = 0.15;
    const blockW = (tlW - gap * (blocks.length - 1)) / blocks.length;
    blocks.forEach((b, i) => {
      const x = tlX + i * (blockW + gap);
      slide.addShape("roundRect", {
        x,
        y: tlY,
        w: blockW,
        h: tlH,
        fill: { color: COLOR.neutralFill },
        line: { color: COLOR.border, width: 1 },
      });
      slide.addText(b.span || "", {
        x,
        y: tlY + 0.1,
        w: blockW,
        h: 0.45,
        fontFace: FONT.face,
        fontSize: 14,
        bold: true,
        color: COLOR.accent,
        align: "center",
      });
      slide.addText(b.label || "", {
        x: x + 0.1,
        y: tlY + 0.55,
        w: blockW - 0.2,
        h: tlH - 0.6,
        fontFace: FONT.face,
        fontSize: 11,
        color: COLOR.text,
        align: "center",
        valign: "middle",
      });
    });
  }

  // Facts box spans the full width under map+timeline.
  const facts = v.facts_box || [];
  if (facts.length > 0) {
    const fbY = mapY + Math.max(mapH, tlY - mapY + tlH) + 0.3;
    const fbH = BODY.h - (fbY - BODY.y) - 0.2;
    slide.addShape("roundRect", {
      x: MARGIN.x,
      y: fbY,
      w: totalW,
      h: fbH,
      fill: { color: COLOR.neutralFill },
      line: { color: COLOR.border, width: 1 },
    });
    const colW = totalW / facts.length;
    facts.forEach((f, i) => {
      const x = MARGIN.x + i * colW;
      slide.addText((f.key || "").toUpperCase(), {
        x: x + 0.15,
        y: fbY + 0.15,
        w: colW - 0.3,
        h: 0.3,
        fontFace: FONT.face,
        fontSize: 10,
        bold: true,
        color: COLOR.muted,
      });
      slide.addText(f.value || "", {
        x: x + 0.15,
        y: fbY + 0.45,
        w: colW - 0.3,
        h: fbH - 0.5,
        fontFace: FONT.face,
        fontSize: 13,
        color: COLOR.text,
        valign: "top",
      });
    });
  }

  const caption = firstText(slideSpec, "caption");
  if (caption) {
    slide.addText(caption, {
      x: MARGIN.x,
      y: BODY.y + BODY.h - 0.05,
      w: totalW,
      h: 0.4,
      fontFace: FONT.face,
      fontSize: 12,
      italic: true,
      color: COLOR.muted,
      align: "center",
    });
  }
}

// -----------------------------------------------------------------------------
// Layout: comparison_table
// -----------------------------------------------------------------------------

function renderComparisonTable(slide, slideSpec) {
  const v = slideSpec.visual || {};
  const columns = v.columns || [];
  const rows = v.rows || [];
  if (columns.length === 0 || rows.length === 0) return;

  const totalW = SLIDE.w - 2 * MARGIN.x;
  const modelColW = 2.2;
  const metricColW = (totalW - modelColW) / columns.length;

  // Subtitle band directly under the title — the visual "hinge" line.
  const subtitle = firstText(slideSpec, "subtitle");
  let headerOffset = 0;
  if (subtitle) {
    slide.addShape("rect", {
      x: MARGIN.x,
      y: BODY.y - 0.3,
      w: totalW,
      h: 0.45,
      fill: { color: COLOR.highlightFill },
      line: { color: COLOR.accent, width: 0.75 },
    });
    slide.addText(subtitle, {
      x: MARGIN.x,
      y: BODY.y - 0.3,
      w: totalW,
      h: 0.45,
      fontFace: FONT.face,
      fontSize: 15,
      bold: true,
      color: COLOR.accent,
      align: "center",
      valign: "middle",
    });
    headerOffset = 0.35;
  }

  if (v.dataset_label) {
    slide.addText(v.dataset_label, {
      x: MARGIN.x,
      y: BODY.y + headerOffset - 0.1,
      w: totalW,
      h: 0.35,
      fontFace: FONT.face,
      fontSize: 12,
      italic: true,
      color: COLOR.muted,
    });
  }

  const improved = new Set(v.delta_coloring?.improved || []);
  const worsened = new Set(v.delta_coloring?.worsened || []);

  // Build tbody.
  const headerRow = [
    { text: "Model", options: { bold: true, color: COLOR.text, fill: { color: COLOR.accentSoft } } },
    ...columns.map((c) => ({
      text: c.label || c.key,
      options: { bold: true, color: COLOR.text, fill: { color: COLOR.accentSoft }, align: "center" },
    })),
  ];

  const dataRows = rows.map((r, rowIdx) => {
    const cells = [{ text: r.model || "", options: { bold: true, color: COLOR.text } }];
    columns.forEach((c) => {
      const raw = (r.values || {})[c.key];
      let cellColor = COLOR.text;
      // Only color the "second" row (the one being compared against the first).
      if (rowIdx > 0) {
        if (improved.has(c.key)) cellColor = COLOR.good;
        else if (worsened.has(c.key)) cellColor = COLOR.bad;
      }
      cells.push({
        text: raw == null ? "" : String(raw),
        options: { color: cellColor, align: "center", bold: rowIdx > 0 && (improved.has(c.key) || worsened.has(c.key)) },
      });
    });
    return cells;
  });

  const tableRows = [headerRow, ...dataRows];
  const tableY = BODY.y + 0.35 + headerOffset;
  slide.addTable(tableRows, {
    x: MARGIN.x,
    y: tableY,
    w: totalW,
    colW: [modelColW, ...columns.map(() => metricColW)],
    fontFace: FONT.face,
    fontSize: 14,
    border: { type: "solid", pt: 0.5, color: COLOR.border },
    rowH: 0.5,
  });

  const caption = firstText(slideSpec, "caption");
  if (caption) {
    slide.addText(caption, {
      x: MARGIN.x,
      y: tableY + 0.5 * tableRows.length + 0.2,
      w: totalW,
      h: 0.5,
      fontFace: FONT.face,
      fontSize: 13,
      italic: true,
      color: COLOR.muted,
      align: "center",
    });
  }

  const thr = v.iaao_thresholds_footer;
  if (thr && thr.enabled && thr.text) {
    slide.addText(thr.text, {
      x: MARGIN.x,
      y: FOOTER.y - 0.45,
      w: totalW,
      h: 0.3,
      fontFace: FONT.face,
      fontSize: 10,
      color: COLOR.muted,
      align: "center",
    });
  }
}

// -----------------------------------------------------------------------------
// Layout: annotated_chart
// -----------------------------------------------------------------------------

// Draw a schematic regressivity scatter (ratio vs log-price) entirely with
// shapes: axes, ~50 synthetic points, r=1 reference line, and a downward
// LOWESS-style trend. Used as the fallback when the real asset is missing.
function drawRegressivityScatter(slide, x, y, w, h) {
  const pad = { left: 0.55, right: 0.25, top: 0.35, bottom: 0.55 };
  const plotX = x + pad.left;
  const plotY = y + pad.top;
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  // Plot frame.
  slide.addShape("rect", {
    x: plotX,
    y: plotY,
    w: plotW,
    h: plotH,
    fill: { color: "FFFFFF" },
    line: { color: COLOR.border, width: 0.75 },
  });

  // Data ranges chosen to be representative (not actual).
  const xMin = 11.0;
  const xMax = 14.2;
  const yMin = 0.6;
  const yMax = 1.5;
  const toPX = (xv) => plotX + ((xv - xMin) / (xMax - xMin)) * plotW;
  const toPY = (yv) => plotY + plotH - ((yv - yMin) / (yMax - yMin)) * plotH;

  // r = 1 reference line (solid, dark).
  slide.addShape("line", {
    x: plotX,
    y: toPY(1.0),
    w: plotW,
    h: 0,
    line: { color: COLOR.text, width: 1, dashType: "dash" },
  });
  slide.addText("r = 1", {
    x: plotX + plotW - 0.6,
    y: toPY(1.0) - 0.22,
    w: 0.55,
    h: 0.22,
    fontFace: FONT.face,
    fontSize: 9,
    italic: true,
    color: COLOR.text,
    align: "right",
  });

  // Generate ~50 deterministic scatter points with a downward trend.
  let seed = 17;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed % 10000) / 10000;
  };
  const nPts = 50;
  const slope = -0.18;
  const intercept = 1.0 - slope * ((xMin + xMax) / 2);
  for (let i = 0; i < nPts; i++) {
    const xv = xMin + (xMax - xMin) * ((i + 0.5) / nPts) + (rand() - 0.5) * 0.25;
    const noise = (rand() - 0.5) * 0.32;
    const yv = intercept + slope * xv + noise;
    const cx = toPX(xv);
    const cy = toPY(yv);
    slide.addShape("ellipse", {
      x: cx - 0.04,
      y: cy - 0.04,
      w: 0.08,
      h: 0.08,
      fill: { color: COLOR.accent, transparency: 30 },
      line: { color: COLOR.accent, width: 0 },
    });
  }

  // Trend line (the LOWESS-style visualization): piecewise straight-line
  // approximation of the regressivity pattern.
  const trendPts = [];
  const nSeg = 12;
  for (let i = 0; i <= nSeg; i++) {
    const xv = xMin + (xMax - xMin) * (i / nSeg);
    const yv = intercept + slope * xv;
    trendPts.push({ x: toPX(xv), y: toPY(yv) });
  }
  for (let i = 0; i < trendPts.length - 1; i++) {
    const a = trendPts[i];
    const b = trendPts[i + 1];
    slide.addShape("line", {
      x: a.x,
      y: a.y,
      w: b.x - a.x,
      h: b.y - a.y,
      line: { color: COLOR.bad, width: 2.25 },
    });
  }

  // Region labels: over-assessed above r=1, under-assessed below.
  slide.addText("over-assessed (r > 1)", {
    x: plotX + 0.1,
    y: plotY + 0.05,
    w: plotW / 2,
    h: 0.3,
    fontFace: FONT.face,
    fontSize: 11,
    italic: true,
    color: COLOR.bad,
  });
  slide.addText("under-assessed (r < 1)", {
    x: plotX + plotW / 2 - 0.1,
    y: plotY + plotH - 0.35,
    w: plotW / 2,
    h: 0.3,
    fontFace: FONT.face,
    fontSize: 11,
    italic: true,
    color: COLOR.bad,
    align: "right",
  });

  // Axis labels.
  slide.addText("log(sale price)", {
    x: plotX,
    y: plotY + plotH + 0.15,
    w: plotW,
    h: 0.3,
    fontFace: FONT.face,
    fontSize: 11,
    color: COLOR.text,
    align: "center",
  });
  slide.addText("assessment ratio r = P̂ / P", {
    x: x,
    y: plotY + plotH / 2 - 0.15,
    w: pad.left - 0.05,
    h: 0.3,
    fontFace: FONT.face,
    fontSize: 10,
    color: COLOR.text,
    align: "right",
  });

  // Y tick labels at 0.8, 1.0, 1.2, 1.4.
  [0.8, 1.0, 1.2, 1.4].forEach((ytick) => {
    slide.addText(String(ytick), {
      x: x,
      y: toPY(ytick) - 0.12,
      w: pad.left - 0.1,
      h: 0.25,
      fontFace: FONT.face,
      fontSize: 9,
      color: COLOR.muted,
      align: "right",
    });
  });
  // X tick labels at 11, 12, 13, 14.
  [11, 12, 13, 14].forEach((xtick) => {
    slide.addText(String(xtick), {
      x: toPX(xtick) - 0.2,
      y: plotY + plotH + 0.02,
      w: 0.4,
      h: 0.22,
      fontFace: FONT.face,
      fontSize: 9,
      color: COLOR.muted,
      align: "center",
    });
  });
}

function renderAnnotatedChart(slide, slideSpec) {
  const v = slideSpec.visual || {};
  const totalW = SLIDE.w - 2 * MARGIN.x;
  const mainFrac = v.main_figure?.width_fraction ?? 0.7;
  const gap = 0.3;
  const mainW = (totalW - gap) * mainFrac;
  const calloutW = totalW - gap - mainW;

  const figY = BODY.y;
  const figH = BODY.h * 0.75;

  // Main figure: try real asset first, fall back to native scatter.
  const assetRel = v.main_figure?.asset_path;
  const assetAbs = assetRel ? path.join(PROJECT_ROOT, assetRel) : null;
  const assetExists = assetAbs && fs.existsSync(assetAbs);

  if (assetExists) {
    slide.addImage({
      path: assetAbs,
      x: MARGIN.x,
      y: figY,
      w: mainW,
      h: figH,
      sizing: { type: "contain", w: mainW, h: figH },
    });
  } else {
    drawRegressivityScatter(slide, MARGIN.x, figY, mainW, figH);
  }

  if (v.main_figure?.dataset_label) {
    const label = sanitizeText(v.main_figure.dataset_label) + (assetExists ? "" : "  (schematic)");
    slide.addText(label, {
      x: MARGIN.x,
      y: figY + figH + 0.05,
      w: mainW,
      h: 0.3,
      fontFace: FONT.face,
      fontSize: 10,
      italic: !assetExists,
      color: COLOR.muted,
      align: "center",
    });
  }

  // Callout cards on the right.
  const cards = (v.callouts && v.callouts.cards) || [];
  const cardsX = MARGIN.x + mainW + gap;
  const notationH = 1.1;
  const cardsAreaH = figH - notationH - 0.2;
  if (cards.length > 0) {
    const cardH = cardsAreaH / cards.length - 0.15;
    cards.forEach((c, i) => {
      const y = figY + i * (cardH + 0.15);
      const isOver = /over/i.test(c.text || "") || /> 1/.test(c.text || "");
      slide.addShape("roundRect", {
        x: cardsX,
        y,
        w: calloutW,
        h: cardH,
        fill: { color: COLOR.neutralFill },
        line: { color: isOver ? COLOR.bad : COLOR.accent, width: 1.25 },
      });
      slide.addText(c.tier || "", {
        x: cardsX + 0.15,
        y: y + 0.1,
        w: calloutW - 0.3,
        h: 0.4,
        fontFace: FONT.face,
        fontSize: 13,
        bold: true,
        color: isOver ? COLOR.bad : COLOR.accent,
      });
      slide.addText(c.text || "", {
        x: cardsX + 0.15,
        y: y + 0.5,
        w: calloutW - 0.3,
        h: cardH - 0.55,
        fontFace: FONT.face,
        fontSize: 12,
        color: COLOR.text,
        valign: "top",
      });
    });
  }

  // Notation box at the bottom of the callout column.
  const notationEntries = (v.notation_box && v.notation_box.entries) || [];
  if (notationEntries.length > 0) {
    const notY = figY + figH - notationH;
    slide.addShape("roundRect", {
      x: cardsX,
      y: notY,
      w: calloutW,
      h: notationH,
      fill: { color: COLOR.accentSoft },
      line: { color: COLOR.accent, width: 0.75 },
    });
    const lines = notationEntries.map((e) => `${e.symbol}: ${e.meaning}`).join("\n");
    slide.addText(lines, {
      x: cardsX + 0.15,
      y: notY + 0.1,
      w: calloutW - 0.3,
      h: notationH - 0.2,
      fontFace: FONT.face,
      fontSize: 11,
      color: COLOR.text,
      valign: "top",
    });
  }

  const caption = firstText(slideSpec, "caption");
  if (caption) {
    slide.addText(caption, {
      x: MARGIN.x,
      y: figY + figH + 0.4,
      w: totalW,
      h: 0.4,
      fontFace: FONT.face,
      fontSize: 13,
      italic: true,
      color: COLOR.muted,
      align: "center",
    });
  }
}

// -----------------------------------------------------------------------------
// Dispatcher
// -----------------------------------------------------------------------------

const RENDERERS = {
  title_slide: renderTitleSlide,
  opener_infographic: renderOpenerInfographic,
  process_flow: renderProcessFlow,
  concept_diagram: renderConceptDiagram,
  comparison_table: renderComparisonTable,
  map_plus_timeline: renderMapPlusTimeline,
  split_panel: renderSplitPanel,
  annotated_chart: renderAnnotatedChart,
};

// Layouts that draw their own title area (skip the default title band).
const LAYOUTS_WITH_CUSTOM_TITLE = new Set(["title_slide"]);
// Layouts that do not get the preliminaries footer (e.g., the opening slide).
const LAYOUTS_WITHOUT_FOOTER = new Set(["title_slide"]);

function renderSlide(pptx, slideSpec, totalSlides) {
  const layout = slideSpec.layout_type;
  if (!SUPPORTED_LAYOUTS.includes(layout)) {
    throw new Error(
      `Unsupported layout_type "${layout}" on slide ${slideSpec.slide_number}. ` +
        `Supported: ${SUPPORTED_LAYOUTS.join(", ")}.`,
    );
  }
  const slide = pptx.addSlide();
  if (!LAYOUTS_WITH_CUSTOM_TITLE.has(layout)) {
    addTitle(slide, slideSpec.title || "");
  }
  RENDERERS[layout](slide, slideSpec);
  if (!LAYOUTS_WITHOUT_FOOTER.has(layout)) {
    addFooter(slide, slideSpec.slide_number, totalSlides);
  }
  addSpeakerNotes(slide, slideSpec);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

function loadDeck() {
  const raw = fs.readFileSync(DECK_JSON_PATH, "utf8");
  const deck = JSON.parse(raw);
  if (!Array.isArray(deck.slides)) throw new Error("deck.json has no slides array.");
  return deck;
}

async function main() {
  const deck = loadDeck();
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.title = `${deck.deck?.project || "Deck"} — ${deck.deck?.section || ""}`.trim();

  const total = deck.slides.length;
  deck.slides
    .slice()
    .sort((a, b) => a.slide_number - b.slide_number)
    .forEach((s) => renderSlide(pptx, s, total));

  await pptx.writeFile({ fileName: OUTPUT_PATH });
  console.log(`Wrote ${OUTPUT_PATH} (${total} slides).`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
