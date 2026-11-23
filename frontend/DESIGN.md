---
name: QMLKit
description: A warm, honest console for live kennel telemetry, VOC screening, and quantum/classical model benchmarking.
colors:
  ground: "#f6f1e8"
  surface: "#ffffff"
  surface-tint: "#fbf8f2"
  ink: "#22201d"
  muted: "#6c675f"
  line: "#e8e1d4"
  line-strong: "#d9cfbd"
  action: "#2f5fd2"
  action-hover: "#244bb0"
  action-tint: "#eaf0ff"
  good: "#2f7d5b"
  good-tint: "#e6f4ec"
  warn: "#a9681f"
  warn-tint: "#fbf1e0"
  bad: "#b23a2f"
  bad-tint: "#fbeae7"
typography:
  display:
    fontFamily: "Nunito Sans, Avenir Next, Segoe UI, system-ui, sans-serif"
    fontSize: "22px"
    fontWeight: 800
    lineHeight: 1.5
    letterSpacing: "-0.03em"
  headline:
    fontFamily: "Nunito Sans, Avenir Next, Segoe UI, system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 800
    lineHeight: 1.5
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Nunito Sans, Avenir Next, Segoe UI, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Nunito Sans, Avenir Next, Segoe UI, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 700
    letterSpacing: "0.06em"
  metric:
    fontFamily: "Nunito Sans, Avenir Next, Segoe UI, system-ui, sans-serif"
    fontSize: "40px"
    fontWeight: 800
    lineHeight: 1
rounded:
  lg: "24px"
  md: "16px"
  sm: "10px"
  pill: "999px"
spacing:
  sm: "8px"
  md: "16px"
  lg: "24px"
components:
  button-primary:
    backgroundColor: "{colors.action}"
    textColor: "#ffffff"
    rounded: "{rounded.pill}"
    padding: "12px 20px"
    typography: "14px / 800"
  button-primary-hover:
    backgroundColor: "{colors.action-hover}"
    textColor: "#ffffff"
    rounded: "{rounded.pill}"
    padding: "12px 20px"
    typography: "14px / 800"
  button-quiet:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "12px 20px"
    typography: "14px / 800"
  field:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "10px 14px"
    typography: "14px"
  chip:
    backgroundColor: "{colors.surface-tint}"
    textColor: "{colors.muted}"
    rounded: "{rounded.pill}"
    padding: "4px 12px"
    typography: "12px / 800"
  chip-good:
    backgroundColor: "{colors.good-tint}"
    textColor: "{colors.good}"
    rounded: "{rounded.pill}"
    padding: "4px 12px"
    typography: "12px / 800"
  chip-warn:
    backgroundColor: "{colors.warn-tint}"
    textColor: "{colors.warn}"
    rounded: "{rounded.pill}"
    padding: "4px 12px"
    typography: "12px / 800"
  chip-bad:
    backgroundColor: "{colors.bad-tint}"
    textColor: "{colors.bad}"
    rounded: "{rounded.pill}"
    padding: "4px 12px"
    typography: "12px / 800"
  card:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.lg}"
    padding: "28px"
  card-sm:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.md}"
    padding: "18px 20px"
  tab-active:
    backgroundColor: "{colors.action-tint}"
    textColor: "{colors.action}"
    rounded: "{rounded.pill}"
    padding: "9px 16px"
    typography: "13px / 700"
  progress-fill:
    backgroundColor: "{colors.action}"
    rounded: "{rounded.pill}"
---

# Design System: QMLKit

## Overview

**Creative North Star: "The Well-Made Laboratory Console"**

QMLKit looks and behaves like a well-made consumer product that happens to show laboratory data. A warm off-white ground carries white, soft-cornered surfaces separated by hairline seams; the only saturated accent is a single indigo action colour reserved for controls and meaningful state. The console is dense enough for telemetry, but plain-spoken: the live route comes first, and each screen leads with what needs attention rather than a wall of identical metric cards.

Confidence is expressed honestly. Status colour is assigned to real conditions — a live stream, an untrained model, an elevated reading — and never invented to decorate a screen. An `untrained` diagnosis is a normal, expected state with its own calm treatment, and every screening view carries its research-aid disclaimer. Depth is quiet: a hairline border and one soft, diffuse shadow, never glow, glass, gradients, or neon instrument-panel effects.

**Key Characteristics:**
- Warm ground (#f6f1e8) with white, soft-cornered surfaces (#ffffff) and hairline seams (#e8e1d4).
- One indigo action colour (#2f5fd2); good/warn/bad (#2f7d5b / #a9681f / #b23a2f) only for honest state.
- Self-hosted variable humanist sans (Nunito Sans, 400–800) with tabular numerals for every reading.
- 8px spacing rhythm, 24/16/10px radii, pill-shaped controls, 1280px container.
- Flat at rest with one soft offset shadow on primary surfaces; one restrained press easing, disabled under reduced motion.

## Colors

The palette pairs a single cool indigo action against a warm neutral ground, with three restrained status pairs for honest state communication.

### Primary
- **Action Indigo** (#2f5fd2): The one accent. Fills the primary button, active tab, active nav item, progress fills, focus outlines, chart lines, and highlighted metric values. Its hover partner is Action Indigo Deep (#244bb0) and its wash is Action Indigo Wash (#eaf0ff).
- **Action Indigo Deep** (#244bb0): Hover state of the primary button only.
- **Action Indigo Wash** (#eaf0ff): Active tabs/nav, selection, and the field focus halo — a quiet way to show "this is live/selected" without a second accent.

### Neutral
- **Warm Ground** (#f6f1e8): Page canvas and sticky header background.
- **Surface White** (#ffffff): Primary cards, fields, quiet buttons, and small surfaces.
- **Tinted White** (#fbf8f2): Quiet-button hover and small chips.
- **Ink** (#22201d): Body and headline text, brand wordmark.
- **Muted Ink** (#6c675f): Secondary text, labels, captions.
- **Hairline** (#e8e1d4): Default border, dividers, progress track.
- **Hairline Strong** (#d9cfbd): Field and quiet-button borders, hover intensification of Hairline.

### Semantic states
- **Good Green** (#2f7d5b) / wash (#e6f4ec): Live, connected, trained, in-range, true-positive states.
- **Warn Amber** (#a9681f) / wash (#fbf1e0): Untrained, calibrating, uncertain, waiting, false-positive/negative states.
- **Bad Red** (#b23a2f) / wash (#fbeae7): Disconnected, action-required, elevated-risk states.

### Named Rules
**The One Action Colour Rule.** Indigo (#2f5fd2) is the only interactive accent. Status colours never become buttons and the action colour never becomes a status — rarity is what makes "actionable" readable.

**The Honest State Rule.** Good/warn/bad are earned by real connection, model, and reading state. `untrained` is shown as warn, never as a fabricated result, and every screening surface states that it is a research aid.

## Typography

**Display Font:** Nunito Sans (self-hosted variable, weights 400–800), with "Avenir Next", "Segoe UI", system-ui, sans-serif fallbacks
**Body Font:** Nunito Sans (same family and fallbacks)
**Label / Numeric Font:** Nunito Sans; all figures use tabular numerals via the `.num` class

**Character:** One rounded humanist family across every role keeps the console friendly without losing precision. Weight and spacing, not a second typeface, carry the hierarchy — heavy negative tracking on the wordmark, restrained tracking on small uppercase labels.

### Hierarchy
- **Display** (weight 800, 22px, tracking -0.03em): The QMLKit wordmark in the header.
- **Headline / Title** (weight 800, 18px, tracking -0.01em): `.section-title` — panel and page headings.
- **Body** (weight 400, 15px, line-height 1.5): Window copy, helper text, table cells.
- **Label** (weight 700, 12px, tracking 0.06em, uppercase, muted): `.label` — form labels and data captions above values.
- **Metric** (weight 800, 40–44px, line-height 1, tabular): Large risk/index numerals in diagnostic cards.

### Named Rules
**The Tabular Figures Rule.** Every telemetry or metric number uses `.num` (tabular-nums). Values that change live must not shift width as they update.

## Layout

A single centered container (`max-width: 1280px`, `px-5` on small screens and `sm:px-8`, `py-8` / `sm:py-10`) holds all five routes. The header is sticky and opaque with a hairline bottom border; the footer carries the research-aid disclaimer. At `lg` and above, content uses a 12-column grid (e.g. 7/5 lead split, 4/8 control/results split); below that it collapses to one column and the nav/footer wrap.

The spacing rhythm is 8px: `.stack` uses 16px between children, `.row` uses 12px, and section groups use 24px (`gap-6`). Cards use 28px padding, small surfaces 18px 20px, and `.divider` carries 20px vertical margins.

## Elevation & Depth

Depth is a hybrid but restrained: surfaces are flat at rest, separated by a 1px hairline, and only the primary `.surface` card carries a soft, diffuse shadow. Depth is never structural or glowing — it is a gentle lift that keeps white cards distinct from the warm ground.

### Shadow Vocabulary
- **Soft surface lift** (`box-shadow: 0 6px 20px -6px rgba(34, 32, 29, 0.12), 0 2px 6px -2px rgba(34, 32, 29, 0.06)`): Applied only to `.surface` primary cards; `.surface-sm` stays flat.

### Named Rules
**The Hairline Seam Rule.** Separation comes first from a 1px border (#e8e1d4), not a heavy shadow. One soft shadow per primary card is the ceiling; no layered, hard-offset, or coloured glow shadows.

**The Restrained Press Rule.** The only motion is a 180ms press (`cubic-bezier(0.22, 1, 0.36, 1)`, scale 0.97) on buttons; colour changes run at 160ms ease. Under `prefers-reduced-motion` the press is removed and colour transitions shorten to 120ms linear.

## Shapes

Soft, human corners throughout: 24px on primary cards (16px under 900px), 16px on small surfaces, 10px on inputs, and a 999px pill for buttons, chips, tabs, nav items, and progress tracks. The brand mark is a 12px-radius square. Borders are a single hairline; progress tracks clip their fill with `overflow: hidden`. No hard offsets, no glass panels, no gradient fills.

### Named Rules
**The Soft Corners Rule.** Cards are soft-cornered, controls are pill-shaped, and every radius comes from the 24/16/10/pill scale. Nothing is square-cornered or sharply cut.

## Components

### Buttons
- **Shape:** Pill (999px), `12px 20px` padding, 14px/800 text.
- **Primary:** Indigo (#2f5fd2) with white text; hover deepens to #244bb0; active scales to 0.97; disabled drops to 0.5 opacity with `not-allowed`.
- **Quiet:** White surface, ink text, Hairline Strong border; hover washes to Tinted White (#fbf8f2).
- **Focus:** 3px action outline at 2px offset (`:focus-visible`).

### Chips
- **Style:** Pill, `4px 12px`, 12px/800, Hairline Strong border on the neutral variant, Tinted White background, muted text.
- **State:** `chip-good` / `chip-warn` / `chip-bad` swap to status tint backgrounds with no border. Used for stream/model state, paradigm tags, and counts.

### Cards / Containers
- **Corner Style:** 24px (`.surface`), 16px (`.surface-sm`).
- **Background:** Surface White; status cards override border and background with a semantic colour pair.
- **Shadow Strategy:** One soft surface lift on `.surface`; `.surface-sm` flat.
- **Border:** 1px Hairline; dashed Hairline for empty states.
- **Internal Padding:** 28px, or 18px 20px for small surfaces.

### Inputs / Fields
- **Style:** White background, Hairline Strong border, 10px radius, `10px 14px`, 14px text; range inputs drop padding/border and take the action accent colour.
- **Focus:** Action border plus a 3px action-tint halo (`.field:focus`); global `:focus-visible` uses a 3px action outline.
- **Error / Disabled:** Error and alert messages use Bad Red with its tint; disabled controls use 0.5 opacity.

### Navigation
- **Style:** Sticky opaque ground header; pill `.nav-link` items in muted 13px/700 with lucide line icons at 16px. Active item is action-tint with action text; hover is white with ink text. Labels hide below `sm`, leaving icons.

### Tabs
- **Style:** Pill `.tab`, 13px/700, Hairline Strong border, white background, muted text; `.tab-active` is action-tint with an action border and text.

### Progress
- **Style:** 10px pill track in Hairline; fill is action by default and takes a status colour when the bar represents a reading.

### Signature: Telemetry Visualizers
- Kennel diagram, IMU waveform canvas, slow-channel bars, loss chart, and biomarker waterfall are all drawn in solid palette values with no gradients or glow. Series use action/good/warn/bad; gridlines and tracks use Hairline; chart text uses muted. The Q glyph brand mark is the only letter-based mark.

## Do's and Don'ts

### Do:
- **Do** keep exactly one interactive accent — Action Indigo (#2f5fd2) — and use status colours only for earned state.
- **Do** show `untrained`, disconnected, and empty states explicitly with their calm treatments and keep every screening surface labelled as a research aid.
- **Do** apply `.num` tabular figures to all changing readings, and consume palette values from the CSS custom properties rather than duplicating hex literals.
- **Do** use the 24/16/10/pill radius scale and the single soft surface lift; let hairline borders do most of the separating.

### Don't:
- **Don't** add neon glows, glass/backdrop-filter panels, gradients, or coloured/hard-offset shadows — the world is flat and warm.
- **Don't** introduce a second action colour, promote a status colour to a control, or fabricate a result to make a screen look complete.
- **Don't** render live figures in proportional type, or scale the press animation beyond the single restrained 180ms button press.
