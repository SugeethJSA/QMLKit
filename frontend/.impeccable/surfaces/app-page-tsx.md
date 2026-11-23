---
version: 1
slug: "app-page-tsx"
primary_target: "app/page.tsx"
related_targets: ["app/diagnostics/page.tsx","app/train/page.tsx","app/collect/page.tsx","app/lab/page.tsx"]
---

Scope: Operate-mode web console across all five routes (/, /diagnostics, /train, /collect, /lab). Audience: researchers/evaluators, kennel operators, and clinical/lab reviewers sharing one live truth. Job: watch kennel telemetry, record labelled sessions, train and compare models, and read a diagnostic with honest confidence.

Task and proof: trace a screening result back to a real VOC evidence window and a labelled kennel session; keep "untrained" and "research aid" states honest. No fabricated evidence, no seeded fake results, no unsupported claims.

Constraints: preserve routes, WebSocket/REST behaviour, responsive layout, accessibility, static export, FastAPI hosting, and desktop embedding. No neon glows, bright gradients, or glass decoration.

THESIS: A laboratory console that behaves like a well-made consumer product: plain language, one action colour, and no decorative glow. It refuses the neon instrument-panel arrangement.

OWN-WORLD: warm off-white ground (#f6f1e8), white soft-cornered surfaces (24px), hairline seams, one indigo action colour (#2f5fd2) reserved for actionable controls and meaningful state, rounded humanist sans, 8px spacing rhythm, one restrained spring press on action only.

STORY: The operator sees the live route first, understands what is working and what needs action, and reaches the next step without hunting.

FIRST VIEWPORT: top navigation with product name; one lead panel showing live kennel signal and what needs attention; a compact supporting status list; the primary action at top right and repeated in the attention panel. No four-identical-card KPI wall.

FORM: Warm Consumer Surface (catalog challenger 6 of 6 in round 1); seed: a6de1d365116deab.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance.
