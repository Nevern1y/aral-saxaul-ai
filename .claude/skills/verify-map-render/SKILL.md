---
name: verify-map-render
description: >-
  Verify the Folium suitability map actually renders on the deployed (or local)
  Streamlit dashboard, by driving a real browser via the Playwright MCP server.
  Pierces the Streamlit components iframe and asserts the Leaflet map + tiles
  loaded with no CSP/sandbox errors. Use after any change to map embedding,
  app.py rendering, or a Streamlit Cloud redeploy.
disable-model-invocation: true
---

# Verify map render

This project's #1 recurring bug is the Folium map silently failing to render on
Streamlit Cloud (CSP / sandboxed-iframe / blocked-tile failures). Unit tests and
AppTest cannot see it — the map is injected via `streamlit.components.v1.html`,
which lives in a nested iframe. Only a real browser catches it. This skill
automates that check with the Playwright MCP server.

## Prerequisites

- Playwright MCP server connected (`mcp__playwright__*` tools). If absent:
  `claude mcp add --scope user playwright -- npx -y @playwright/mcp@latest`
- A target URL: the deployed app
  (https://aral-saxaul-ai-7skntrgzemfiwqmjd7feoo.streamlit.app/) or a local
  `streamlit run app.py` (http://localhost:8501).

## Steps

1. **Navigate** to the target URL with `browser_navigate`. The map lives in the
   "Карта и итоги" (analytics) tab, which is the default tab — no click needed,
   but if verifying the logistics map, select that tab first.

2. **Wait** ~4-5s with `browser_wait_for` for Streamlit hydration + async tile
   loads. The map is slow; don't probe too early.

3. **Check console** with `browser_console_messages` (level=error). Assert NO
   errors matching the known failure fingerprints:
   - "Refused to frame" / "violates the following Content Security Policy"
   - "blocked a frame with origin ... from accessing a cross-origin frame"
   - "Blocked script execution ... because the document's frame is sandboxed"

4. **Pierce the iframe and probe Leaflet** with `browser_evaluate`. Run:

   ```js
   () => {
     const f = document.querySelector('iframe[data-testid="stIFrame"]');
     let d = null;
     try { d = f && (f.contentDocument || f.contentWindow.document); } catch (e) {}
     const c = d && d.querySelector('.leaflet-container');
     const tiles = d ? [...d.querySelectorAll('.leaflet-tile, .leaflet-tile-pane img')] : [];
     return {
       hasFrame: !!f,
       reachable: !!d,
       hasMap: !!c,
       mapW: c && c.offsetWidth,
       mapH: c && c.offsetHeight,
       tileCount: tiles.length,
       loadedTiles: tiles.filter(t => t.naturalWidth > 0).length,
     };
   }
   ```

   **PASS** requires: `hasFrame && hasMap && mapW > 0 && loadedTiles > 0`.
   - `hasFrame` false → the components.html embed didn't render at all.
   - `reachable` false → sandbox blocks same-origin access; fall back to
     targeting the iframe element directly via `browser_snapshot` (its
     accessibility tree descends into frames). Treat "cannot reach the frame at
     all" as itself a red flag and capture a screenshot.
   - `hasMap` false → Leaflet JS never ran (the classic CSP/sandbox failure).
   - `tileCount > 0` but `loadedTiles == 0` → tiles requested but blocked
     (gray-box map).

5. **Check tile network** with `browser_network_requests`. Filter to the
   basemap tile host; assert no blocked / 4xx / CORS-failed tile requests.

6. **Screenshot** with `browser_take_screenshot` as an artifact whenever any
   check fails (the human-visible gray-box state), and optionally on success.

## Report

State PASS/FAIL with the probe object's numbers, the offending console/network
lines on failure, and the screenshot path. On failure, the most likely cause is
the map-embedding regression described in `streamlit-deploy-reviewer` — name
which embedding approach is currently in `_render_map` (app.py:65-67) and
whether it matches the last-known-good `components.html` path.

## Note on the deprecation

`st.components.v1.html` is deprecated (removal announced after 2026-06-01) in
favor of `st.iframe`. If app.py migrates to `st.iframe`, the wrapping iframe may
change `data-testid`; re-confirm the selector. If it ever migrates to `st.html`,
the map moves to the TOP document (no stIFrame) and the selector becomes a
top-level `.leaflet-container` — but st.html strips JS, so verify extra hard.
