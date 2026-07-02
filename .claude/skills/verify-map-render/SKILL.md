---
name: verify-map-render
description: >-
  Verify the Folium V6 salinity-risk map actually renders on the deployed (or
  local) Streamlit dashboard, by driving a real browser via Playwright. Pierces
  the Streamlit iframe and asserts Leaflet, tiles, decision panel, hover/click
  lookup, and mobile layout load with no CSP/sandbox errors. Use after map,
  app.py rendering, or Streamlit Cloud deploy changes.
disable-model-invocation: true
---

# Verify V6 Map Render

This project's #1 recurring UI risk is the Folium V6 salinity-risk map silently
failing to render on Streamlit Cloud. Unit tests and Streamlit AppTest cannot see
it because the map is injected through `st.iframe`. Only a real browser catches
CSP, sandbox, Leaflet, tile, and mobile-overlay failures.

## Prerequisites

- Playwright MCP/browser tools connected.
- Target URL: deployed app
  `https://aral-saxaul-ai-7skntrgzemfiwqmjd7feoo.streamlit.app/` or local
  `http://localhost:8501` after `streamlit run app.py`.

## Checks

1. **Navigate** to the target URL. The primary V6 map lives in the
   `Карта и итоги` tab. If Streamlit restores another tab, click `Карта и итоги`.

2. **Wait** for Streamlit hydration and map initialization. Do not probe the iframe
   before the title `Aral Saxaul V6` and heading `Карта риска соли V6` are visible.

3. **Console errors** must be empty for relevant failures. Fail on:
   - Content Security Policy script/frame blocking;
   - sandboxed script execution failures;
   - Leaflet JS errors;
   - tile-loading 4xx/CORS failures.

4. **Iframe/Leaflet probe** should confirm:
   - a visible `iframe` exists;
   - `.leaflet-container` exists and has non-zero dimensions;
   - `window.L` is present inside the frame;
   - `#decision-help` exists;
   - layer control includes `Зоны риска соли V6` and `Балл V6: ниже риск соли`;
   - the HTML does not contain the old blue AOI outline style `#3388ff`.

5. **Interaction probe** should perform a real mouse move/click on a colored V6
   area. The decision panel must update with:
   - `Балл низкого риска соли`;
   - approximate salinity risk;
   - zone class;
   - next action;
   - coordinates.

6. **Mobile probe** should resize to about `390x844`. The map must remain usable;
   decision panel and legend should have bounded height/scroll and must not cover
   the entire map.

7. **Network/tile probe** should show no blocked/failed basemap tile requests that
   make the map unusable.

## Report

State PASS/FAIL with concrete probe numbers: iframe size, Leaflet present, loaded
tiles, decision panel state, console errors, and mobile overlay dimensions. On
failure, name the likely broken path (`_render_map`, `st.iframe`, generated V6
HTML, or tile/network issue) and include a screenshot path when possible.
