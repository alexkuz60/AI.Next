# Project Summary

## Overall Goal
Optimize and document a Dash/Plotly application (AI.Next) for analyzing and predicting random numeric event data, including code quality improvements, library API updates, and mathematical documentation.

## Key Knowledge
- **Tech Stack:** Dash 4.4.1, Plotly 7.0.0, dash-mantine-components 2.8.0, NumPy 2.5.2, SciPy 1.18.1, Pandas 3.0.5
- **Project Structure:** `app_next.py` (main Dash app), `libdatas.py` (data processing/FFT/matrix calculations), `libgraphs.py` (Plotly visualization), `libmatrix.py` (experimental, unused), `csv_data/` (input CSV files)
- **Virtual Environment:** `.venv/bin/python` must be used to run the app; system Python lacks dependencies
- **DMC 2.8 Limitations:** No `ColorSchemeScript` component; use `pre_render_color_scheme()` instead. No `dcc.Style` or `html.Style`; use `app.index_string` with `{%metas%}` syntax (not `{metas}`)
- **Plotly 7 Changes:** `hoversort` values changed from `"ascending"` to `"value ascending"`
- **User Preference:** Communication in Russian; user is non-professional developer needing guidance
- **Debug Mode:** Controlled via `APP_DEBUG` environment variable (default: "true")
- **Dark Theme Fix:** Required `pre_render_color_scheme()` + `app.index_string` CSS injection + `MantineProvider(forceColorScheme="dark")` for consistent rendering across browsers (Firefox had white background artifacts)

## Recent Actions
- **[DONE] Code Audit:** Created `audit_qwen.md` identifying 12 unused/experimental functions, critical bugs (`matchDiffDeep` unreachable branch, empty callback without Output, drawer never closing, hardcoded debug=True), and architectural issues
- **[DONE] Full Optimization:** Rewrote all 3 main files with Dash 4/Plotly 7/DMC 2 APIs, type hints, docstrings, bug fixes; reduced requirements.txt from 34 to 7 packages
- **[DONE] Firefox Dark Theme Fix:** Resolved white background artifacts using `pre_render_color_scheme()`, global CSS via `app.index_string`, and `forceColorScheme="dark"`
- **[DONE] Mathematical Documentation:** Wrote comprehensive LaTeX-formatted algorithm description in README.md covering event matrix calculation, differentiated wave graphs, FFT analysis, prognosis system, and inverse matrix
- **[DONE] Syntax Verification:** All files compile successfully; app starts and serves on http://127.0.0.1:8050/

## Current Plan
1. [DONE] Complete code audit and optimization
2. [DONE] Fix browser compatibility issues (Firefox dark theme)
3. [DONE] Write mathematical documentation in README.md
4. [TODO] Continue development in next session (user indicated they will return later)

---

## Summary Metadata
**Update time**: 2026-09-09T15:12:48.267Z
