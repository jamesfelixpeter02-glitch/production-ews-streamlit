# Production Early Warning System (Python/Streamlit)

This dashboard detects early warning signs in a production system (e.g., oil well) and suggests proactive tweaks (like choke changes, tests, and water management actions).

## Features
- Monitors Qo, WCUT, PI, PR/PWF/WHP, GOR
- Rolling z-score, EWMA smoothing, simple CUSUM change detection
- Instant health score and rule-based recommendations
- Interactive line charts and alert table
- Downloadable analysis CSV

## Quick Start
```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the dashboard
streamlit run app.py
```

Upload a CSV/Excel with at least a date column plus any of: oil rate (Qo), water rate (Qw), water cut (%), reservoir pressure (PR), bottomhole pressure (PWF), wellhead pressure (WHP), gas rate (Qg), or GOR. You can map columns in the UI.

## How detection works
- **Rolling z-score** flags large deviations vs recent history.
- **EWMA** smooths noise to expose level shifts.
- **CUSUM** gives simple change-point hints.
- **Health score** penalizes risky signals.
- **Recommendations** use practical rules (e.g., rising WCUT → reduce drawdown, run diagnostics; PI drop → skin/scale check).

## Customize
- Adjust thresholds/windows in the sidebar.
- Edit the `map_recommendations()` function in `app.py` to match your facility rules (e.g., gas-lift tuning, chemical injection, test frequency).

## Notes
- Keep units consistent.
- PI is computed as `Qo / (PR - PWF)` when columns are available.
- If WCUT column is missing, it is estimated from Qw and Qo.