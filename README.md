# Wealth Vision — refactored structure

Originally a single 6,541-line `app.py`. Split into:

```
app.py                  # entrypoint: page config, CSS, auth gate, nav, page dispatch, footer
config.py                # NEWS_API_KEY now read from env var, not hardcoded
core/                     # business logic, no Streamlit UI code
  news_sentiment.py, price_predictor.py, data_loader.py, strategy_engine.py,
  ml_predictor.py, portfolio_manager.py, stock_analyzer.py, utils.py
  auth.py, multi_exchange.py, advanced_strategies.py, risk_analytics.py,
  backtester.py, technical_patterns.py, fundamental_analysis.py, export_utils.py
ui/
  auth_page.py            # login / signup / guest screen
  shared.py                # display_stock_analysis(), used by several pages
  pages/                    # one file per nav tab (11 files)
```

## Before running
Set your NewsAPI key as an environment variable instead of the old hardcoded value:

```
export NEWS_API_KEY="your-key-here"
```
(A working key was previously committed to this repo in plain text — rotate it on newsapi.org.)

Then:
```
pip install -r requirements.txt
streamlit run app.py
```

## Known follow-ups (not done automatically)
- `st.image(...)` and `page_icon=...` in `app.py` still reference a hardcoded
  Windows path to a logo file (`C:\Users\Lenovo\...`). Replace with a relative
  path to a logo file bundled in the repo.
- `.auth_data/users.json` was excluded from this refactor and should be removed
  from git history (it was committed to a public repo).
