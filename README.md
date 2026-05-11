# AI Infra Screener

Donchian-channel breakout + pattern scanner scoped to four AI-infrastructure theme buckets:

- **Semis** — NVDA, AMD, AVGO, TSM, ASML, ARM, equipment (AMAT/LRCX/KLAC), specialty (ALAB, CRDO), …
- **Hyperscalers** — MSFT, AMZN, GOOGL, META, ORCL
- **Neoclouds** — CRWV, NBIS, APLD, IREN, WULF
- **Power** — VRT, ETN, GEV, CEG, VST, TLN, SMR, OKLO, BWXT, …
- **Optics** — COHR, LITE, CIEN, FN, AAOI, GLW, ANET, POET

Forked from [breakout-detector](https://github.com/leosabourin50-beep/breakout-detector). Adds a category mapping, momentum-ranked hero block per theme, and category filtering on the full scan.

## Run locally

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "POLYGON_API_KEY=your_key_here" > .env
.venv/bin/streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (already done — must be **public** for the free tier).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick this repo, `main` branch, main file `app.py`.
3. Under **Advanced settings → Secrets**, paste:
   ```
   POLYGON_API_KEY = "your_key_here"
   ```
4. Deploy. First build takes ~3–5 min.

`detector.py` and `polygon_adapter.py` read `POLYGON_API_KEY` from the environment or a `.env` file; Streamlit Cloud exposes secrets as env vars automatically.
