# Dawson's Bus Tool 🚌

A Streamlit web app to generate an optimized school bus route from a CSV of students pre-grouped into bus stops. It geocodes the school & stop addresses (OpenStreetMap/Nominatim), builds an approximate TSP route, shows an interactive map, and exports a driver-ready PDF and Excel.

## CSV format
Provide at least these columns (case-insensitive):
- `name`
- `stop_address` (include city/state)
- Optional: `stop_id` (if omitted, the app derives one per unique address)

Example: see `sample_students.csv`.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push these files to a GitHub repo.
2. Go to https://share.streamlit.io and deploy.
3. Set the entrypoint file to `app.py`.