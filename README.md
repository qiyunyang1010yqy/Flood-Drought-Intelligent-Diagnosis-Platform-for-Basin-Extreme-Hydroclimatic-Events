# Flood-Drought Intelligent Diagnosis Platform

洪旱智诊·流域极端水文事件融合分析平台

This repository contains a Streamlit demonstration platform for Yangtze Basin extreme hydroclimatic event analysis.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Settings

- Main file path: `app.py`
- Python dependencies: `requirements.txt`

## Notes

The platform reads prepared result tables from:

- `07_event_identification/`
- `08_driver_analysis/`
- `09_ecological_response/`
- `01_boundary/derived_four_basins/`

The current version is a demonstration and visualization platform. It does not download raw data, re-run event identification, re-train models, or re-calculate ecological response during online display.
