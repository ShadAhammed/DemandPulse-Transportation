# DemandPulse - Transportation

A local-first transportation demand intelligence platform that ingests any mobility CSV, auto-detects schema roles, trains an XGBoost regressor, surfaces peak/low demand windows, and uses a local LLM to produce executive operational briefings - all without sending data to any external service.

---

## Capabilities

| Stage | What it does |
|---|---|
| **Data & Setup** | Ingests CSV or Excel, auto-detects index/target/feature columns, infers output type (binary, discrete, continuous) |
| **Training** | Auto-selects XGBClassifier or XGBRegressor; reports regression metrics (RMSE, R²) or classification metrics (accuracy, F1) |
| **Model & Test Results** | Trains the model, shows test metrics/charts, and summarises feature values plus ML output on test data |
| **Executive Briefing** | Local LLM reads data profiles, feature patterns, and test ML results |

---

## How the pipeline works

One button - **Run Full Pipeline** - runs training, demand insight extraction, and the LLM executive briefing in sequence. Results are cached across tab switches; nothing reruns unless you change hyperparameters or upload new data.

The platform is mode-agnostic: bike sharing, ride-hail, transit, or any tabular demand dataset with arbitrary feature columns.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- Qwen model pulled locally:

```
ollama pull qwen3
```

---

## Installation

```bash
git clone <your-repo-url>
cd DemandPulse-Transportation
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8502](http://localhost:8502).

> **Note:** If you also run ExplainAI / AIExplanator, it typically uses port **8501**. DemandPulse uses **8502** so both can run at the same time without confusion.

---

## Data format

- CSV (`.csv`) or Excel (`.xlsx`, `.xls`)
- One column as the demand target (e.g. `count`, `trips`, `rides`) - numeric regression target
- Optional separate test file; if it lacks the target column, the model generates forecasts only
- Identifier columns (timestamps, UUIDs, row IDs) are auto-detected and excluded; overridable via the Index Columns selector

### Sample bike-sharing data

Bundled under `data/`:

- `train.csv` - hourly features with demand target (`count`)
- `test.csv` - holdout features for forecast-only evaluation

---

## Project structure

```
DemandPulse/
├── app.py                      # UI layout and pipeline orchestration
├── demand_pulse/
│   ├── column_analyzer.py      # Schema inference: index, target, features
│   ├── data_loader.py          # Ingestion, preprocessing, train/test split
│   ├── target_detector.py      # Output type inference and model routing
│   ├── model_trainer.py        # Adaptive XGBoost training and evaluation
│   ├── demand_insights.py      # Data, feature value, and test ML result analysis
│   ├── llm_analyst.py          # Ollama Qwen executive briefing
│   └── session.py              # Session state key registry
├── data/
│   ├── train.csv
│   └── test.csv
├── .cursorrules                # Cursor rules entrypoint
├── .cursor/rules/              # Modular Cursor engineering rules
└── requirements.txt
```

---

## Cursor rules

Engineering standards from [Cursorrules-Shad](https://github.com/ShadAhammed/Cursorrules-Shad) are applied under `.cursor/rules/`. Re-apply after updates:

```bash
pip install git+https://github.com/ShadAhammed/Cursorrules-Shad.git
crs apply
```

Restart Cursor after applying.

---

## License

MIT - Built by Abu Shad Ahammed
