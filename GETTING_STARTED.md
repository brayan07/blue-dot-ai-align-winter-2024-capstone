# Getting Started

This guide walks you through setting up the project from scratch so you can run debate experiments and view the results. **No Google Cloud credentials or Kedro knowledge required** — a setup script configures everything for local storage.

## Prerequisites

- **Python 3.9+** (3.10 recommended; the Docker image uses 3.10)
- **An OpenAI API key** with access to `gpt-4o-mini` (used by the debate agents and judge)
- **curl** (for downloading the dataset)
- **Docker & Docker Compose** (optional, only needed to run the debate transcript viewer locally via Docker)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/brayan07/blue-dot-ai-align-winter-2024-capstone.git
cd blue-dot-ai-align-winter-2024-capstone
```

### 2. Run the setup script

This downloads the [QuALITY dataset](https://github.com/nyu-mll/quality) and creates local configuration files that store data on your filesystem (bypassing the GCS bucket used in the original experiments):

```bash
bash scripts/setup_local.sh
```

### 3. Set your OpenAI API key

Edit `debate-for-ai-alignment/conf/local/parameters.yml` (created by the setup script) and replace the placeholder with your key:

```yaml
llm_config:
  model: gpt-4o-mini
  api_key: "sk-..."
```

Alternatively, set the `OPENAI_API_KEY` environment variable and use OmegaConf's resolver:

```yaml
llm_config:
  model: gpt-4o-mini
  api_key: "${oc.env:OPENAI_API_KEY}"
```

> **Note:** `conf/local/` is gitignored, so your credentials will not be committed.

### 4. Create a virtual environment and install dependencies

```bash
cd debate-for-ai-alignment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e '.[dev]'
```

### 5. Run the preprocessing pipeline

This filters the QuALITY dataset down to the 146 questions used in the experiments and partitions them by article:

```bash
kedro run --pipeline preprocessing
```

### 6. Run a single experiment

Each debate protocol is tagged in the Kedro pipeline. Start with a fast one:

```bash
kedro run --tags naive
```

Available protocol tags:

| Tag | Protocol | Description |
|-----|----------|-------------|
| `naive` | Naive Judge | Judge sees only the article title, no body |
| `expert` | Expert Judge | Judge sees both title and body |
| `consultancy` | Consultancy | Judge interacts with a single consultant agent |
| `unstructured_debate` | Unstructured Debate | Two agents debate in free-form text |
| `structured_debate` | Structured Debate | Two agents debate with structured responses |

To run all protocols at once:

```bash
kedro run
```

> **Cost note:** Running all protocols processes 97 question sets across 5 protocols (~2,300 trials), each making multiple `gpt-4o-mini` API calls. A full run may take several hours and cost significant API credits. Start with a single protocol to verify your setup works.

Results are saved to `data/02_intermediate/unique_set_{protocol}_results/`.

### 7. View results

#### Hosted web app (no setup required)

The debate transcripts from the original experiments are available at:
**https://quality-data-debate-app.onrender.com/**

#### Run the transcript viewer locally with Docker

From the repository root:

```bash
docker compose up --build
```

Then open http://localhost:8050 in your browser.

#### Run the transcript viewer without Docker

```bash
# From the repository root, with your .venv activated
pip install -r debate-ui/requirements.txt
cd debate-ui
python -m src.app
```

Then open http://localhost:8050.

### 8. Explore the analysis notebooks

```bash
cd debate-for-ai-alignment
kedro jupyter lab
```

- `notebooks/ReportAnalysis.ipynb` — generates the charts and statistical analysis shown in the README
- `notebooks/simple_debate.ipynb` — walkthrough of a single debate interaction
- `notebooks/scratch_notebook.ipynb` — scratchpad for ad-hoc exploration

## For Kedro Users

If you're already familiar with Kedro, here's what you need to know:

- **Kedro version:** 0.19.10
- **Project root:** `debate-for-ai-alignment/` (this is the Kedro project directory)
- **Pipelines:** `preprocessing` (data filtering) and `debate` (all five protocols)
- **Catalog:** `conf/base/catalog.yml` uses GCS paths; `scripts/setup_local.sh` creates a `conf/local/catalog.yml` override for local filesystem storage. If you have GCS credentials, you can skip the setup script and place a `credentials.yml` in `conf/local/` with a `gcp_data_sa` key instead.
- **Parameters:** `conf/base/parameters.yml` (model config, data paths)
- **Kedro-Viz:** Enabled. Run `kedro viz run` to visualize the pipeline DAG.

Standard Kedro commands work as expected:

```bash
cd debate-for-ai-alignment
kedro run                          # Run all pipelines
kedro run --tags naive             # Run only the naive judge protocol
kedro run --pipeline preprocessing # Run only data preprocessing
kedro viz run                      # Launch the pipeline visualization
pytest                             # Run tests (requires dev dependencies)
kedro jupyter lab                  # Open JupyterLab with Kedro context
```

## Project Structure

```
blue-dot-ai-align-winter-2024-capstone/
├── README.md                          # Research findings and results
├── GETTING_STARTED.md                 # This file
├── docker-compose.yml                 # Docker setup for the transcript viewer
├── scripts/
│   └── setup_local.sh                 # Local setup (downloads data, no GCS needed)
├── debate-for-ai-alignment/           # Kedro project (experiments)
│   ├── conf/
│   │   ├── base/
│   │   │   ├── catalog.yml            # Data catalog (GCS datasets)
│   │   │   └── parameters.yml         # Model config and parameters
│   │   └── local/                     # Local overrides (gitignored, created by setup script)
│   ├── src/debate_for_ai_alignment/
│   │   ├── pipelines/
│   │   │   ├── preprocessing/         # QuALITY dataset filtering
│   │   │   └── debate/                # Debate protocol implementations
│   │   │       ├── naive.py           # Naive judge (title only)
│   │   │       ├── expert.py          # Expert judge (title + body)
│   │   │       ├── consultancy.py     # Single-agent consultancy
│   │   │       ├── unstructured_debate.py  # Free-form two-agent debate
│   │   │       └── structured_debate/ # Structured two-agent debate
│   │   ├── pipeline_registry.py
│   │   └── settings.py
│   ├── notebooks/                     # Analysis notebooks
│   ├── requirements.txt               # Python dependencies
│   └── pyproject.toml
└── debate-ui/                         # Dash web app for viewing transcripts
    ├── src/
    │   ├── app.py                     # Main Dash application
    │   └── data_manager.py            # Data loading utilities
    ├── docker/
    │   └── Dockerfile
    └── requirements.txt
```

## Troubleshooting

**`kedro run` fails with credential or GCS errors**
- Make sure you ran `bash scripts/setup_local.sh` first — it creates a local catalog that stores data on your filesystem instead of GCS.

**OpenAI API errors**
- Check that your API key is set correctly in `conf/local/parameters.yml` or via the `OPENAI_API_KEY` environment variable.
- Ensure your OpenAI account has access to `gpt-4o-mini`.

**`ModuleNotFoundError: kedro_viz.integrations.kedro.sqlite_store`**
- This happens with `kedro-viz>=11.0`. The `requirements.txt` pins `kedro-viz>=6.7.0,<11.0` to avoid this. If you hit it, run `pip install 'kedro-viz>=6.7.0,<11.0'`.

**Import errors when running the UI**
- Make sure you installed the Kedro project first (`pip install -e ./debate-for-ai-alignment`) before installing the UI requirements. The UI depends on the main project package.

**Missing dataset**
- If preprocessing fails, verify the dataset was downloaded:
  ```bash
  ls debate-for-ai-alignment/data/QuALITY.v1.0.1/QuALITY.v1.0.1.htmlstripped.train
  ```
  If not, re-run `bash scripts/setup_local.sh`.
