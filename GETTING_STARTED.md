# Getting Started

This guide walks you through setting up the project from scratch so you can run debate experiments and view the results. No prior knowledge of [Kedro](https://kedro.org/) is required.

## Prerequisites

- **Python 3.10+** (3.10 recommended; the Docker image uses 3.10)
- **An OpenAI API key** with access to `gpt-4o-mini` (used by the debate agents and judge)
- **Google Cloud credentials** with read access to the project's GCS data bucket (see [Data Access](#data-access) below)
- **Docker & Docker Compose** (optional, only needed to run the debate transcript viewer locally)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/brayan07/blue-dot-ai-align-winter-2024-capstone.git
cd blue-dot-ai-align-winter-2024-capstone
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the Kedro project (includes all experiment dependencies)
pip install ./debate-for-ai-alignment

# Install dev dependencies (for running analysis notebooks and tests)
pip install -r debate-for-ai-alignment/requirements-dev.txt
```

### 3. Configure your OpenAI API key

Edit `debate-for-ai-alignment/conf/base/parameters.yml` and set your API key:

```yaml
llm_config:
  model: gpt-4o-mini
  api_key: "sk-your-openai-api-key-here"
```

> **Tip:** To avoid accidentally committing your key, you can instead create a local override at `debate-for-ai-alignment/conf/local/parameters.yml` with the same structure. Kedro merges local config on top of base config, and `conf/local/` is gitignored.

### 4. Set up data access

<a id="data-access"></a>

The experiment data (QuALITY dataset subsets and results) is stored in a Google Cloud Storage bucket. You need GCP credentials to read from it.

1. Create a GCP service account (or use an existing one) with **Storage Object Viewer** access to the bucket `blue-dot-ai-align-winter-2024-capstone`.
2. Download the service account key JSON file.
3. Create a credentials file at `debate-for-ai-alignment/conf/local/credentials.yml`:

```yaml
gcp_data_sa:
  type: service_account
  project_id: your-gcp-project-id
  private_key_id: "..."
  private_key: "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
  client_email: "...@...iam.gserviceaccount.com"
  client_id: "..."
  auth_uri: "https://accounts.google.com/o/oauth2/auth"
  token_uri: "https://oauth2.googleapis.com/token"
```

Alternatively, if you have `gcloud` CLI configured with appropriate permissions, you may be able to use application default credentials by setting:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

> **Note:** `conf/local/` is gitignored, so your credentials will not be committed.

### 5. Run a single experiment

Each debate protocol is tagged in the Kedro pipeline. To run a single protocol (e.g., the naive judge baseline):

```bash
cd debate-for-ai-alignment
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

> **Warning:** Running all protocols processes 146 questions across 5 protocols with multiple variations each (2,336 total trials). This will make many OpenAI API calls and may take several hours. Start with a single protocol to verify your setup works.

### 6. View results

#### Option A: Hosted web app (no setup required)

The debate transcripts from the original experiments are available at:
**https://quality-data-debate-app.onrender.com/**

#### Option B: Run the transcript viewer locally with Docker

From the repository root:

```bash
docker compose up --build
```

Then open http://localhost:8050 in your browser.

This builds a container that installs the project, loads results from GCS, and serves a Dash web application where you can browse debate transcripts, compare protocols, and view model confidence scores.

> **Note:** The Docker container also needs GCP credentials. Mount your credentials file or set environment variables as needed. The container looks for credentials at `/etc/secrets/credentials.yml` and `/etc/secrets/parameters.yml` (see `debate-ui/docker/scripts/move_sensitive_files_to_local_conf.sh`).

#### Option C: Run the transcript viewer without Docker

```bash
# From the repository root, with your .venv activated
pip install -r debate-ui/requirements.txt
cd debate-ui
python -m src.app
```

Then open http://localhost:8050.

### 7. Explore the analysis notebooks

Two Jupyter notebooks are included for exploring and analyzing results:

```bash
cd debate-for-ai-alignment

# Launch JupyterLab with Kedro integration
kedro jupyter lab
```

- `notebooks/ReportAnalysis.ipynb` — generates the charts and statistical analysis shown in the README
- `notebooks/scratch_notebook.ipynb` — scratchpad for ad-hoc exploration

## For Kedro Users

If you're already familiar with Kedro, here's what you need to know:

- **Kedro version:** 0.19.10
- **Project root:** `debate-for-ai-alignment/` (this is the Kedro project directory)
- **Pipelines:** `preprocessing` (data filtering) and `debate` (all five protocols)
- **Catalog:** GCS-backed datasets defined in `conf/base/catalog.yml`
- **Parameters:** `conf/base/parameters.yml` (model config, data paths)
- **Credentials:** `conf/local/credentials.yml` (GCP service account, gitignored)
- **Kedro-Viz:** Enabled. Run `kedro viz run` to visualize the pipeline DAG.

Standard Kedro commands work as expected:

```bash
cd debate-for-ai-alignment
kedro run                          # Run all pipelines
kedro run --tags naive             # Run only the naive judge protocol
kedro run --pipeline preprocessing # Run only data preprocessing
kedro viz run                      # Launch the pipeline visualization
kedro test                         # Run tests
kedro jupyter lab                  # Open JupyterLab with Kedro context
```

## Project Structure

```
blue-dot-ai-align-winter-2024-capstone/
├── README.md                          # Research findings and results
├── GETTING_STARTED.md                 # This file
├── docker-compose.yml                 # Docker setup for the transcript viewer
├── debate-for-ai-alignment/           # Kedro project (experiments)
│   ├── conf/
│   │   ├── base/
│   │   │   ├── catalog.yml            # Data catalog (GCS datasets)
│   │   │   └── parameters.yml         # Model config and parameters
│   │   └── local/                     # Local overrides (gitignored)
│   │       └── credentials.yml        # GCP credentials (you create this)
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

**"No credentials found" or GCS access errors**
- Ensure `debate-for-ai-alignment/conf/local/credentials.yml` exists and contains valid GCP service account credentials.
- The credential key in the YAML must be `gcp_data_sa` to match the catalog configuration.

**OpenAI API errors**
- Check that your API key is set correctly in `conf/base/parameters.yml` (or `conf/local/parameters.yml`).
- Ensure your OpenAI account has access to `gpt-4o-mini`.

**Import errors when running the UI**
- Make sure you installed the Kedro project first (`pip install ./debate-for-ai-alignment`) before installing the UI requirements. The UI depends on the main project package.

**Docker build fails**
- The Docker build installs the Kedro project and UI requirements from within the container. Ensure you haven't modified `pyproject.toml` or `requirements.txt` in a way that breaks the install.
