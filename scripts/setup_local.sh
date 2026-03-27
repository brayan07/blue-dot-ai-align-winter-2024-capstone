#!/usr/bin/env bash
# Setup script for local development (no GCS credentials required).
# Run from the repository root: bash scripts/setup_local.sh

set -euo pipefail

KEDRO_DIR="debate-for-ai-alignment"
DATA_DIR="${KEDRO_DIR}/data/QuALITY.v1.0.1"
CONF_LOCAL="${KEDRO_DIR}/conf/local"
QUALITY_URL="https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.train"

echo "=== Setting up local environment ==="

# 1. Download QuALITY dataset
echo ""
echo "--- Downloading QuALITY dataset ---"
mkdir -p "${DATA_DIR}"
if [ -f "${DATA_DIR}/QuALITY.v1.0.1.htmlstripped.train" ]; then
    echo "Dataset already exists, skipping download."
else
    curl -L -o "${DATA_DIR}/QuALITY.v1.0.1.htmlstripped.train" "${QUALITY_URL}"
    echo "Downloaded QuALITY dataset (~11 MB)."
fi

# 2. Create local catalog (overrides GCS paths with local filesystem)
echo ""
echo "--- Creating local catalog ---"
mkdir -p "${CONF_LOCAL}"
cat > "${CONF_LOCAL}/catalog.yml" << 'CATALOG'
# Local catalog — overrides GCS paths from conf/base/catalog.yml
# so you can run experiments without Google Cloud credentials.

_local_data_root: "data"

quality_filtered_train:
  type: json.JSONDataset
  filepath: "${_local_data_root}/01_raw/quality_filtered_train.json"

quality_filtered_train_with_direct_answers:
  type: pandas.CSVDataset
  filepath: "${_local_data_root}/02_intermediate/quality_filtered_train_with_direct_answers.json"

quality_filtered_train_answers_with_biases:
  type: pandas.CSVDataset
  filepath: "${_local_data_root}/02_intermediate/quality_filtered_train_answers_with_biases.csv"

partitioned_quality_filtered_train:
  type: partitions.PartitionedDataset
  path: "${_local_data_root}/01_raw/quality_filtered_train_partitioned"
  dataset: json.JSONDataset

quality_filtered_train_{set_unique_id}:
  type: json.JSONDataset
  filepath: "${_local_data_root}/01_raw/quality_filtered_train_partitioned/{set_unique_id}/unique_set.json"

partitioned_unique_set_{kind}_results:
  type: partitions.PartitionedDataset
  path: "${_local_data_root}/02_intermediate/unique_set_{kind}_results"
  dataset: json.JSONDataset

partitioned_unique_set_{kind}_results@incremental:
  type: partitions.IncrementalDataset
  path: "${_local_data_root}/02_intermediate/unique_set_{kind}_results"
  dataset: json.JSONDataset

unique_set_{kind}_results_{set_unique_id}:
  type: json.JSONDataset
  filepath: "${_local_data_root}/02_intermediate/unique_set_{kind}_results/{set_unique_id}/unique_set.json"
CATALOG
echo "Created ${CONF_LOCAL}/catalog.yml"

# 3. Create local parameters (with API key placeholder)
echo ""
echo "--- Creating local parameters ---"
if [ -f "${CONF_LOCAL}/parameters.yml" ]; then
    echo "Local parameters.yml already exists, skipping."
    echo "Make sure your OpenAI API key is set in ${CONF_LOCAL}/parameters.yml"
else
    cat > "${CONF_LOCAL}/parameters.yml" << 'PARAMS'
# Local parameter overrides.
# Set your OpenAI API key here — this file is gitignored.
#
# Option 1: Paste your key directly below.
# Option 2: Set the OPENAI_API_KEY env var and use the OmegaConf resolver:
#   api_key: "${oc.env:OPENAI_API_KEY}"
llm_config:
  model: gpt-4o-mini
  api_key: "sk-your-openai-api-key-here"
PARAMS
    echo "Created ${CONF_LOCAL}/parameters.yml"
    echo "  -> Set your OpenAI API key: either edit the file or export OPENAI_API_KEY"
fi

# 4. Create required data directories
echo ""
echo "--- Creating data directories ---"
mkdir -p "${KEDRO_DIR}/data/01_raw"
mkdir -p "${KEDRO_DIR}/data/02_intermediate"
echo "Data directories ready."

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Set your OpenAI API key:"
echo "     export OPENAI_API_KEY='sk-...'"
echo "     (or edit ${CONF_LOCAL}/parameters.yml directly)"
echo ""
echo "  2. Install dependencies:"
echo "     cd ${KEDRO_DIR}"
echo "     python -m venv .venv && source .venv/bin/activate"
echo "     pip install -e '.[dev]'"
echo ""
echo "  3. Run the preprocessing pipeline:"
echo "     kedro run --pipeline preprocessing"
echo ""
echo "  4. Run a single debate protocol (e.g., naive):"
echo "     kedro run --tags naive"
