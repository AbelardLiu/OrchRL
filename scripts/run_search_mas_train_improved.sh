#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/orchrl/config/search"

# Use the role-share configuration (3 agents share 1 model with 2-GPU model parallel)
DEFAULT_CONFIG_NAME="search_mas_nosearch_external_roleshare_50step"
DEFAULT_CUDA_VISIBLE_DEVICES="3,4"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_LOG_PATH="$REPO_ROOT/logs/search_mas_train_improved_${TIMESTAMP}.log"
DEFAULT_ARCHIVE_ROOT="$REPO_ROOT/logs/archives"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
CONFIG_NAME="${CONFIG_NAME:-$DEFAULT_CONFIG_NAME}"
LOG_PATH="${LOG_PATH:-$DEFAULT_LOG_PATH}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$DEFAULT_ARCHIVE_ROOT}"

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$ARCHIVE_ROOT"

CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

source "$REPO_ROOT/scripts/utils/export_repo_pythonpath.sh"

if ! eval "$(CONFIG_DIR="$CONFIG_DIR" CONFIG_NAME="$CONFIG_NAME" python3 - <<'PY'
import os
import shlex
from hydra import compose, initialize_config_dir

config_dir = os.environ['CONFIG_DIR']
config_name = os.environ['CONFIG_NAME']
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name=config_name)

values = {
    'MAS_WORK_DIR': cfg.training.mate.mas_work_dir,
    'CONFIG_TEMPLATE_PATH': cfg.training.mate.config_template_path,
    'PROMPT_DATA_PATH': cfg.training.mate.prompt_loader.path,
}

# Handle both role-share (single model) and multi-model configurations
if hasattr(cfg.base_models, 'shared_policy'):
    # Role-share mode: single shared model
    values['MODEL_PATH_0'] = cfg.base_models.shared_policy.path
else:
    # Multi-model mode: separate models for each agent
    values['MODEL_PATH_0'] = cfg.base_models.policy_0.path
    if hasattr(cfg.base_models, 'policy_1'):
        values['MODEL_PATH_1'] = cfg.base_models.policy_1.path
    if hasattr(cfg.base_models, 'policy_2'):
        values['MODEL_PATH_2'] = cfg.base_models.policy_2.path

for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"; then
  echo "[ERROR] Failed to resolve runtime paths from Hydra config: $CONFIG_NAME" >&2
  exit 1
fi

for required_dir in "$MAS_WORK_DIR"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "[ERROR] Required directory not found: $required_dir" >&2
    exit 1
  fi
done

for required_file in "$CONFIG_TEMPLATE_PATH" "$PROMPT_DATA_PATH" "$MODEL_PATH_0"; do
  if [[ ! -e "$required_file" ]]; then
    echo "[ERROR] Required path not found: $required_file" >&2
    exit 1
  fi
done

# Check optional model paths (for multi-model mode)
if [[ -n "${MODEL_PATH_1:-}" ]] && [[ ! -e "$MODEL_PATH_1" ]]; then
  echo "[ERROR] Model path not found: $MODEL_PATH_1" >&2
  exit 1
fi
if [[ -n "${MODEL_PATH_2:-}" ]] && [[ ! -e "$MODEL_PATH_2" ]]; then
  echo "[ERROR] Model path not found: $MODEL_PATH_2" >&2
  exit 1
fi

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export MAS_ARCHIVE_ROOT="$ARCHIVE_ROOT"

# Export Search MAS runtime environment variables
export SEARCH_MAS_LLM_BASE_URL="${SEARCH_MAS_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export SEARCH_MAS_LLM_API_KEY="${SEARCH_MAS_LLM_API_KEY:-empty}"
export SEARCH_MAS_LLM_MODEL="${SEARCH_MAS_LLM_MODEL:-/data1/lll/models/Qwen3-4B-Instruct-2507}"
export SEARCH_MAS_RETRIEVAL_SERVICE_URL="${SEARCH_MAS_RETRIEVAL_SERVICE_URL:-http://8.92.9.155:18010/retrieve}"

cd "$REPO_ROOT"

echo "=========================================="
echo "🚀 Role-Share Training Configuration"
echo "=========================================="
echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Config: $CONFIG_NAME"
echo "[INFO] Mode: Role-Share (3 agents share 1 model)"
echo "[INFO] Model Parallel: 2-GPU tensor parallelism"
echo "[INFO] Training steps: 50 (improved from 5)"
echo "[INFO] Batch size: 8 (2x from baseline)"
echo "[INFO] Samples per step: 16 (4x from baseline)"
echo "[INFO] Validation frequency: every 10 steps"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] Log path: $LOG_PATH"
echo "[INFO] Archive root: $ARCHIVE_ROOT"
echo "[INFO] MAS work dir: $MAS_WORK_DIR"
echo "[INFO] Prompt data: $PROMPT_DATA_PATH"
echo "[INFO] Model path: $MODEL_PATH_0"
echo ""
echo "🎯 Key Improvements:"
echo "  ✓ Training steps: 5 → 50 (10x)"
echo "  ✓ Sample num: 4 → 16 per step (4x)"
echo "  ✓ Batch size: 4 → 8 (2x)"
echo "  ✓ Role-share: 3 models → 1 shared model"
echo "  ✓ Model parallel: 2-GPU tensor parallelism"
echo "  ✓ Memory efficient: ~50% less GPU memory"
echo "  ✓ Partial credit reward (0.0 - 1.0)"
echo "  ✓ Enhanced prompts (reduced hallucinations)"
echo "=========================================="
echo ""

python3 -m orchrl.trainer.train \
  --config-path "$CONFIG_DIR" \
  --config-name "$CONFIG_NAME" 2>&1 | tee "$LOG_PATH"
