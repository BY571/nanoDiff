#!/usr/bin/env bash
# Spark-side autonomous chain runner.
#
# Watches an in-flight training run, and once it finishes CLEANLY launches the
# next two schedule-sweep variants sequentially. Refuses to chain if the
# upstream run crashed, NaN'd, or finished with a clearly broken val loss.
#
# Designed to be launched once via `nohup ... &` and forgotten about. All
# decisions and inputs/outputs are logged to AUTO_LOG so you can audit it.
#
# Usage:
#     nohup bash scripts/auto_chain_sweep.sh <upstream_pid> > /dev/null 2>&1 &
set -u  # not -e: we want to keep going past command failures and log them

UPSTREAM_PID="${1:?usage: auto_chain_sweep.sh <upstream_pid>}"
LOG_DIR="/tmp/nd_spark"
AUTO_LOG="$LOG_DIR/auto_sweep.log"
UPSTREAM_LOG="$LOG_DIR/train_50m_v2.log"
COSINE_LOG="$LOG_DIR/train_50m_v2_cosine.log"
LONGDECAY_LOG="$LOG_DIR/train_50m_v2_long_decay.log"
REPO_DIR="$HOME/nanoDiff"

# Sanity bounds on final val. Higher than 4.5 = something is wrong, do not chain.
# Lower than 3.5 = something is very wrong (or we got lucky); still chain but flag.
MAX_OK_VAL=4.5

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$AUTO_LOG"
}

wait_for_pid() {
    local pid="$1"
    local label="$2"
    log "waiting for $label PID=$pid to terminate..."
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    log "$label PID=$pid terminated"
}

verify_clean_completion() {
    local log_path="$1"
    local label="$2"
    local ckpt_path="$3"

    if ! grep -qE '^done\. best val loss' "$log_path"; then
        log "REFUSE: $label log has no 'done. best val loss' marker (probably crashed)"
        return 1
    fi
    if [ ! -f "$ckpt_path" ]; then
        log "REFUSE: $label ckpt_final.pt missing at $ckpt_path"
        return 1
    fi
    local val
    val=$(grep -E '^done\. best val loss' "$log_path" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [ -z "$val" ]; then
        log "REFUSE: $label could not parse final val from log"
        return 1
    fi
    log "$label finished with best val loss = $val"
    # awk handles float comparison portably
    if awk -v v="$val" -v m="$MAX_OK_VAL" 'BEGIN{exit !(v <= m)}'; then
        return 0
    else
        log "REFUSE: $label final val $val exceeds sanity max $MAX_OK_VAL"
        return 1
    fi
}

launch_run() {
    local config="$1"
    local log_path="$2"
    local label="$3"

    log "launching $label  (config=$config)"
    cd "$REPO_DIR" || { log "FATAL: cannot cd to $REPO_DIR"; return 1; }
    rm -f "$log_path"
    env NANODIFF_WANDB=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HF_HOME=/tmp/hf_cache_spark \
        .venv/bin/python -u train.py --config "$config" > "$log_path" 2>&1
    local rc=$?
    log "$label exited with code $rc"
    return $rc
}

# --- main ---
log "===== auto_chain_sweep.sh started, watching upstream PID=$UPSTREAM_PID ====="

wait_for_pid "$UPSTREAM_PID" "v2 baseline (50m_v2)"

if ! verify_clean_completion "$UPSTREAM_LOG" "v2 baseline" \
        "$REPO_DIR/checkpoints/50m_v2_spark/ckpt_final.pt"; then
    log "v2 baseline did NOT pass sanity checks; skipping sweep"
    log "===== auto_chain_sweep.sh aborted ====="
    exit 1
fi
log "v2 baseline passed sanity checks; proceeding to sweep"

# Run #1: cosine
launch_run "configs/train_50m_v2_cosine_spark.py" "$COSINE_LOG" "cosine variant"
verify_clean_completion "$COSINE_LOG" "cosine variant" \
    "$REPO_DIR/checkpoints/50m_v2_cosine_spark/ckpt_final.pt" || true

# Run #2: WSD long-decay (run even if cosine had issues -- they are independent)
launch_run "configs/train_50m_v2_long_decay_spark.py" "$LONGDECAY_LOG" "long-decay variant"
verify_clean_completion "$LONGDECAY_LOG" "long-decay variant" \
    "$REPO_DIR/checkpoints/50m_v2_long_decay_spark/ckpt_final.pt" || true

log "===== auto_chain_sweep.sh complete ====="
