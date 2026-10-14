#!/bin/bash
set -e

# ---------------------------------------------------------------
# Create workspace on shared filesystem
# ---------------------------------------------------------------
SLURM_LOG_BASE="{slurm_log_dir}"
mkdir -p "${{SLURM_LOG_BASE}}"
WORK_DIR=$(mktemp -d "${{SLURM_LOG_BASE}}/job_XXXXXX")
SBATCH_FILE="${{WORK_DIR}}/job.sbatch"

# ---------------------------------------------------------------
# Write inner sbatch script
# ---------------------------------------------------------------
cat > "${{SBATCH_FILE}}" << 'SBATCH_SCRIPT_EOF'
{sbatch_script_content}
SBATCH_SCRIPT_EOF

# Replace the log-dir placeholder with the actual work directory
sed -i "s|LOG_DIR_PLACEHOLDER|${{WORK_DIR}}|g" "${{SBATCH_FILE}}"

# ---------------------------------------------------------------
# Submit the job
# ---------------------------------------------------------------
SBATCH_OUTPUT=$(sbatch "${{SBATCH_FILE}}" 2>&1)
echo "${{SBATCH_OUTPUT}}"

JOB_ID=$(echo "${{SBATCH_OUTPUT}}" | grep -oP 'Submitted batch job \K\d+' || true)
if [[ -z "${{JOB_ID}}" ]]; then
    echo "[ERROR] Failed to submit sbatch job" >&2
    cat "${{SBATCH_FILE}}" >&2
    exit 1
fi
echo "Slurm job ID: ${{JOB_ID}}"

# ---------------------------------------------------------------
# Trap: cancel the Slurm job if this wrapper is killed
# ---------------------------------------------------------------
cleanup_wrapper() {{
    scancel "${{JOB_ID}}" 2>/dev/null || true
    kill 0 2>/dev/null || true
}}
trap cleanup_wrapper SIGTERM SIGINT

# ---------------------------------------------------------------
# Wait for the log file to appear
# ---------------------------------------------------------------
LOG_FILE="${{WORK_DIR}}/slurm_${{JOB_ID}}.out"
ERR_FILE="${{WORK_DIR}}/slurm_${{JOB_ID}}.err"

WAIT_COUNT=0
MAX_WAIT=600  # 20 minutes max wait for job to start
while [[ ! -f "${{LOG_FILE}}" ]] && [[ ${{WAIT_COUNT}} -lt ${{MAX_WAIT}} ]]; do
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 1))
    JOB_STATE=$(squeue -j "${{JOB_ID}}" -h -o "%T" 2>/dev/null || echo "UNKNOWN")
    if [[ -z "${{JOB_STATE}}" ]]; then
        echo "[WARN] Job ${{JOB_ID}} is no longer in queue before output appeared" >&2
        sleep 2
        break
    fi
    if (( WAIT_COUNT % 15 == 0 )); then
        echo "[INFO] Waiting for Slurm job ${{JOB_ID}} to start (state: ${{JOB_STATE}}, waited $((WAIT_COUNT * 2))s)..."
    fi
done

if [[ ! -f "${{LOG_FILE}}" ]]; then
    echo "[WARN] Log file ${{LOG_FILE}} not found after waiting" >&2
    JOB_STATE=$(squeue -j "${{JOB_ID}}" -h -o "%T" 2>/dev/null || echo "")
    if [[ -z "${{JOB_STATE}}" ]]; then
        echo "[ERROR] Slurm job ${{JOB_ID}} exited before producing output" >&2
        scontrol show job "${{JOB_ID}}" 2>/dev/null || true
        exit 1
    fi
fi

# ---------------------------------------------------------------
# Stream logs via tail -f
# ---------------------------------------------------------------
TAIL_PID=""
TAIL_ERR_PID=""

if [[ -f "${{LOG_FILE}}" ]]; then
    tail -f "${{LOG_FILE}}" &
    TAIL_PID=$!
fi

if [[ -f "${{ERR_FILE}}" ]]; then
    tail -f "${{ERR_FILE}}" >&2 &
    TAIL_ERR_PID=$!
fi

# ---------------------------------------------------------------
# Poll for job completion
# ---------------------------------------------------------------
while true; do
    JOB_STATE=$(squeue -j "${{JOB_ID}}" -h -o "%T" 2>/dev/null || echo "")
    if [[ -z "${{JOB_STATE}}" ]]; then
        break
    fi
    case "${{JOB_STATE}}" in
        PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|SUSPENDED)
            sleep 5
            ;;
        *)
            break
            ;;
    esac
done

# Let tail catch up with final output
sleep 3

# Kill tail processes
if [[ -n "${{TAIL_PID}}" ]]; then
    kill "${{TAIL_PID}}" 2>/dev/null || true
    wait "${{TAIL_PID}}" 2>/dev/null || true
fi
if [[ -n "${{TAIL_ERR_PID}}" ]]; then
    kill "${{TAIL_ERR_PID}}" 2>/dev/null || true
    wait "${{TAIL_ERR_PID}}" 2>/dev/null || true
fi

# ---------------------------------------------------------------
# Determine exit code from Slurm job
# ---------------------------------------------------------------
EXIT_CODE=$(scontrol show job "${{JOB_ID}}" 2>/dev/null | grep -oP 'ExitCode=\K\d+' | head -1 || echo "1")
EXIT_CODE=${{EXIT_CODE:-1}}

echo ""
echo "Slurm job ${{JOB_ID}} finished with exit code ${{EXIT_CODE}}"

# Clear the trap so we don't scancel a completed job
trap - SIGTERM SIGINT

exit ${{EXIT_CODE}}
