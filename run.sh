#!/usr/bin/env bash
# =============================================================================
# run.sh — NDVI Pipeline Docker helper
# =============================================================================
set -euo pipefail

IMAGE="ndvi-pipeline:latest"

cmd="${1:-help}"
province="${2:-}"

_latest_log() {
    local prov="$1"
    local env_file="envs/${prov}.env"
    if [[ ! -f "$env_file" ]]; then
        echo "env file not found: $env_file" >&2; exit 1
    fi
    local output_dir
    output_dir=$(grep '^OUTPUT_DIR=' "$env_file" | cut -d= -f2-)
    local log_dir="${output_dir}/${prov}/logs"
    ls -t "${log_dir}"/ndvi_*.log 2>/dev/null | head -1 || echo ""
}

case "$cmd" in

  build)
    echo ">>> Building image: $IMAGE"
    docker build -t "$IMAGE" .
    echo ">>> Size: $(docker image inspect "$IMAGE" --format='{{.Size}}' | awk '{printf "%.1f GB\n", $1/1024/1024/1024}')"
    ;;

  run)
    [[ -z "$province" ]] && { echo "Usage: $0 run <province>"; exit 1; }
    ENV_FILE="envs/${province}.env"
    [[ ! -f "$ENV_FILE" ]] && { echo "Not found: $ENV_FILE"; exit 1; }

    CONTAINER="ndvi_${province}"
    docker rm -f "$CONTAINER" 2>/dev/null || true

    echo ">>> Running province=${province}  container=${CONTAINER}"
    docker run -d \
      --name "$CONTAINER" \
      --env-file "$ENV_FILE" \
      --shm-size=2g \
      --memory=120g \
      -v "$(pwd)":/workspaces \
      -v /fs1:/workspaces/fs1 \
      -v /fs2:/workspaces/fs2 \
      -v /fs3:/workspaces/fs3 \
      -v /fs4:/workspaces/fs4 \
      -v /fs5:/workspaces/fs5 \
      -v /fs6:/workspaces/fs6 \
      -v /fs7:/workspaces/fs7 \
      -v /fs8:/workspaces/fs8 \
      -e PYTHONUNBUFFERED=1 \
      -e GDAL_CACHEMAX=512 \
      -e GDAL_NUM_THREADS=4 \
      -e OGR_SQLITE_SYNCHRONOUS=OFF \
      "$IMAGE" \
      python3 /workspaces/ndvi_pipeline.py

    echo ">>> Logs: ./run.sh tail ${province}"
    ;;

  shell)
    [[ -z "$province" ]] && { echo "Usage: $0 shell <province>"; exit 1; }
    ENV_FILE="envs/${province}.env"
    [[ ! -f "$ENV_FILE" ]] && { echo "Not found: $ENV_FILE"; exit 1; }

    echo ">>> Opening shell for province=${province}"
    docker run -it --rm \
      --env-file "$ENV_FILE" \
      --shm-size=2g \
      -v "$(pwd)":/workspaces \
      -v /fs2:/workspaces/fs2 \
      -e PYTHONUNBUFFERED=1 \
      "$IMAGE" bash
    ;;

  status)
    echo ">>> Running NDVI containers:"
    docker ps --filter "name=ndvi_" --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}"
    ;;

  tail)
    [[ -z "$province" ]] && { echo "Usage: $0 tail <province>"; exit 1; }
    LOG=$(_latest_log "$province")
    [[ -z "$LOG" ]] && { echo "No log file found for province=${province}"; exit 1; }
    echo ">>> Tailing: $LOG"
    tail -f "$LOG"
    ;;

  stop)
    [[ -z "$province" ]] && { echo "Usage: $0 stop <province>"; exit 1; }
    CONTAINER="ndvi_${province}"
    echo ">>> Stopping: $CONTAINER"
    docker stop "$CONTAINER" && docker rm "$CONTAINER"
    ;;

  help|*)
    echo "Usage: $0 {build|run|shell|status|tail|stop} [province]"
    echo ""
    echo "Available env files:"
    ls envs/*.env 2>/dev/null | sed 's|envs/||;s|.env||' | xargs -I{} echo "  {}"
    ;;

esac