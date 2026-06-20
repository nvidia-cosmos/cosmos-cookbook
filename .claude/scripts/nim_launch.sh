#!/bin/bash
# NIM launch script for any Cosmos Reason 2 NIM (2B / 8B / 32B) on any host.
# Run as: bash nim_launch.sh <NGC_API_KEY> [HF_TOKEN]
# Env overrides:
#   MODEL            — short id: cosmos-reason2-8b (default) | cosmos-reason2-2b | cosmos-reason2-32b
#   IMAGE            — full image override (defaults to nvcr.io/nim/nvidia/$MODEL:latest)
#   PORT             — host port (default 8000)
#   CONTAINER_NAME   — docker container name (default cosmos-nim)
#   LOCAL_NIM_CACHE  — host cache dir (default $HOME/.cache/nim)
#   MAX_WAIT         — seconds to wait for /v1/models (default 1800; first-pull + first-load can be long)
#   SHM_SIZE         — --shm-size value (default 32GB)
#
# Output: detached container on $PORT serving OpenAI-compatible API at http://localhost:$PORT/v1.
# Logs streamed to /tmp/nim_launch.log; container logs via `docker logs $CONTAINER_NAME`.

set -e
NGC_API_KEY="${1:-${NGC_API_KEY:-}}"
HF_TOKEN="${2:-${HF_TOKEN:-}}"

MODEL="${MODEL:-cosmos-reason2-8b}"
IMAGE="${IMAGE:-nvcr.io/nim/nvidia/${MODEL}:latest}"
PORT="${PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-cosmos-nim}"
LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"
MAX_WAIT="${MAX_WAIT:-1800}"
SHM_SIZE="${SHM_SIZE:-32GB}"

LOG=/tmp/nim_launch.log
exec > >(tee -a "$LOG") 2>&1

echo "=== NIM launch $(date) ==="
echo "MODEL=$MODEL"
echo "IMAGE=$IMAGE"
echo "PORT=$PORT"
echo "CONTAINER_NAME=$CONTAINER_NAME"
echo "LOCAL_NIM_CACHE=$LOCAL_NIM_CACHE"
echo "HOME=$HOME"

if [ -z "$NGC_API_KEY" ]; then
    echo "ERROR: NGC_API_KEY required as first argument or env var"
    echo "Usage: bash nim_launch.sh <NGC_API_KEY> [HF_TOKEN]"
    exit 1
fi

# 0. Idempotency: if a container of this name is already running AND /v1/models
#    is healthy, reuse it. This protects in-progress first-run model downloads.
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "Container $CONTAINER_NAME is already running and healthy on port $PORT — reusing."
        echo "API endpoint: http://localhost:${PORT}/v1"
        exit 0
    fi
    echo "Container $CONTAINER_NAME is running but /v1/models not yet ready — leaving it alone."
    echo "If you want to force a restart: docker rm -f $CONTAINER_NAME && rerun this script."
    # Fall through to wait loop on the existing container.
    SKIP_LAUNCH=1
fi

# 1. Authenticate with nvcr.io (idempotent)
if [ -z "${SKIP_LAUNCH:-}" ]; then
    echo "Logging into nvcr.io..."
    echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
fi

# 2. Pull image (resumes if already pulled)
if [ -z "${SKIP_LAUNCH:-}" ]; then
    echo "Pulling $IMAGE (first pull ~10-30 min depending on model size)..."
    docker pull "$IMAGE"
fi

# 3. Stop + remove any existing container with the same name (only if we are launching fresh)
if [ -z "${SKIP_LAUNCH:-}" ]; then
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
fi

# 4. Ensure NIM cache dir exists with proper perms
mkdir -p "$LOCAL_NIM_CACHE"

if [ -z "${SKIP_LAUNCH:-}" ]; then
    # 5. Free any existing process on $PORT (best-effort, ignores failure)
    if command -v fuser &>/dev/null; then
        fuser -k "${PORT}/tcp" 2>/dev/null || true
    fi

    # 6. Launch container in detached mode (mirrors official docker run from build.nvidia.com)
    echo "Starting NIM container on port $PORT..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --gpus all \
        --ipc host \
        --shm-size="$SHM_SIZE" \
        -e NGC_API_KEY="$NGC_API_KEY" \
        -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
        -u "$(id -u)" \
        -p "${PORT}:8000" \
        "$IMAGE"
fi

# 7. Wait for /v1/models (model download + load can be lengthy on first run)
echo "Waiting for NIM /v1/models on http://localhost:${PORT} (max ${MAX_WAIT}s)..."
ELAPSED=0
INTERVAL=10
READY=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    BODY=$(curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null || echo "")
    if [ -n "$BODY" ]; then
        echo "NIM is ready!"
        echo "$BODY"
        READY=1
        break
    fi
    # Check container is still running
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        echo "ERROR: container $CONTAINER_NAME exited prematurely. Last logs:"
        docker logs --tail 80 "$CONTAINER_NAME" 2>&1 || true
        exit 2
    fi
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "  ...still waiting (${ELAPSED}s elapsed)"
    fi
done

if [ $READY -eq 0 ]; then
    echo "ERROR: NIM did not respond within ${MAX_WAIT}s. Container logs:"
    docker logs --tail 100 "$CONTAINER_NAME" 2>&1 || true
    exit 3
fi

echo "=== NIM launch complete ==="
echo "API endpoint: http://localhost:${PORT}/v1"
echo "Test:  curl http://localhost:${PORT}/v1/models"
echo "Logs:  docker logs -f $CONTAINER_NAME"
echo ""
echo "To launch Gradio with NIM backend:"
echo "  INFERENCE_BACKEND=nim_local VLLM_BASE_URL=http://localhost:${PORT}/v1 python3 /tmp/gradio_cr2_byo.py"
