#!/bin/bash
# Build the LLMZero verl 0.9.0 training image.
#
# This wraps `docker build` for docker/Dockerfile.verl. The build compiles a
# large CUDA stack from source (apex, TransformerEngine, flash-attn, DeepEP,
# vLLM, Megatron) and can take well over an hour on a CUDA build host.
#
# Usage:
#   ./docker/build.sh [IMAGE_TAG] [extra docker build args...]
#
# Examples:
#   ./docker/build.sh                          # builds llmzero-verl:0.9.0
#   ./docker/build.sh my-registry/llmzero:0.9.0
#   ./docker/build.sh llmzero-verl:0.9.0 --build-arg APT_MIRROR=http://mirror.example.com/ubuntu
#
# Then point a backend at it, e.g.:
#   llmzero -i ./task -t "..." --compute-backend docker --docker-image llmzero-verl:0.9.0 --num-gpus 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.verl"

IMAGE_TAG="${1:-llmzero-verl:0.9.0}"
shift || true  # remaining args pass through to docker build

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found on PATH." >&2
  exit 1
fi

echo "Building ${IMAGE_TAG} from ${DOCKERFILE}"
echo "This is a heavy from-source build; expect a long build time."

docker build \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_TAG}" \
  "$@" \
  "${SCRIPT_DIR}"

echo
echo "Built ${IMAGE_TAG}"
echo "Run training with: llmzero -i <task> -t '<instruction>' --compute-backend docker --docker-image ${IMAGE_TAG} --num-gpus <N>"
