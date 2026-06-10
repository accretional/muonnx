#!/usr/bin/env bash
# prep_embed.sh — provision the bits a self-contained muonnx binary embeds.
#
# muonnx's embed convention (see docs/runtime.md):
#   - Model weights live INSIDE each model package: models/<pkg>/onnx/<name>.onnx
#     (fp32) and models/<pkg>/onnx/fp16/<name>.onnx (fp16). A model too big to
#     embed (go:embed caps at 2 GB/file) ships only in the /onnx weight source at
#     runtime — Resolve falls back to it automatically.
#   - The ORT shared library is staged into src/muonnx/ort/ so a future fat build
#     can go:embed it; until then muonnx.Init discovers it on disk.
#
# Safe to re-run. Adapted from accretional/vad's prep-embed.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# 1. Regenerate the reproducible test models (pure-Go authoring, no Python).
echo "prep_embed: regenerating test models via tools/genmodels"
go run ./tools/genmodels

# 2. Stage the ORT shared library for embedding (best effort).
EMBED_ORT="src/muonnx/ort"
OS=$(uname -s); ARCH=$(uname -m)
case "$OS-$ARCH" in
    Darwin-arm64)  PLATFORM="osx-arm64";    LIB="libonnxruntime.dylib" ;;
    Darwin-x86_64) PLATFORM="osx-x86_64";   LIB="libonnxruntime.dylib" ;;
    Linux-x86_64)  PLATFORM="linux-x64";    LIB="libonnxruntime.so" ;;
    Linux-aarch64) PLATFORM="linux-aarch64";LIB="libonnxruntime.so" ;;
    *) echo "prep_embed: unsupported $OS-$ARCH; skipping ORT stage" >&2; PLATFORM=""; LIB="" ;;
esac

ORT_SRC="${ONNXRUNTIME_LIB:-}"
if [[ -z "$ORT_SRC" && -n "$LIB" ]]; then
    ORT_SRC=$(ls -d third_party/onnxruntime-${PLATFORM}-*/lib/${LIB} 2>/dev/null | head -1 || true)
fi
if [[ -n "$ORT_SRC" && -f "$ORT_SRC" ]]; then
    ORT_REAL=$(python3 -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$ORT_SRC")
    mkdir -p "$EMBED_ORT"
    cp "$ORT_REAL" "$EMBED_ORT/$LIB"
    echo "prep_embed: staged ORT lib $ORT_REAL -> $EMBED_ORT/$LIB ($(du -h "$EMBED_ORT/$LIB" | awk '{print $1}'))"
else
    echo "prep_embed: no ORT lib found (set ONNXRUNTIME_LIB or place third_party/onnxruntime-*); muonnx.Init will discover at runtime"
fi

echo "prep_embed: done"
