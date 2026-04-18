#!/usr/bin/env bash
# デモランチャ: バッチ特徴抽出 (SMIRK / DECA)
#
# 実行例:
#   bash demos/run_demo_batch.sh
#   bash demos/run_demo_batch.sh --extractor deca
#   bash demos/run_demo_batch.sh --mp_delegate gpu --overwrite
#   bash demos/run_demo_batch.sh --input_dir data/multimodal_dialogue_formed

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# EGL ベンダーピニング + PYTHONPATH 設定
# shellcheck disable=SC1091
. "$HERE/_env.sh"

cd "$ROOT"
exec python "$HERE/demo_batch_extract.py" "$@"
