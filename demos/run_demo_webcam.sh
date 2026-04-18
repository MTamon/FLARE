#!/usr/bin/env bash
# デモランチャ: Webカメラリアルタイム SMIRK + FlashAvatar
#
# GPU MediaPipe を使う場合は --mp_delegate gpu を追加してください。
# EGL ベンダーピニングは _env.sh が自動設定します。
#
# 実行例:
#   bash demos/run_demo_webcam.sh --checkpoint_dir checkpoints/flashavatar/person01
#   bash demos/run_demo_webcam.sh --mp_delegate gpu --checkpoint_dir checkpoints/flashavatar/person01
#   bash demos/run_demo_webcam.sh --source data/raw/sample.mp4 --checkpoint_dir checkpoints/flashavatar/person01

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# shellcheck disable=SC1091
. "$HERE/_env.sh"

cd "$ROOT"
exec python "$HERE/demo_webcam.py" "$@"
