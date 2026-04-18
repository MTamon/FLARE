#!/usr/bin/env bash
# デモランチャ: Webカメラリアルタイム DECA + FlashAvatar
#
# GPU MediaPipe を使う場合は --mp_delegate gpu を追加してください。
# EGL ベンダーピニングは _env.sh が自動設定します。
#
# DECA は eyes_pose / eyelids を出力しないため、本ランチャはデフォルトで
# MediaPipe Face Landmarker による補完 (use_mediapipe_supplement=True) を
# 有効にしています。DECA 本来挙動を維持したい場合は --no_eye_supplement を
# 渡してください。
#
# 実行例:
#   bash demos/run_demo_webcam_deca.sh --checkpoint_dir checkpoints/flashavatar/person01
#   bash demos/run_demo_webcam_deca.sh --mp_delegate gpu --checkpoint_dir checkpoints/flashavatar/person01
#   bash demos/run_demo_webcam_deca.sh --source data/raw/sample.mp4 --checkpoint_dir checkpoints/flashavatar/person01

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# shellcheck disable=SC1091
. "$HERE/_env.sh"

cd "$ROOT"
exec python "$HERE/demo_webcam_deca.py" "$@"
