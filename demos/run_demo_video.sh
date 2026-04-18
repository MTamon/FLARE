#!/usr/bin/env bash
# デモランチャ: 動画ファイルでの FlashAvatar レンダリングテスト
#
# 実行例:
#   bash demos/run_demo_video.sh \
#       --input_video data/raw/sample.mp4 \
#       --checkpoint_dir checkpoints/flashavatar/person01
#
#   bash demos/run_demo_video.sh \
#       --input_video data/raw/sample.mp4 \
#       --extractor deca \
#       --checkpoint_dir checkpoints/flashavatar/person01

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# shellcheck disable=SC1091
. "$HERE/_env.sh"

cd "$ROOT"
exec python "$HERE/demo_video_render.py" "$@"
