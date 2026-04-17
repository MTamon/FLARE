#!/bin/bash
# setup_flashavatar.sh
# FlashAvatar (MTamon/FlashAvatar@release/cuda128-fixed) のセットアップスクリプト
#
# 前提条件:
#   - Python 3.11 がアクティブな仮想環境
#   - CUDA Toolkit 12.8 + nvcc がインストール済み
#   - gcc-11 / g++-11 がインストール済み (pytorch3d・diff-gaussian-rasterization ビルドに必要)
#   - git サブモジュールが初期化済み (git submodule update --init third_party/FlashAvatar)
#
# Usage:
#   bash install/setup_flashavatar.sh
#
# 実行内容:
#   1. third_party/FlashAvatar サブモジュールの初期化・更新
#   2. FlashAvatar の CUDA 12.8 対応 install_128.sh を実行
#      - pytorch3d v0.7.8 をソースビルド
#      - diff-gaussian-rasterization (3DGS ラスタライザ) をビルド
#      - simple-knn をビルド
#   3. FlashAvatar の対象人物モデルの学習手順を案内

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLARE_ROOT="$(dirname "$SCRIPT_DIR")"
FA_DIR="$FLARE_ROOT/third_party/FlashAvatar"
FA_CKPT_DIR="$FLARE_ROOT/checkpoints/flashavatar"

echo "=== FlashAvatar (CUDA 12.8) セットアップ ==="
echo "FLARE root     : $FLARE_ROOT"
echo "FlashAvatar dir: $FA_DIR"
echo ""

# ---- 事前チェック ----
echo "[0/3] 事前環境チェック..."

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc が見つかりません。CUDA Toolkit 12.8 をインストールして"
    echo "       nvcc が PATH に含まれていることを確認してください。"
    exit 1
fi
CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "      CUDA バージョン: $CUDA_VER"

if ! command -v gcc-11 &> /dev/null; then
    echo "WARNING: gcc-11 が見つかりません。"
    echo "         Ubuntu の場合: sudo apt-get install gcc-11 g++-11"
    echo "         pytorch3d・diff-gaussian-rasterization のビルドに必要です。"
fi
echo ""

# ---- 1. サブモジュールの初期化・更新 (再帰: diff-gaussian-rasterization, simple-knn 含む) ----
echo "[1/3] サブモジュールの初期化・更新..."
cd "$FLARE_ROOT"
git submodule update --init --recursive third_party/FlashAvatar
echo "      ブランチ: $(cd "$FA_DIR" && git branch --show-current)"
echo "      コミット: $(cd "$FA_DIR" && git rev-parse --short HEAD)"
echo ""

# ---- 2. FlashAvatar の CUDA 12.8 依存パッケージをインストール ----
echo "[2/3] FlashAvatar CUDA 12.8 依存パッケージのインストール..."
cd "$FA_DIR"
if [ ! -f "install_128.sh" ]; then
    echo "ERROR: install_128.sh が見つかりません: $FA_DIR/install_128.sh"
    exit 1
fi
bash install_128.sh
echo ""

# ---- 3. チェックポイントディレクトリの準備と案内 ----
echo "[3/3] FlashAvatar チェックポイントディレクトリの準備..."
mkdir -p "$FA_CKPT_DIR"

echo ""
echo "INFO: FlashAvatar は対象人物ごとに個別学習が必要です。"
echo "      汎用の事前学習済みモデルはありません。"
echo ""
echo "対象人物モデルの学習手順:"
echo "  1. 対象人物のモノクラー正面動画を用意 (最低 600 フレーム推奨)"
echo "  2. FLAME トラッカーで per-frame パラメータを抽出:"
echo "     cd $FA_DIR"
echo "     # metrical-tracker (別途インストール必要) で .frame ファイルを生成"
echo "  3. 意味論的セグメンテーションマスクを準備 (parsing/, alpha/)"
echo "  4. FlashAvatar モデルを学習 (約 30 分 on RTX 3090):"
echo "     cd $FA_DIR"
echo "     python train.py --idname <person_id> --iterations 30000"
echo "  5. 学習済みモデルを FLARE にコピー:"
echo "     cp -r $FA_DIR/dataset/<person_id>/log/point_cloud \\"
echo "           $FA_CKPT_DIR/<person_id>/"
echo "     # → $FA_CKPT_DIR/<person_id>/point_cloud/iteration_30000/point_cloud.ply"
echo ""
echo "FLARE での設定 (configs/realtime_flame.yaml):"
echo "  renderer:"
echo "    type: flash_avatar"
echo "    model_path: ./checkpoints/flashavatar/<person_id>/"
echo "    repo_dir: ./third_party/FlashAvatar"
echo ""

echo "=== FlashAvatar セットアップ完了 ==="
echo ""
echo "動作確認:"
echo "  cd $FLARE_ROOT"
echo "  python tool.py render --dry-run \\"
echo "      --input-dir ./data/features/ \\"
echo "      --output-dir ./data/rendered/ \\"
echo "      --route flame \\"
echo "      --renderer flashavatar \\"
echo "      --avatar-model ./checkpoints/flashavatar/<person_id>/"
