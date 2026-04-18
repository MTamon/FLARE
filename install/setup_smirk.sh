#!/bin/bash
# setup_smirk.sh
# SMIRK (MTamon/smirk@release/cuda128) のセットアップスクリプト
#
# 前提条件:
#   - Python 3.11 がアクティブな仮想環境
#   - CUDA Toolkit 12.8 + nvcc がインストール済み
#   - gcc-11 / g++-11 がインストール済み
#   - git サブモジュールが初期化済み (git submodule update --init third_party/smirk)
#
# Usage:
#   bash install/setup_smirk.sh
#
# 実行内容:
#   1. third_party/smirk サブモジュールの初期化・更新
#   2. SMIRK の CUDA 12.8 対応 install_128.sh を実行
#      (DECA128 / FlashAvatar128 と同じ pinned バージョンセット + SMIRK 固有依存)
#   3. SMIRK 学習済みモデル (SMIRK_em1.pt) のダウンロード案内
#   4. demo 用アセットの準備 (任意, prepare_demos.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLARE_ROOT="$(dirname "$SCRIPT_DIR")"
SMIRK_DIR="$FLARE_ROOT/third_party/smirk"
SMIRK_CKPT_DIR="$FLARE_ROOT/checkpoints/smirk"

echo "=== SMIRK (CUDA 12.8) セットアップ ==="
echo "FLARE root  : $FLARE_ROOT"
echo "SMIRK dir   : $SMIRK_DIR"
echo ""

# ---- 事前チェック ----
echo "[0/4] 事前環境チェック..."

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc が見つかりません。CUDA Toolkit 12.8 をインストールして"
    echo "       nvcc が PATH に含まれていることを確認してください。"
    exit 1
fi
CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "      CUDA バージョン: $CUDA_VER"
echo ""

# ---- 1. サブモジュールの初期化・更新 ----
echo "[1/4] サブモジュールの初期化・更新..."
cd "$FLARE_ROOT"
git submodule update --init --recursive third_party/smirk
echo "      ブランチ: $(cd "$SMIRK_DIR" && git branch --show-current)"
echo "      コミット: $(cd "$SMIRK_DIR" && git rev-parse --short HEAD)"
echo ""

# ---- 2. SMIRK の CUDA 12.8 依存パッケージをインストール ----
echo "[2/4] SMIRK CUDA 12.8 依存パッケージのインストール..."
cd "$SMIRK_DIR"
if [ ! -f "install_128.sh" ]; then
    echo "ERROR: install_128.sh が見つかりません: $SMIRK_DIR/install_128.sh"
    exit 1
fi
bash install_128.sh
echo ""

# ---- 3. チェックポイントのダウンロード ----
echo "[3/4] SMIRK チェックポイントの準備..."
mkdir -p "$SMIRK_CKPT_DIR"

SMIRK_MODEL_PATH="$SMIRK_CKPT_DIR/SMIRK_em1.pt"
if [ -f "$SMIRK_MODEL_PATH" ]; then
    echo "      [SKIP] 既存チェックポイントを検出: $SMIRK_MODEL_PATH"
else
    echo "      SMIRK_em1.pt をダウンロードします..."
    echo "      (Google Drive ID: 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE)"

    if command -v gdown &> /dev/null; then
        gdown 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE -O "$SMIRK_MODEL_PATH"
    else
        pip install --quiet gdown
        gdown 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE -O "$SMIRK_MODEL_PATH"
    fi

    if [ -f "$SMIRK_MODEL_PATH" ]; then
        echo "      ダウンロード完了: $SMIRK_MODEL_PATH"
    else
        echo "WARNING: ダウンロードに失敗しました。"
        echo "         手動でダウンロードして $SMIRK_MODEL_PATH に配置してください。"
        echo "         URL: https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE"
    fi
fi

# upstream pretrained_models/ にもシンボリックリンクで配置 (demo 互換)
SMIRK_UPSTREAM_LINK="$SMIRK_DIR/pretrained_models/SMIRK_em1.pt"
if [ -f "$SMIRK_MODEL_PATH" ] && [ ! -e "$SMIRK_UPSTREAM_LINK" ]; then
    mkdir -p "$SMIRK_DIR/pretrained_models"
    ln -s "$SMIRK_MODEL_PATH" "$SMIRK_UPSTREAM_LINK"
    echo "      upstream link: $SMIRK_UPSTREAM_LINK -> $SMIRK_MODEL_PATH"
fi
echo ""

# ---- 4. demo 用追加アセット (任意) ----
echo "[4/4] demo 用追加アセット..."
PREPARE_DEMOS="$SMIRK_DIR/prepare_demos.sh"
if [ -f "$PREPARE_DEMOS" ]; then
    echo "      $SMIRK_DIR/prepare_demos.sh を実行すると、demo 用アセット"
    echo "      (FLAME2020 generic_model.pkl + face_landmarker.task) が自動取得されます。"
    echo "      - FLAME2020 はライセンス同意が必要なので --no_flame オプションも検討。"
    echo "      実行コマンド (FLARE 内 demo を動かしたい場合のみ):"
    echo "        cd $SMIRK_DIR && bash prepare_demos.sh"
else
    echo "      $PREPARE_DEMOS が見つかりません (demo 用アセットは任意のためスキップ)"
fi
echo ""

echo "=== SMIRK セットアップ完了 ==="
echo ""
echo "動作確認:"
echo "  cd $FLARE_ROOT"
echo "  python -c \"from flare.extractors.smirk import SMIRKExtractor; \\"
echo "    e = SMIRKExtractor( \\"
echo "        model_path='$SMIRK_MODEL_PATH', \\"
echo "        device='cuda:0', \\"
echo "        smirk_dir='$SMIRK_DIR'); \\"
echo "    print('SMIRK extractor loaded:', e.param_dim)\""
