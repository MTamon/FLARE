#!/bin/bash
# setup_deca.sh
# DECA (MTamon/DECA@cuda128) のセットアップスクリプト
#
# 前提条件:
#   - Python 3.11 がアクティブな仮想環境
#   - CUDA Toolkit 12.8 + nvcc がインストール済み
#   - git サブモジュールが初期化済み (git submodule update --init third_party/DECA)
#
# Usage:
#   bash install/setup_deca.sh
#
# 実行内容:
#   1. third_party/DECA サブモジュールの初期化・更新
#   2. DECA の CUDA 12.8 対応 install_128.sh を実行
#   3. DECA チェックポイント (deca_model.tar) のダウンロード案内

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLARE_ROOT="$(dirname "$SCRIPT_DIR")"
DECA_DIR="$FLARE_ROOT/third_party/DECA"
DECA_CKPT_DIR="$FLARE_ROOT/checkpoints/deca"

echo "=== DECA (CUDA 12.8) セットアップ ==="
echo "FLARE root : $FLARE_ROOT"
echo "DECA dir   : $DECA_DIR"
echo ""

# ---- 1. サブモジュールの初期化・更新 ----
echo "[1/3] サブモジュールの初期化・更新..."
cd "$FLARE_ROOT"
git submodule update --init --recursive third_party/DECA
echo "      ブランチ: $(cd "$DECA_DIR" && git branch --show-current)"
echo "      コミット: $(cd "$DECA_DIR" && git rev-parse --short HEAD)"
echo ""

# ---- 2. DECA の CUDA 12.8 依存パッケージをインストール ----
echo "[2/3] DECA CUDA 12.8 依存パッケージのインストール..."
cd "$DECA_DIR"
if [ ! -f "install_128.sh" ]; then
    echo "ERROR: install_128.sh が見つかりません: $DECA_DIR/install_128.sh"
    exit 1
fi
bash install_128.sh
echo ""

# ---- 3. チェックポイントのダウンロード ----
echo "[3/3] DECA チェックポイットの準備..."
mkdir -p "$DECA_CKPT_DIR"

DECA_MODEL_PATH="$DECA_CKPT_DIR/deca_model.tar"
if [ -f "$DECA_MODEL_PATH" ]; then
    echo "      [SKIP] 既存チェックポイントを検出: $DECA_MODEL_PATH"
else
    echo "      deca_model.tar をダウンロードします..."
    echo "      (Google Drive ID: 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje)"

    if command -v gdown &> /dev/null; then
        gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O "$DECA_MODEL_PATH"
    else
        pip install --quiet gdown
        gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O "$DECA_MODEL_PATH"
    fi

    if [ -f "$DECA_MODEL_PATH" ]; then
        echo "      ダウンロード完了: $DECA_MODEL_PATH"
    else
        echo "WARNING: ダウンロードに失敗しました。"
        echo "         手動でダウンロードして $DECA_MODEL_PATH に配置してください。"
        echo "         URL: https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje"
    fi
fi

# FLAME generic_model.pkl (メッシュ可視化に必要・学術ライセンス)
FLAME_MODEL_PATH="$FLARE_ROOT/checkpoints/flame/generic_model.pkl"
if [ ! -f "$FLAME_MODEL_PATH" ]; then
    echo ""
    echo "INFO: FLAME モデル (generic_model.pkl) が見つかりません。"
    echo "      メッシュ可視化 (demo_visualize.py --mode mesh) を使用する場合は"
    echo "      以下の手順でダウンロードしてください:"
    echo "      1. https://flame.is.tue.mpg.de/ でアカウント登録・ライセンス同意"
    echo "      2. FLAME 2020 をダウンロードして展開"
    echo "      3. generic_model.pkl を $FLARE_ROOT/checkpoints/flame/ に配置"
fi

echo ""
echo "=== DECA セットアップ完了 ==="
echo ""
echo "動作確認:"
echo "  cd $FLARE_ROOT"
echo "  python tool.py lhg-extract --dry-run \\"
echo "      --path ./data/multimodal_dialogue_formed \\"
echo "      --output ./data/movements \\"
echo "      --config configs/lhg_extract_deca.yaml"
