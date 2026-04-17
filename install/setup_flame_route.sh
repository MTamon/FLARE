#!/bin/bash
# setup_flame_route.sh
# FLAME ルート (DECA + FlashAvatar) の統合セットアップスクリプト
#
# FLARE の Route B (FLAME) を利用するために必要な外部リポジトリを
# 一括セットアップします。
#
#   Route B (FLAME):
#     動画入力 → DECA (特徴抽出) → [DECAToFlameAdapter] → FlashAvatar (3DGS レンダリング)
#
# 前提条件:
#   - Python 3.11 がアクティブな仮想環境
#   - CUDA Toolkit 12.8 + nvcc がインストール済み
#   - gcc-11 / g++-11 がインストール済み
#   - FLARE 本体のセットアップ済み (build_environment.sh 実行済み)
#
# Usage:
#   bash install/setup_flame_route.sh [--deca-only | --flashavatar-only]
#
# オプション:
#   --deca-only         DECA のみセットアップ (特徴抽出・前処理のみ使用する場合)
#   --flashavatar-only  FlashAvatar のみセットアップ (チェックポイント移植時等)
#   (引数なし)          両方セットアップ

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLARE_ROOT="$(dirname "$SCRIPT_DIR")"

SETUP_DECA=true
SETUP_FA=true

for arg in "$@"; do
    case $arg in
        --deca-only)         SETUP_FA=false ;;
        --flashavatar-only)  SETUP_DECA=false ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: bash setup_flame_route.sh [--deca-only | --flashavatar-only]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  FLARE FLAME ルート (Route B) 統合セットアップ"
echo "  DECA: $SETUP_DECA  /  FlashAvatar: $SETUP_FA"
echo "============================================================"
echo ""

# ---- Step 0: FLARE 本体の依存パッケージ確認 ----
echo "[Step 0] FLARE 本体の依存パッケージ確認..."
cd "$FLARE_ROOT"
if ! python -c "import torch" 2>/dev/null; then
    echo "WARNING: PyTorch が見つかりません。"
    echo "         先に build_environment.sh を実行してください:"
    echo "         bash install/build_environment.sh --cuda 12.8"
fi
echo ""

# ---- Step 1: DECA セットアップ ----
if [ "$SETUP_DECA" = true ]; then
    echo "[Step 1] DECA (MTamon/DECA@cuda128) セットアップ"
    echo "------------------------------------------------------------"
    bash "$SCRIPT_DIR/setup_deca.sh"
    echo ""
else
    echo "[Step 1] DECA セットアップ: スキップ"
    echo ""
fi

# ---- Step 2: FlashAvatar セットアップ ----
if [ "$SETUP_FA" = true ]; then
    echo "[Step 2] FlashAvatar (MTamon/FlashAvatar@release/cuda128-fixed) セットアップ"
    echo "------------------------------------------------------------"
    bash "$SCRIPT_DIR/setup_flashavatar.sh"
    echo ""
else
    echo "[Step 2] FlashAvatar セットアップ: スキップ"
    echo ""
fi

# ---- 完了サマリ ----
echo "============================================================"
echo "  セットアップ完了"
echo "============================================================"
echo ""
echo "FLAME ルートで利用可能な操作:"
echo ""
echo "  A. LHG 前処理 (バッチ特徴抽出):"
echo "     python tool.py lhg-extract \\"
echo "         --path ./data/multimodal_dialogue_formed \\"
echo "         --output ./data/movements \\"
echo "         --config configs/lhg_extract_deca.yaml"
echo ""
echo "  B. リアルタイムパイプライン (Webカメラ):"
echo "     python examples/realtime_extract.py \\"
echo "         --config configs/realtime_flame.yaml \\"
echo "         --source 0"
echo ""
echo "  C. バッチ特徴抽出:"
echo "     python tool.py extract \\"
echo "         --input-dir ./data/videos/ \\"
echo "         --output-dir ./data/features/ \\"
echo "         --route flame --extractor deca --gpu 0"
echo ""
echo "  D. バッチレンダリング (FlashAvatar モデル学習後):"
echo "     python tool.py render \\"
echo "         --input-dir ./data/features/ \\"
echo "         --output-dir ./data/rendered/ \\"
echo "         --route flame --renderer flashavatar \\"
echo "         --avatar-model ./checkpoints/flashavatar/<person_id>/"
echo ""
echo "詳細は docs/guide_deca_flashavatar.md を参照してください。"
