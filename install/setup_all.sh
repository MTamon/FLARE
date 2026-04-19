#!/bin/bash
# setup_all.sh
# FLARE 全コンポーネントの統合セットアップスクリプト (SMIRK + DECA + FlashAvatar)
#
# 各サブスクリプトを順次呼び出す。各スクリプトは冪等に設計されており、
# 既にセットアップ済みの場合は重複インストールをスキップする。
#
# 前提条件:
#   - Python 3.11 がアクティブな仮想環境
#   - CUDA Toolkit 12.8 + nvcc が PATH に含まれていること
#   - gcc-11 / g++-11 がインストール済み (pytorch3d / diff-gaussian-rasterization ビルドに必要)
#   - git がインストール済み
#
# Usage:
#   bash install/setup_all.sh
#   bash install/setup_all.sh --smirk-only
#   bash install/setup_all.sh --deca-only
#   bash install/setup_all.sh --flashavatar-only
#   bash install/setup_all.sh --skip-smirk
#   bash install/setup_all.sh --skip-deca
#   bash install/setup_all.sh --skip-flashavatar
#
# 所要時間の目安 (初回、RTX 3090 / 高速インターネット環境):
#   - SMIRK:       約 5 分 (install_128.sh + checkpoint ダウンロード)
#   - DECA:        約 5 分 (install_128.sh + checkpoint ダウンロード)
#   - FlashAvatar: 約 20 分 (pytorch3d ソースビルド + diff-gaussian-rasterization ビルド)
#   合計:          約 30 分

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLARE_ROOT="$(dirname "$SCRIPT_DIR")"

SETUP_SMIRK=true
SETUP_DECA=true
SETUP_FA=true

for arg in "$@"; do
    case $arg in
        --smirk-only)       SETUP_DECA=false; SETUP_FA=false ;;
        --deca-only)        SETUP_SMIRK=false; SETUP_FA=false ;;
        --flashavatar-only) SETUP_SMIRK=false; SETUP_DECA=false ;;
        --skip-smirk)       SETUP_SMIRK=false ;;
        --skip-deca)        SETUP_DECA=false ;;
        --skip-flashavatar) SETUP_FA=false ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: bash setup_all.sh [--smirk-only | --deca-only | --flashavatar-only]"
            echo "                         [--skip-smirk] [--skip-deca] [--skip-flashavatar]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  FLARE 統合セットアップ"
echo "  SMIRK: $SETUP_SMIRK  /  DECA: $SETUP_DECA  /  FlashAvatar: $SETUP_FA"
echo "============================================================"
echo ""

# ---- 事前チェック ----
echo "[前提確認]"

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc が見つかりません。CUDA Toolkit 12.8 をインストールして"
    echo "       nvcc が PATH に含まれていることを確認してください。"
    exit 1
fi
CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "  CUDA バージョン: $CUDA_VER"

if ! python -c "import torch" 2>/dev/null; then
    echo "WARNING: PyTorch が見つかりません。"
    echo "         先に build_environment.sh を実行してください:"
    echo "         bash install/build_environment.sh --cuda 12.8"
    echo ""
fi

# git サブモジュールが初期化済みかチェック (未初期化でも各 setup_*.sh が対応)
cd "$FLARE_ROOT"
echo ""

# ---- SMIRK ----
if [ "$SETUP_SMIRK" = true ]; then
    echo "============================================================"
    echo "  [1/3] SMIRK セットアップ"
    echo "============================================================"
    bash "$SCRIPT_DIR/setup_smirk.sh"
    echo ""
else
    echo "[1/3] SMIRK セットアップ: スキップ"
    echo ""
fi

# ---- DECA ----
if [ "$SETUP_DECA" = true ]; then
    echo "============================================================"
    echo "  [2/3] DECA セットアップ"
    echo "============================================================"
    bash "$SCRIPT_DIR/setup_deca.sh"
    echo ""
else
    echo "[2/3] DECA セットアップ: スキップ"
    echo ""
fi

# ---- FlashAvatar ----
if [ "$SETUP_FA" = true ]; then
    echo "============================================================"
    echo "  [3/3] FlashAvatar セットアップ"
    echo "============================================================"
    bash "$SCRIPT_DIR/setup_flashavatar.sh"
    echo ""
else
    echo "[3/3] FlashAvatar セットアップ: スキップ"
    echo ""
fi

# ---- 完了サマリ ----
echo "============================================================"
echo "  セットアップ完了"
echo "============================================================"
echo ""
echo "次のステップ:"
echo ""

if [ "$SETUP_SMIRK" = true ]; then
    echo "  [SMIRK] 動作確認:"
    echo "    python -c \"from flare.extractors.smirk import SMIRKExtractor; print('OK')\""
    echo ""
fi

if [ "$SETUP_DECA" = true ]; then
    echo "  [DECA] 動作確認:"
    echo "    python -c \"from flare.extractors.deca import DECAExtractor; print('OK')\""
    echo ""
fi

if [ "$SETUP_FA" = true ]; then
    echo "  [FlashAvatar] 対象人物の学習 (チェックポイント未配置の場合):"
    echo "    python scripts/train_flashavatar.py \\"
    echo "        --id_name <person_id> \\"
    echo "        --video data/raw/<person>.mp4 \\"
    echo "        --config configs/train_flashavatar.yaml"
    echo ""
fi

echo "  デモの実行:"
echo "    bash demos/run_demo_webcam.sh --checkpoint_dir checkpoints/flashavatar/<person_id>  # SMIRK"
echo "    bash demos/run_demo_webcam_deca.sh --checkpoint_dir checkpoints/flashavatar/<person_id>  # DECA"
echo ""
echo "  詳細は demos/DEMOS.md を参照してください。"
