#!/bin/bash
# train_flashavatar.sh
# FlashAvatar レンダラ学習の全パイプラインスクリプト
#
# DECA で抽出した FLAME 特徴量と入力動画から FlashAvatar の
# 3D Gaussian Splatting モデルを学習する一連の処理を実行する。
#
# パイプライン:
#   Step 1: 動画フレーム抽出 + DECA per-frame 特徴抽出
#   Step 2: MediaPipe によるセグメンテーションマスク生成
#   Step 3: DECA .pt → FlashAvatar .frame 変換 (FlameConverter)
#   Step 4: FlashAvatar train.py で 3DGS モデルを学習
#   Step 5: FlashAvatar test.py で検証動画を生成
#
# Usage:
#   bash scripts/train_flashavatar.sh \
#       --id_name person01 \
#       --video /path/to/person01.mp4 \
#       [options]
#
# Options:
#   --id_name         学習 ID (データ識別子、ファイル名に使用)     [必須]
#   --video           入力動画ファイルパス                        [必須]
#   --device          CUDA デバイス (例: cuda:0)                  [既定: cuda:0]
#   --img_size        フレーム解像度 (px)                         [既定: 512]
#   --iterations      FlashAvatar 学習イテレーション数             [既定: 30000]
#   --model_path      DECA チェックポイントパス                    [既定: checkpoints/deca/deca_model.tar]
#   --deca_dir        DECA リポジトリパス                          [既定: third_party/DECA]
#   --fa_dir          FlashAvatar リポジトリパス                   [既定: third_party/FlashAvatar]
#   --data_root       学習データ出力ルートディレクトリ              [既定: data/flashavatar_training]
#   --skip_extract    Step 1 をスキップ (抽出済みの場合)
#   --skip_masks      Step 2 をスキップ (マスク生成済みの場合)
#   --skip_convert    Step 3 をスキップ (.frame 変換済みの場合)
#   --test_only       Step 5 のみ実行 (学習済みモデルで推論)

set -euo pipefail

# ---------------------------------------------------------------------------
# デフォルト値
# ---------------------------------------------------------------------------
ID_NAME=""
VIDEO=""
DEVICE="cuda:0"
IMG_SIZE=512
ITERATIONS=30000
MODEL_PATH="./checkpoints/deca/deca_model.tar"
DECA_DIR="./third_party/DECA"
FA_DIR="./third_party/FlashAvatar"
DATA_ROOT="./data/flashavatar_training"
SKIP_EXTRACT=false
SKIP_MASKS=false
SKIP_CONVERT=false
TEST_ONLY=false

# ---------------------------------------------------------------------------
# 引数パース
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --id_name)      ID_NAME="$2";      shift 2 ;;
        --video)        VIDEO="$2";        shift 2 ;;
        --device)       DEVICE="$2";       shift 2 ;;
        --img_size)     IMG_SIZE="$2";     shift 2 ;;
        --iterations)   ITERATIONS="$2";   shift 2 ;;
        --model_path)   MODEL_PATH="$2";   shift 2 ;;
        --deca_dir)     DECA_DIR="$2";     shift 2 ;;
        --fa_dir)       FA_DIR="$2";       shift 2 ;;
        --data_root)    DATA_ROOT="$2";    shift 2 ;;
        --skip_extract) SKIP_EXTRACT=true; shift ;;
        --skip_masks)   SKIP_MASKS=true;   shift ;;
        --skip_convert) SKIP_CONVERT=true; shift ;;
        --test_only)    TEST_ONLY=true;    SKIP_EXTRACT=true; SKIP_MASKS=true; SKIP_CONVERT=true; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//' | head -40
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# バリデーション
# ---------------------------------------------------------------------------
if [[ -z "$ID_NAME" ]]; then
    echo "ERROR: --id_name は必須です"
    exit 1
fi
if [[ "$TEST_ONLY" = false && "$SKIP_EXTRACT" = false && -z "$VIDEO" ]]; then
    echo "ERROR: --video は --skip_extract なしでは必須です"
    exit 1
fi

FLARE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FA_ABS="$(cd "$FLARE_ROOT" && realpath "$FA_DIR")"
DATA_DIR="$FLARE_ROOT/$DATA_ROOT/$ID_NAME"
DECA_PT_DIR="$DATA_DIR/deca_outputs"
FRAME_DIR="$FA_ABS/metrical-tracker/output/$ID_NAME/checkpoint"

echo "======================================================================"
echo "  FlashAvatar 学習パイプライン"
echo "======================================================================"
echo "  ID        : $ID_NAME"
echo "  VIDEO     : ${VIDEO:-（スキップ）}"
echo "  DEVICE    : $DEVICE"
echo "  ITERS     : $ITERATIONS"
echo "  DATA_DIR  : $DATA_DIR"
echo "  FA_DIR    : $FA_ABS"
echo "======================================================================"
echo ""

# ---------------------------------------------------------------------------
# FLARE ルートへ移動 (相対パスのため)
# ---------------------------------------------------------------------------
cd "$FLARE_ROOT"

# ---------------------------------------------------------------------------
# 前提チェック
# ---------------------------------------------------------------------------
echo "[CHECK] 前提条件の確認..."

# FLAME モデルアセット
FLAME_MODEL="$FA_ABS/flame/generic_model.pkl"
FLAME_MASKS="$FA_ABS/flame/FLAME_masks/FLAME_masks.pkl"
if [[ ! -f "$FLAME_MODEL" ]]; then
    echo "ERROR: FLAME モデルが見つかりません: $FLAME_MODEL"
    echo "       https://flame.is.tue.mpg.de/ でダウンロードして配置してください。"
    echo "       詳細: docs/guide_deca_flashavatar.md"
    exit 1
fi
if [[ ! -f "$FLAME_MASKS" ]]; then
    echo "ERROR: FLAME マスクが見つかりません: $FLAME_MASKS"
    echo "       https://flame.is.tue.mpg.de/ から FLAME_masks.zip をダウンロードして"
    echo "       $FA_ABS/flame/FLAME_masks/ に展開してください。"
    exit 1
fi
echo "      FLAME モデルアセット: OK"

# FlashAvatar が初期化済みか
if [[ ! -f "$FA_ABS/train.py" ]]; then
    echo "ERROR: FlashAvatar サブモジュールが未初期化です"
    echo "       bash install/setup_flashavatar.sh を実行してください"
    exit 1
fi
echo "      FlashAvatar: OK"

# DECA チェックポイント
if [[ "$SKIP_EXTRACT" = false && ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: DECA チェックポイントが見つかりません: $MODEL_PATH"
    echo "       bash install/setup_deca.sh を実行してください"
    exit 1
fi
echo "      DECA チェックポイント: OK"
echo ""

# ---------------------------------------------------------------------------
# Step 1: フレーム抽出 + DECA per-frame 特徴抽出
# ---------------------------------------------------------------------------
if [[ "$SKIP_EXTRACT" = false ]]; then
    echo "[Step 1] フレーム抽出 + DECA per-frame 特徴抽出..."
    echo "         入力動画: $VIDEO"
    echo "         出力先  : $DATA_DIR"
    echo ""

    python scripts/extract_deca_frames.py \
        --video "$VIDEO" \
        --out_dir "$DATA_DIR" \
        --model_path "$MODEL_PATH" \
        --deca_dir "$DECA_DIR" \
        --device "$DEVICE" \
        --img_size "$IMG_SIZE"

    echo ""
    echo "[Step 1] 完了"
    echo ""
else
    echo "[Step 1] スキップ (--skip_extract 指定)"
    echo ""
fi

# フレーム数の確認
N_FRAMES=$(ls "$DECA_PT_DIR"/*.pt 2>/dev/null | wc -l)
if [[ $N_FRAMES -lt 600 ]]; then
    echo "WARNING: フレーム数が少ない可能性があります ($N_FRAMES フレーム)"
    echo "         FlashAvatar は最低 600 フレームを推奨します"
fi
echo "      検出フレーム数: $N_FRAMES"
echo ""

# ---------------------------------------------------------------------------
# Step 2: MediaPipe によるマスク生成
# ---------------------------------------------------------------------------
if [[ "$SKIP_MASKS" = false ]]; then
    echo "[Step 2] MediaPipe によるセグメンテーションマスク生成..."
    echo "         入力  : $DATA_DIR/imgs"
    echo "         出力  : $DATA_DIR/{parsing, alpha}"
    echo ""

    python scripts/generate_masks_mediapipe.py \
        --imgs_dir "$DATA_DIR/imgs" \
        --out_dir "$DATA_DIR" \
        --img_size "$IMG_SIZE"

    echo ""
    echo "[Step 2] 完了"
    echo ""
else
    echo "[Step 2] スキップ (--skip_masks 指定)"
    echo ""
fi

# マスクの確認
N_NECKHEAD=$(ls "$DATA_DIR/parsing/"*_neckhead.png 2>/dev/null | wc -l)
N_MOUTH=$(ls "$DATA_DIR/parsing/"*_mouth.png 2>/dev/null | wc -l)
N_ALPHA=$(ls "$DATA_DIR/alpha/"*.jpg 2>/dev/null | wc -l)
echo "      neckhead マスク: $N_NECKHEAD"
echo "      mouth マスク   : $N_MOUTH"
echo "      alpha マスク   : $N_ALPHA"
echo ""

# ---------------------------------------------------------------------------
# Step 3: DECA .pt → FlashAvatar .frame 変換
# ---------------------------------------------------------------------------
if [[ "$SKIP_CONVERT" = false ]]; then
    echo "[Step 3] DECA .pt → FlashAvatar .frame 変換..."
    echo "         入力  : $DECA_PT_DIR"
    echo "         出力  : $FRAME_DIR"
    echo ""

    mkdir -p "$FRAME_DIR"

    # FlashAvatar の flame_converter.py を使用
    cd "$FA_ABS"
    python utils/flame_converter.py \
        --tracker deca \
        --input_dir "$DECA_PT_DIR" \
        --output_dir "$FRAME_DIR" \
        --img_size "${IMG_SIZE},${IMG_SIZE}" \
        --ext ".pt"
    cd "$FLARE_ROOT"

    N_FRAMES_OUT=$(ls "$FRAME_DIR"/*.frame 2>/dev/null | wc -l)
    echo ""
    echo "[Step 3] 完了: $N_FRAMES_OUT .frame ファイル生成"
    echo ""
else
    echo "[Step 3] スキップ (--skip_convert 指定)"
    echo ""
fi

# ---------------------------------------------------------------------------
# FlashAvatar データ構造のセットアップ (シンボリックリンク)
# ---------------------------------------------------------------------------
echo "[SETUP] FlashAvatar データ構造の準備..."

# dataset/<id_name> → FLARE の学習データ
mkdir -p "$FA_ABS/dataset"
FA_DATASET_LINK="$FA_ABS/dataset/$ID_NAME"
if [[ -L "$FA_DATASET_LINK" ]]; then
    rm "$FA_DATASET_LINK"
fi
if [[ -d "$FA_DATASET_LINK" ]]; then
    echo "WARNING: $FA_DATASET_LINK はシンボリックリンクではなくディレクトリです。"
    echo "         手動で確認してください。"
else
    ln -s "$DATA_DIR" "$FA_DATASET_LINK"
    echo "      シンボリックリンク作成: $FA_DATASET_LINK → $DATA_DIR"
fi

# metrical-tracker/output/<id_name> は Step 3 で直接書き込み済み
echo "      .frame ディレクトリ: $FRAME_DIR"
echo ""

# ---------------------------------------------------------------------------
# Step 4: FlashAvatar 学習
# ---------------------------------------------------------------------------
if [[ "$TEST_ONLY" = false ]]; then
    echo "[Step 4] FlashAvatar 学習 (iterations=$ITERATIONS)..."
    echo "         学習データ: $FA_ABS/dataset/$ID_NAME"
    echo "         フレームデータ: $FRAME_DIR"
    echo ""

    CKPT_DIR="$DATA_DIR/log/ckpt"
    if [[ -d "$CKPT_DIR" ]] && ls "$CKPT_DIR"/*.pth 1>/dev/null 2>&1; then
        LATEST_CKPT=$(ls -t "$CKPT_DIR"/*.pth | head -1)
        echo "      既存チェックポイントを検出: $LATEST_CKPT"
        echo "      --start_checkpoint を設定して学習を再開します"
        START_CKPT_ARG="--start_checkpoint $LATEST_CKPT"
    else
        START_CKPT_ARG=""
    fi

    cd "$FA_ABS"
    python train.py \
        --idname "$ID_NAME" \
        --iterations "$ITERATIONS" \
        --image_res "$IMG_SIZE" \
        $START_CKPT_ARG
    cd "$FLARE_ROOT"

    echo ""
    echo "[Step 4] 学習完了"
    echo "         チェックポイント: $DATA_DIR/log/ckpt/"
    echo ""
else
    echo "[Step 4] スキップ (--test_only 指定)"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 5: 検証動画の生成
# ---------------------------------------------------------------------------
echo "[Step 5] 検証動画の生成..."

CKPT_DIR="$DATA_DIR/log/ckpt"
if ls "$CKPT_DIR"/*.pth 1>/dev/null 2>&1; then
    LATEST_CKPT=$(ls -t "$CKPT_DIR"/*.pth | head -1)
    echo "         使用チェックポイント: $LATEST_CKPT"

    cd "$FA_ABS"
    python test.py \
        --idname "$ID_NAME" \
        --checkpoint "$LATEST_CKPT"
    cd "$FLARE_ROOT"

    TEST_VIDEO="$DATA_DIR/log/test.avi"
    echo ""
    echo "[Step 5] 完了: $TEST_VIDEO"
    echo ""
else
    echo "WARNING: チェックポイントが見つかりません。Step 5 をスキップします。"
fi

# ---------------------------------------------------------------------------
# FLARE チェックポイントへのコピー
# ---------------------------------------------------------------------------
echo "[COPY] 学習済みモデルを FLARE チェックポイントに配置..."

FLARE_FA_CKPT="$FLARE_ROOT/checkpoints/flashavatar/$ID_NAME"
POINT_CLOUD_SRC="$DATA_DIR/log/point_cloud"

if [[ -d "$POINT_CLOUD_SRC" ]]; then
    mkdir -p "$FLARE_FA_CKPT"
    # シンボリックリンクで配置 (コピーしない)
    POINT_CLOUD_LINK="$FLARE_FA_CKPT/point_cloud"
    if [[ -L "$POINT_CLOUD_LINK" ]]; then
        rm "$POINT_CLOUD_LINK"
    fi
    ln -s "$POINT_CLOUD_SRC" "$POINT_CLOUD_LINK"
    echo "      $FLARE_FA_CKPT/point_cloud → $POINT_CLOUD_SRC"
else
    echo "WARNING: point_cloud ディレクトリが見つかりません: $POINT_CLOUD_SRC"
    echo "         train.py が正常完了しているか確認してください"
fi

# ---------------------------------------------------------------------------
# 完了サマリ
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "  パイプライン完了"
echo "======================================================================"
echo ""
echo "  学習済みモデル : checkpoints/flashavatar/$ID_NAME/point_cloud/"
echo "  検証動画       : data/flashavatar_training/$ID_NAME/log/test.avi"
echo ""
echo "  FLARE でのリアルタイム動作:"
echo "    # configs/realtime_flame.yaml の renderer.model_path を更新"
echo "    sed -i \"s|model_path: ./checkpoints/flashavatar/.*|model_path: ./checkpoints/flashavatar/$ID_NAME/|\" \\"
echo "        configs/realtime_flame.yaml"
echo ""
echo "    python examples/realtime_extract.py \\"
echo "        --config configs/realtime_flame.yaml \\"
echo "        --source 0"
echo ""
echo "  FLARE での LHG 前処理 (特徴抽出のみ、レンダリングなし):"
echo "    python tool.py lhg-extract \\"
echo "        --path ./data/multimodal_dialogue_formed \\"
echo "        --output ./data/movements \\"
echo "        --config configs/lhg_extract_deca.yaml"
echo "======================================================================"
