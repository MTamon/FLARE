# Guide: DECA + FlashAvatar 統合ガイド (FLAME Route B)

FLARE の Route B (FLAME ベース) における一連の動作手順をまとめたガイドです。

```
動画入力 → DECA (特徴抽出) → DECAToFlameAdapter → FlashAvatar (3DGS レンダリング) → 顔画像
```

---

## 目次

1. [前提条件](#1-前提条件)
2. [環境構築](#2-環境構築)
3. [チェックポイントの準備](#3-チェックポイントの準備)
4. [FlashAvatar モデルの学習](#4-flashavatar-モデルの学習)
5. [LHG 前処理 (バッチ特徴抽出)](#5-lhg-前処理-バッチ特徴抽出)
6. [リアルタイムパイプライン](#6-リアルタイムパイプライン)
7. [バッチ特徴抽出・レンダリング](#7-バッチ特徴抽出レンダリング)
8. [パラメータ変換の詳細](#8-パラメータ変換の詳細)
9. [トラブルシューティング](#9-トラブルシューティング)

---

## 1. 前提条件

| 項目 | 要件 |
|------|------|
| OS | Ubuntu 20.04 以上 |
| Python | 3.11 |
| CUDA Toolkit | 12.8 (`nvcc --version` で確認) |
| コンパイラ | gcc-11 / g++-11 (`gcc-11 --version` で確認) |
| GPU VRAM | 12 GB 以上 (DECA + FlashAvatar 同時使用時) |
| FLARE 本体 | `build_environment.sh` 実行済み |

```bash
# コンパイラのインストール (Ubuntu)
sudo apt-get install gcc-11 g++-11

# nvcc の確認
nvcc --version  # CUDA 12.8 であること
```

---

## 2. 環境構築

### 統合セットアップ (推奨)

```bash
cd /path/to/FLARE

# FLAME ルート全体を一括セットアップ
bash install/setup_flame_route.sh

# DECA のみセットアップ (特徴抽出・前処理だけ使う場合)
bash install/setup_flame_route.sh --deca-only

# FlashAvatar のみセットアップ
bash install/setup_flame_route.sh --flashavatar-only
```

### 個別セットアップ

```bash
# DECA セットアップ (MTamon/DECA@cuda128)
bash install/setup_deca.sh

# FlashAvatar セットアップ (MTamon/FlashAvatar@release/cuda128-fixed)
bash install/setup_flashavatar.sh
```

### サブモジュールの構造

統合セットアップ後、以下のディレクトリが作成されます:

```
FLARE/
└── third_party/
    ├── DECA/              # MTamon/DECA@cuda128
    │   ├── decalib/       # DECA コアライブラリ (sys.path 追加対象)
    │   ├── install_128.sh
    │   └── requirements_128.txt
    └── FlashAvatar/       # MTamon/FlashAvatar@release/cuda128-fixed
        ├── scene/         # GaussianModel 等 (sys.path 追加対象)
        ├── gaussian_renderer/
        ├── submodules/    # diff-gaussian-rasterization, simple-knn
        ├── install_128.sh
        └── requirements_128.txt
```

**FLARE → 外部リポジトリのインポートパス:**

| モジュール | インポートパス | リポジトリ |
|---|---|---|
| `decalib.deca.DECA` | `third_party/DECA/decalib/` | MTamon/DECA |
| `decalib.utils.config.cfg` | `third_party/DECA/decalib/` | MTamon/DECA |
| `scene.gaussian_model.GaussianModel` | `third_party/FlashAvatar/` | MTamon/FlashAvatar |
| `gaussian_renderer.render` | `third_party/FlashAvatar/` | MTamon/FlashAvatar |

これらのインポートは YAML 設定ファイルの `repo_dir` フィールドで制御されます
(コード変更なしでパスを差し替え可能)。

---

## 3. チェックポイントの準備

### DECA チェックポイント

```bash
# 自動ダウンロード (setup_deca.sh 実行時に自動実行)
gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O checkpoints/deca/deca_model.tar

# または FLARE の一括ダウンロードスクリプト
python scripts/download_checkpoints.py --model deca
```

**配置先:** `checkpoints/deca/deca_model.tar`

### FLAME モデル (メッシュ可視化のみ必要)

DECA の特徴抽出・前処理には不要です。
`demo_visualize.py --mode mesh` でメッシュ表示を行う場合のみ必要です。

1. https://flame.is.tue.mpg.de/ でアカウント登録・ライセンス同意
2. "FLAME 2020" をダウンロードして展開
3. `generic_model.pkl` を `checkpoints/flame/` に配置

### FlashAvatar チェックポイント

FlashAvatar は **対象人物ごとの個別学習** が必要です。
汎用の事前学習済みモデルはありません。詳細は [Section 4](#4-flashavatar-モデルの学習) を参照。

---

## 4. FlashAvatar モデルの学習

FlashAvatar のレンダリングを使用するには、対象人物の動画から事前に 3DGS モデルを学習する必要があります。

### 4.1 入力データの準備

```
dataset/<person_id>/
├── imgs/               # 動画フレーム (00001.jpg, 00002.jpg, ...)
├── parsing/            # 頭部・口部のセグメンテーションマスク (PNG)
└── alpha/              # 不透明度マスク (グレースケール PNG)
```

**フレーム数の目安:** 最低 600 フレーム、推奨 2000 フレーム以上

```bash
# ffmpeg でフレーム展開
mkdir -p dataset/<person_id>/imgs
ffmpeg -i input_video.mp4 -q:v 2 dataset/<person_id>/imgs/%05d.jpg
```

### 4.2 FLAME トラッキング

per-frame の FLAME パラメータが必要です。
[MICA (metrical-tracker)](https://github.com/Zielon/metrical-tracker) を使用します:

```bash
# metrical-tracker は FlashAvatar とは別途インストール
# トラッキング結果の配置先:
# metrical-tracker/output/<person_id>/checkpoint/<frame>.frame
```

### 4.3 学習実行

```bash
cd third_party/FlashAvatar

# 学習 (RTX 3090 で約 30 分 @ 30,000 iterations)
python train.py \
    --idname <person_id> \
    --iterations 30000

# 学習済みチェックポイントの確認
ls dataset/<person_id>/log/point_cloud/iteration_30000/
# → point_cloud.ply
```

### 4.4 FLARE へのチェックポイント配置

```bash
PERSON_ID=<person_id>

# FLARE のチェックポイントディレクトリにコピー
mkdir -p checkpoints/flashavatar/${PERSON_ID}
cp -r third_party/FlashAvatar/dataset/${PERSON_ID}/log/point_cloud \
      checkpoints/flashavatar/${PERSON_ID}/

# 確認
ls checkpoints/flashavatar/${PERSON_ID}/point_cloud/iteration_30000/
# → point_cloud.ply が存在すること
```

---

## 5. LHG 前処理 (バッチ特徴抽出)

対面対話データセットから DECA で FLAME パラメータを一括抽出します。

### 入力データ構造

```
data/multimodal_dialogue_formed/
├── data001/
│   ├── comp.mp4   # 参加者動画
│   └── host.mp4   # ホスト動画
├── data002/
│   ├── comp.mp4
│   └── host.mp4
...
```

### 実行コマンド

```bash
# 設定確認のみ (モデル読み込みなし)
python tool.py lhg-extract --dry-run \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --config configs/lhg_extract_deca.yaml

# 実行
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --config configs/lhg_extract_deca.yaml

# GPU 指定
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --config configs/lhg_extract_deca.yaml \
    --gpus 0
```

### 出力データ構造

```
data/movements/
├── data001/
│   ├── comp/
│   │   └── deca_comp_00000_04499.npz  # フレーム 0〜4499
│   └── host/
│       └── deca_host_00000_04499.npz
├── data002/
│   ...
├── metadata.json     # 処理サマリ
└── summary.csv       # 各動画の統計
```

各 `.npz` の内容:

| キー | 形状 | 説明 |
|------|------|------|
| `exp` | `(T, 50)` | FLAME 表情パラメータ (正規化済み) |
| `shape` | `(300,)` | FLAME 形状パラメータ (シーケンス集約) |
| `angle` | `(T, 3)` | 頭部姿勢 (axis-angle, SLERP 補間済み) |
| `centroid` | `(T, 2)` | 顔中心座標 |
| `jaw_pose` | `(T, 3)` | 顎関節角度 |
| `face_size` | `(T, 1)` | 顔スケール |

### 設定ファイル

`configs/lhg_extract_deca.yaml` の主要パラメータ:

```yaml
extractor:
  type: deca
  model_path: ./checkpoints/deca/deca_model.tar
  repo_dir: ./third_party/DECA  # サブモジュールパス

lhg_extract:
  interpolation:
    linear_order: linear  # linear or pchip
    rotation_order: slerp # slerp or linear
    max_gap_sec: 0.4      # 補間許容ギャップ (秒)
  sequence:
    min_length: 100       # 最小シーケンス長 (フレーム)
  output:
    shape_aggregation: median  # median / first / mean
```

---

## 6. リアルタイムパイプライン

Webカメラや動画ファイルからリアルタイムに特徴抽出・レンダリングを行います。

> **注意:** リアルタイムレンダリングには FlashAvatar の学習済みモデルが必要です。

### 実行コマンド

```bash
# Webカメラ (デバイス 0)
python examples/realtime_extract.py \
    --config configs/realtime_flame.yaml \
    --source 0

# 動画ファイル
python examples/realtime_extract.py \
    --config configs/realtime_flame.yaml \
    --source ./data/test_video.mp4

# PyQt6 GUI 表示
python examples/realtime_extract.py \
    --config configs/realtime_flame.yaml \
    --source 0 \
    --display pyqt
```

停止: ウィンドウで `q` キーを押す

### 設定ファイル

`configs/realtime_flame.yaml`:

```yaml
extractor:
  type: deca
  model_path: ./checkpoints/deca/deca_model.tar
  repo_dir: ./third_party/DECA          # DECA サブモジュールパス

renderer:
  type: flash_avatar
  model_path: ./checkpoints/flashavatar/<person_id>/
  output_size: [512, 512]
  repo_dir: ./third_party/FlashAvatar  # FlashAvatar サブモジュールパス

pipeline:
  converter_chain:
    - type: deca_to_flame   # DECA → FlashAvatar パラメータ変換

device_map:
  extractor: cuda:0
  lhg_model: cuda:0
  renderer: cuda:0
```

---

## 7. バッチ特徴抽出・レンダリング

### バッチ特徴抽出

```bash
python tool.py extract \
    --input-dir ./data/videos/ \
    --output-dir ./data/features/ \
    --route flame \
    --extractor deca \
    --gpu 0 \
    --batch-size 32
```

### バッチレンダリング (FlashAvatar)

```bash
python tool.py render \
    --input-dir ./data/features/ \
    --output-dir ./data/rendered/ \
    --route flame \
    --renderer flashavatar \
    --avatar-model ./checkpoints/flashavatar/<person_id>/ \
    --resolution 512,512
```

---

## 8. パラメータ変換の詳細

DECA と FlashAvatar はどちらも FLAME モデルベースですが、パラメータ形式が異なります。
`DECAToFlameAdapter` (`flare/converters/deca_to_flame.py`) がこの変換を担います。

| パラメータ | DECA 出力 | FlashAvatar 入力 | 変換方法 |
|---|---|---|---|
| 表情 | `exp` (50D) | `expr` (100D) | ゼロパディング ※1 |
| 顎回転 | `pose[3:6]` (axis-angle 3D) | `jaw_pose` (rotation_6d 6D) | axis-angle → 回転行列 → 6D 表現 |
| 眼球回転 | (なし) | `eyes_pose` (12D) | 単位回転行列の 6D × 2 |
| まぶた | (なし) | `eyelids` (2D) | ゼロ埋め |

**※1 ゼロパディングの根拠:**
DECA の `exp` (50D) と FlashAvatar の `expr` (100D) は同一の FLAME expression PCA 空間に属します。
両者は同一の `generic_model.pkl` から `shapedirs[:,:,300:300+n_exp]` をスライスしており、
DECA は `n_exp=50`、FlashAvatar は `n_exp=100` を使用します。

```python
# DECAToFlameAdapter の変換ロジック (flare/converters/deca_to_flame.py)
from flare.converters.deca_to_flame import DECAToFlameAdapter

adapter = DECAToFlameAdapter()
flash_params = adapter.convert({
    "exp": deca_exp_50d,    # (B, 50)
    "pose": deca_pose_6d,   # (B, 6)
})
# flash_params["expr"].shape  == (B, 100)
# flash_params["jaw_pose"].shape == (B, 6)
# flash_params["eyes_pose"].shape == (B, 12)
# flash_params["eyelids"].shape == (B, 2)
```

---

## 9. トラブルシューティング

| 症状 | 原因と対策 |
|------|-----------|
| `ModuleNotFoundError: No module named 'decalib'` | DECA サブモジュール未セットアップ。`bash install/setup_deca.sh` を実行 |
| `ModuleNotFoundError: No module named 'scene'` | FlashAvatar サブモジュール未セットアップ。`bash install/setup_flashavatar.sh` を実行 |
| `ModuleNotFoundError: No module named 'diff_gaussian_rasterization'` | 3DGS ラスタライザ未ビルド。`bash install/setup_flashavatar.sh` を再実行 |
| `FileNotFoundError: FlashAvatar PLY not found` | FlashAvatar モデル未学習。`train.py` を実行してチェックポイントを配置 |
| `CUDA out of memory` | `device_map` で GPU 分散、または `batch_size` を減らす |
| `nvcc: command not found` | CUDA Toolkit 12.8 未インストール。`sudo apt-get install cuda-toolkit-12-8` |
| `gcc-11: command not found` | `sudo apt-get install gcc-11 g++-11` |
| `pytorch3d` インストールエラー | `install_128.sh` の実行前に CUDA 環境変数を確認: `echo $CUDA_HOME` |
| FLAME ダウンロードエラー | https://flame.is.tue.mpg.de/ でアカウント登録・ライセンス同意が必要 |

### デバッグ: インポートパスの確認

```python
import sys
# DECA が sys.path に追加されているか確認
print([p for p in sys.path if 'DECA' in p])

# FlashAvatar が sys.path に追加されているか確認
print([p for p in sys.path if 'FlashAvatar' in p])
```

### サブモジュールのリセット

```bash
# サブモジュールを指定ブランチの最新状態に更新
git submodule update --remote third_party/DECA
git submodule update --remote third_party/FlashAvatar
```
