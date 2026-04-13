# Guide B: LHG モデル学習用の前処理（バッチ特徴抽出）

LHG (Listening Head Generation) モデルの深層学習に使用する訓練データを、対面対話動画データセットから一括抽出する手順。

## 概要

`lhg-extract` コマンドは、対面対話動画データセット (`multimodal_dialogue_formed/`) から per-frame に 3DMM パラメータを抽出し、ギャップ補間・シーケンス分割・正規化を行って、下流の `databuild_nx8.py` と互換な `.npz` ファイルとして保存します。

```
入力: multimodal_dialogue_formed/dataXXX/{comp,host}.mp4
  ↓ MediaPipe 顔検出 → crop → DECA/Deep3D 推論
  ↓ SLERP 回転補間 / 線形補間
  ↓ 長ギャップでシーケンス分割 (0.4s)
  ↓ 対話単位正規化 (mean/std)
出力: movements/dataXXX/{comp,host}/{prefix}_{role}_{SSSSS}_{EEEEE}.npz
```

## ステップ 1: データセットの配置

```
data/multimodal_dialogue_formed/
├── data001/
│   ├── comp.mp4          # 対話参加者（comparator）の動画
│   ├── host.mp4          # ホスト側の動画
│   └── participant.json  # 話者メタデータ (必須)
├── data002/
│   └── ...
└── data042/
    └── ...
```

`participant.json` の必須キー:

```json
{
  "comp": "田中太郎",
  "comp_no": 12,
  "host": "鈴木花子",
  "host_no": 7
}
```

| キー | 用途 |
|------|------|
| `comp` / `host` | 話者名 → npz の `speaker_name` |
| `comp_no` / `host_no` | 話者 ID → npz の `speaker_id` (`databuild_nx8.py` が参照) |

## ステップ 2: チェックポイントの配置

使用する Extractor に応じてモデルファイルを配置します。

| Extractor | ルート | チェックポイント |
|-----------|--------|-----------------|
| DECA | FLAME | `checkpoints/deca_model.tar` |
| Deep3DFaceRecon | BFM | `checkpoints/deep3d_epoch20.pth` |
| SMIRK | FLAME | `checkpoints/smirk_encoder.pt` |
| 3DDFA | BFM | `checkpoints/mb1_120x120.onnx` |

## ステップ 3: 設定ファイルの選択

プリセット YAML を 2 種類用意しています:

| ファイル | Extractor | 係数体系 |
|----------|-----------|---------|
| [`configs/lhg_extract_deca.yaml`](../configs/lhg_extract_deca.yaml) | DECA | FLAME (shape 100, exp 50) |
| [`configs/lhg_extract_bfm.yaml`](../configs/lhg_extract_bfm.yaml) | Deep3DFaceRecon | BFM (id 80, exp 64) |

主要な設定パラメータ:

| YAML パス | 既定値 | 意味 |
|-----------|--------|------|
| `lhg_extract.interpolation.linear_order` | `linear` | 表情/位置の補間。`pchip` にすると滑らか (scipy 要) |
| `lhg_extract.interpolation.rotation_order` | `slerp` | 回転の補間。球面線形補間で大角度にも安全 |
| `lhg_extract.interpolation.max_gap_sec` | `0.4` | この秒数超の未検出区間でシーケンス分割 |
| `lhg_extract.sequence.min_length` | `100` | 100 フレーム未満のシーケンスは破棄 |
| `lhg_extract.output.shape_aggregation` | `median` | shape 係数を中央値で集約 (1動画→1ベクトル) |

## ステップ 4: 実行

### 基本実行

```bash
# DECA (FLAME) ルート
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --config configs/lhg_extract_deca.yaml

# Deep3DFaceRecon (BFM) ルート
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --config configs/lhg_extract_bfm.yaml
```

### ドライラン（設定確認のみ、モデル読み込みなし）

```bash
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --config configs/lhg_extract_deca.yaml \
    --dry-run
```

### CLI フラグによるオーバーライド

```bash
# Extractor 種別をコマンドラインで指定
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --extractor deca \
    --model-path ./checkpoints/deca_model.tar

# GPU 指定
python tool.py lhg-extract \
    --path ... --output ... \
    --config configs/lhg_extract_deca.yaml \
    --gpus 1

# 既存出力を上書き
python tool.py lhg-extract \
    --path ... --output ... \
    --config configs/lhg_extract_deca.yaml \
    --redo
```

## ステップ 5: 出力の確認

### 出力ディレクトリ構造

```
data/movements/
├── data001/
│   ├── comp/
│   │   ├── deca_comp_00000_04499.npz  ← フレーム 0~4499
│   │   └── deca_comp_04620_08999.npz  ← ギャップ分割後
│   ├── host/
│   │   └── deca_host_00000_08999.npz
│   └── participant.json               ← 入力からコピー
└── data002/
    └── ...
```

ファイル名規約: `{prefix}_{role}_{開始フレーム:05d}_{終了フレーム:05d}.npz`

### npz の中身

#### DECA ルートの場合

| キー | 形状 | 内容 |
|------|------|------|
| `section` | `int32 (2,)` | `[start_frame, end_frame]` |
| `speaker_id` | `int64 ()` | 話者 ID |
| `fps` | `float32 ()` | 動画 FPS |
| `angle` | `float32 (T, 3)` | 正規化済み頭部回転 (軸角) |
| `centroid` | `float32 (T, 3)` | 正規化済み頭部位置 |
| `expression` | `float32 (T, 50)` | 正規化済み FLAME 表情係数 |
| `shape` | `float32 (100,)` | FLAME 形状 (シーケンス中央値) |
| `jaw_pose` | `float32 (T, 3)` | 顎回転 (DECA 固有) |
| `face_size` | `float32 (T,)` | 顔サイズ (DECA 固有) |
| `angle_mean/std` | `float32 (3,)` | 正規化統計量 |
| `expression_mean/std` | `float32 (50,)` | 正規化統計量 |
| `centroid_mean/std` | `float32 (3,)` | 正規化統計量 |

#### BFM ルートの場合の違い

| キー | 形状 | 違い |
|------|------|------|
| `expression` | `float32 (T, 64)` | BFM は 64 次元 |
| `shape` | `float32 (80,)` | BFM id 係数 80 次元 |
| `jaw_pose` | --- | 存在しない |
| `face_size` | --- | 存在しない |

### Python での読み込み

```python
import numpy as np

data = np.load("data/movements/data001/comp/deca_comp_00000_04499.npz")

# 正規化済みデータ（学習にはこれをそのまま使用）
angle = data["angle"]           # (T, 3)
expression = data["expression"] # (T, 50)
centroid = data["centroid"]     # (T, 3)
shape = data["shape"]           # (100,)
speaker_id = int(data["speaker_id"])

# デノーマライズ（元スケールに戻す場合）
angle_raw = data["angle"] * data["angle_std"] + data["angle_mean"]

# PyTorch で使う場合
import torch
angle_tensor = torch.from_numpy(data["angle"])  # float32
```

### 一括読み込み

```python
from pathlib import Path
import numpy as np

root = Path("data/movements")
for data_dir in sorted(root.iterdir()):
    for role in ("comp", "host"):
        role_dir = data_dir / role
        if not role_dir.exists():
            continue
        for npz in sorted(role_dir.glob("*.npz")):
            d = np.load(npz)
            T = d["angle"].shape[0]
            print(f"{npz.name}: T={T}, speaker={int(d['speaker_id'])}")
```

## 補間の技術的背景

ギャップ補間は以下の方針で行われます:

- **回転 (angle)**: SLERP (球面線形補間) — SO(3) マニフォルド上で短弧を選択し、大角度でも安全
- **線形特徴量 (expression, centroid 等)**: 線形補間 (既定) または PCHIP (Fritsch-Carlson 法、scipy 要)
- **0.4 秒超のギャップ**: 補間せずシーケンスを分割（旧 MediaPipe パイプラインの `FIX_SEC=0.4` を踏襲）
- **100 フレーム未満**: 破棄（旧 `MIN_DATA_SIZE=100` を踏襲）

詳細は設計ドキュメントを参照:
- [`docs/design/interpolation.md`](design/interpolation.md)
- [`docs/design/rotation_interpolation.md`](design/rotation_interpolation.md)
