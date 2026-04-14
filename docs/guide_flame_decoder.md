# Guide D: FLAME 系デコーダの学習と可視化

DECA / SMIRK で抽出した FLAME パラメータから顔を可視化する方法。2 つのモードがあります:

1. **メッシュ可視化** (学習不要) — FLAME パラメトリックモデルでワイヤフレーム描画
2. **ニューラルデコーダ可視化** (学習必要) — 学習済みネットワークでフォトリアル画像生成

## モード 1: FLAME メッシュ可視化（学習不要）

FLAME パラメトリックモデル (`generic_model.pkl`) を使い、3DMM パラメータからメッシュ頂点を計算し、ワイヤフレームで描画します。パラメータのサニティチェックに最適です。

### 必要なファイル

| ファイル | 入手元 |
|----------|--------|
| `checkpoints/flame/generic_model.pkl` | [FLAME website](https://flame.is.tue.mpg.de/) (academic license) |

DECA チェックポイントに同梱されていることもあります (`data/` 以下)。

### 単一 npz の可視化

```bash
python scripts/demo_visualize.py \
    --npz data/movements/data001/comp/deca_comp_00000_04499.npz \
    --mode mesh \
    --flame-model ./checkpoints/flame/generic_model.pkl \
    --output demo_mesh.mp4
```

### ディレクトリ内の全 npz を連結

```bash
python scripts/demo_visualize.py \
    --npz-dir data/movements/data001/comp/ \
    --mode mesh \
    --flame-model ./checkpoints/flame/generic_model.pkl \
    --output demo_all.mp4
```

### フレーム画像として保存

```bash
python scripts/demo_visualize.py \
    --npz data/movements/data001/comp/deca_comp_00000_04499.npz \
    --mode mesh \
    --flame-model ./checkpoints/flame/generic_model.pkl \
    --output-dir demo_frames/ \
    --save-frames
```

### オプション

| オプション | 既定値 | 説明 |
|-----------|--------|------|
| `--image-size` | 512 | 出力画像サイズ (正方形) |
| `--max-frames` | なし | レンダリング最大フレーム数 |

### Python API

```python
from flare.decoders.flame_mesh_renderer import FLAMEMeshRenderer
import numpy as np

renderer = FLAMEMeshRenderer("./checkpoints/flame/generic_model.pkl")

# 単一フレーム
image = renderer.render(
    shape=np.zeros(100),
    expression=np.random.randn(50) * 0.5,
    global_pose=np.array([0.0, 0.3, 0.0]),  # 少し右を向く
    jaw_pose=np.array([0.3, 0.0, 0.0]),     # 口を開ける
    image_size=512,
)

# npz から
data = np.load("movements/data001/comp/deca_comp_00000_04499.npz")
angle_raw = data["angle"] * data["angle_std"] + data["angle_mean"]
exp_raw = data["expression"] * data["expression_std"] + data["expression_mean"]

image = renderer.render(
    shape=data["shape"],
    expression=exp_raw[0],
    global_pose=angle_raw[0],
    jaw_pose=data["jaw_pose"][0] if "jaw_pose" in data else None,
)
```

---

## モード 2: ニューラルデコーダ可視化（学習必要）

対象人物の動画から per-person のニューラルデコーダを学習し、FLAME パラメータからフォトリアルな顔画像を生成します。

### アーキテクチャ

```
ソース画像 (3, 256, 256)
  → ResNet-18 Encoder → 特徴マップ (512, 8, 8)
                                    ↓
FLAME パラメータ (exp + pose + jaw = 56D)
  → MLP → スタイルベクトル (512,) → AdaIN 条件付け
                                    ↓
                          5 段アップサンプリング
                                    ↓
                         生成画像 (3, 256, 256)
```

- **損失関数**: L1 + Perceptual Loss (VGG-16)
- **学習時間**: 100 epoch で数時間 (GPU 依存)
- **推論速度**: ~200 FPS @ 256x256

### ステップ 1: 学習データの準備と学習

対象人物が映っている動画 1 本から、自動的にフレーム抽出・DECA 推論・学習を行います。

```bash
python scripts/train_face_decoder.py \
    --video ./data/multimodal_dialogue_formed/data001/comp.mp4 \
    --deca-path ./checkpoints/deca/deca_model.tar \
    --output-dir ./checkpoints/face_decoder/comp01/ \
    --device cuda:0
```

#### 設定ファイルを使用する場合

```bash
python scripts/train_face_decoder.py \
    --video ./data/multimodal_dialogue_formed/data001/comp.mp4 \
    --deca-path ./checkpoints/deca/deca_model.tar \
    --output-dir ./checkpoints/face_decoder/comp01/ \
    --config configs/train_face_decoder.yaml \
    --device cuda:0
```

[`configs/train_face_decoder.yaml`](../configs/train_face_decoder.yaml) の主要パラメータ:

| パラメータ | 既定値 | 説明 |
|-----------|--------|------|
| `epochs` | 100 | エポック数 (品質重視なら 200-300) |
| `batch_size` | 16 | VRAM 12GB: 16, 24GB: 32 |
| `lr` | 1e-4 | 学習率 |
| `perceptual_weight` | 0.5 | VGG Perceptual Loss の重み (0 で L1 のみ) |
| `max_frames` | 10000 | 動画から抽出する最大フレーム数 |

#### 抽出済み npz を使う場合

`lhg-extract` で既に npz を抽出済みなら、動画 + npz から効率的にデータを構築できます:

```bash
python scripts/train_face_decoder.py \
    --video ./data/multimodal_dialogue_formed/data001/comp.mp4 \
    --npz-dir ./data/movements/data001/comp/ \
    --output-dir ./checkpoints/face_decoder/comp01/ \
    --device cuda:0
```

### ステップ 2: 学習結果の確認

学習中、以下が出力されます:

```
checkpoints/face_decoder/comp01/
├── face_decoder.pth        ← 最良モデル (推論に使用)
├── checkpoint_epoch0100.pth ← エポックごとのチェックポイント
├── source_image.png         ← ソース画像 (推論時に必要)
├── train_config.json        ← 学習設定の記録
└── samples/
    ├── epoch_0001.png       ← 学習進捗サンプル
    ├── epoch_0005.png       ←  (source | predicted | target の横並び)
    └── epoch_0100.png
```

`samples/` のサンプル画像で学習進捗を確認してください:
- 左: ソース画像 (固定)
- 中: 予測画像 (学習とともに改善)
- 右: 正解画像 (各フレーム)

### ステップ 3: 学習済みデコーダで可視化

```bash
python scripts/demo_visualize.py \
    --npz data/movements/data001/comp/deca_comp_00000_04499.npz \
    --mode neural \
    --decoder-path ./checkpoints/face_decoder/comp01/face_decoder.pth \
    --source-image ./checkpoints/face_decoder/comp01/source_image.png \
    --output demo_neural.mp4
```

出力動画は「ソース画像 | 生成画像」の横並びレイアウトです。

### ステップ 4: 複数人物の学習

各人物ごとに別々にデコーダを学習します:

```bash
# comp (対話参加者) 用
python scripts/train_face_decoder.py \
    --video ./data/multimodal_dialogue_formed/data001/comp.mp4 \
    --output-dir ./checkpoints/face_decoder/data001_comp/

# host (ホスト) 用
python scripts/train_face_decoder.py \
    --video ./data/multimodal_dialogue_formed/data001/host.mp4 \
    --output-dir ./checkpoints/face_decoder/data001_host/
```

---

## 全体のワークフロー

```
対面対話動画
  │
  ├── [lhg-extract] → npz (FLAME パラメータ)
  │
  ├── [メッシュ可視化] → ワイヤフレーム動画 (即座に実行可能)
  │     (generic_model.pkl のみ必要)
  │
  └── [ニューラルデコーダ] → フォトリアル動画
        ├── Step 1: train_face_decoder.py (数時間)
        └── Step 2: demo_visualize.py --mode neural
```

## FlashAvatar / FLARE Renderer との違い

| | FaceDecoderNet (本ガイド) | FlashAvatar | PIRender |
|---|---|---|---|
| 方式 | AdaIN CNN | 3D Gaussian Splatting | NeRF + Flow |
| 学習時間 | 数時間 | ~30 分 (RTX 3090) | 事前学習済み |
| 品質 | 中 | 極めて高い | 高い |
| 外部依存 | なし | diff-gaussian-rasterization | PIRender repo |
| 用途 | プロトタイプ・検証 | 本番デモ | BFM ルート |

本ガイドの `FaceDecoderNet` は外部リポジトリ不要で完結する軽量デコーダです。より高品質な可視化が必要な場合は FlashAvatar の使用を推奨します。
