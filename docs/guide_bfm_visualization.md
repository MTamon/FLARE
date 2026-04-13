# Guide E: BFM 系の可視化 (PIRender)

Deep3DFaceRecon / 3DDFA で抽出した BFM パラメータから、PIRender を使ってフォトリアルな顔画像を生成する方法。

## 概要

BFM ルート (Route A) の可視化には **PIRender** を使用します。PIRender は NeRF ベースの顔レンダラで、ソース画像の外見を保持しつつ、BFM パラメータ (exp + pose + trans) で表情と姿勢を制御します。

```
ソース画像 (正面顔写真)
  → PIRender.setup() → モーションディスクリプタ抽出
                            ↓
BFM パラメータ (exp 64D + pose 6D + trans 3D)
  → PIRender.render() → フローフィールド推定 → ソース画像ワープ
                            ↓
                       出力画像 (3, 256, 256)
```

**FLAME 系との違い**: PIRender はソース画像が必須です (FlashAvatar は不要)。対象人物の正面顔写真を 1 枚用意してください。

## 前提条件

### 必要なファイル

| ファイル | 説明 | 入手元 |
|----------|------|--------|
| PIRender チェックポイント | `checkpoints/pirender/epoch_00190_*.pt` | [PIRender repo](https://github.com/RenYurui/PIRender) |
| PIRender リポジトリ | ソースコード (import に必要) | `git clone https://github.com/RenYurui/PIRender.git` |
| ソース画像 | 対象人物の正面顔写真 | 動画から手動選定 or 自動選定 |

### PIRender のセットアップ

```bash
# PIRender リポジトリをクローン
git clone https://github.com/RenYurui/PIRender.git
cd PIRender
pip install -r requirements.txt
cd ..

# チェックポイントを配置
mkdir -p checkpoints/pirender/
# PIRender のリリースから epoch_00190_iteration_000400000_checkpoint.pt をダウンロード
```

## 抽出済み npz からの可視化

BFM ルートで抽出した npz を PIRender で可視化するサンプルスクリプトを用意しています。

### サンプルスクリプトの実行

```bash
python examples/visualize_bfm_pirender.py \
    --npz data/movements/data001/comp/bfm_comp_00000_04499.npz \
    --source-image ./data/source_portrait.png \
    --pirender-model ./checkpoints/pirender/epoch_00190_iteration_000400000_checkpoint.pt \
    --pirender-dir ./PIRender \
    --output demo_bfm.mp4 \
    --device cuda:0
```

詳細は [`examples/visualize_bfm_pirender.py`](../examples/visualize_bfm_pirender.py) を参照。

### オプション

| オプション | 既定値 | 説明 |
|-----------|--------|------|
| `--npz` | (必須) | 抽出済み npz ファイルパス |
| `--npz-dir` | なし | npz ディレクトリ (全ファイルを連結) |
| `--source-image` | (必須) | ソース画像 (対象人物の正面顔) |
| `--pirender-model` | (必須) | PIRender チェックポイントパス |
| `--pirender-dir` | `./PIRender` | PIRender リポジトリパス |
| `--output` | `output_bfm.mp4` | 出力動画パス |
| `--device` | `cuda:0` | 推論デバイス |
| `--max-frames` | なし | 最大レンダリングフレーム数 |

## Python API での使用

```python
import numpy as np
import torch
from flare.renderers.pirender import PIRenderRenderer

# 1. レンダラ初期化
renderer = PIRenderRenderer(
    model_path="./checkpoints/pirender/epoch_00190_iteration_000400000_checkpoint.pt",
    device="cuda:0",
    pirender_dir="./PIRender",  # PIRender リポジトリのパス
)

# 2. ソース画像でセットアップ (必須)
import cv2
src = cv2.imread("./data/source_portrait.png")
src_rgb = cv2.cvtColor(cv2.resize(src, (256, 256)), cv2.COLOR_BGR2RGB)
src_tensor = torch.from_numpy(src_rgb).permute(2, 0, 1).float() / 255.0
src_tensor = src_tensor.unsqueeze(0).to("cuda:0")  # (1, 3, 256, 256)
renderer.setup(source_image=src_tensor)

# 3. npz 読み込み + デノーマライズ
data = np.load("data/movements/data001/comp/bfm_comp_00000_04499.npz")
exp_raw = data["expression"] * data["expression_std"] + data["expression_mean"]
angle_raw = data["angle"] * data["angle_std"] + data["angle_mean"]
centroid_raw = data["centroid"] * data["centroid_std"] + data["centroid_mean"]

# 4. 1 フレーム分のレンダリング
exp_t = torch.from_numpy(exp_raw[0:1]).float().to("cuda:0")     # (1, 64)
pose_t = torch.from_numpy(
    np.concatenate([angle_raw[0:1], centroid_raw[0:1, :3]], axis=-1)
).float().to("cuda:0")                                           # (1, 6)
trans_t = torch.from_numpy(centroid_raw[0:1]).float().to("cuda:0")  # (1, 3)

output = renderer.render({
    "exp": exp_t,
    "pose": pose_t,
    "trans": trans_t,
})
# output: (1, 3, 256, 256), 値域 [0, 1]
```

## ソース画像の選定

PIRender の出力品質はソース画像の品質に大きく依存します。以下の条件を満たす画像を選んでください:

- 正面顔 (yaw / pitch ともに小さい)
- 均一な照明 (影が少ない)
- 自然な表情 (極端な表情でない)
- 解像度 256x256 以上

動画から自動選定する例:

```python
# 動画の中央付近のフレームを使うことが多い
import cv2
cap = cv2.VideoCapture("data/multimodal_dialogue_formed/data001/comp.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ret, frame = cap.read()
cap.release()
cv2.imwrite("source_portrait.png", frame)
```

## FLAME パラメータから BFM への変換

DECA (FLAME) で抽出した npz を PIRender で可視化したい場合は、`FlameToPIRenderAdapter` で変換します:

```python
from flare.converters.flame_to_pirender import FlameToPIRenderAdapter

adapter = FlameToPIRenderAdapter()
pirender_params = adapter.convert({
    "expr": torch.from_numpy(exp_raw[0:1]).unsqueeze(0),  # (1, 50+)
    "jaw_pose": ...,  # rotation_6d (1, 6)
    "rotation": ...,  # axis-angle (1, 3)
})
# → {"exp": (1, 64), "pose": (1, 6), "trans": (1, 3)}
```

ただし FLAME→BFM の変換は近似的であり、情報の欠損が生じます。BFM ルートの可視化には Deep3DFaceRecon で抽出した npz を直接使用することを推奨します。

## FLAME メッシュ可視化との比較

| | PIRender (BFM) | FLAME メッシュ |
|---|---|---|
| 学習 | 不要 (事前学習済みモデルを使用) | 不要 |
| 出力 | フォトリアル画像 | ワイヤフレーム |
| 外部依存 | PIRender repo + チェックポイント | generic_model.pkl のみ |
| ソース画像 | 必須 | 不要 |
| 速度 | ~100 FPS | ~300 FPS |

BFM パラメータの定性チェックだけなら、Guide D のメッシュ可視化 (`--mode mesh`) でも十分です。ただしメッシュ可視化は FLAME パラメータ用なので、BFM で抽出した場合は PIRender を使うか、`BFMToFlameAdapter` で変換してからメッシュ可視化してください。
