# Guide A: リアルタイム特徴抽出パイプライン

LHG (Listening Head Generation) システムのリアルタイム動作時に、FLARE をリアルタイム 3DMM 特徴抽出器として使用する方法。

## 概要

`RealtimePipeline` は 5 スレッドのマルチスレッド構成で動作し、Webカメラまたは動画ファイルからの映像をリアルタイムに処理します。

```
Webカメラ/動画
  → [キャプチャスレッド] フレーム取得 (CPU)
  → [抽出スレッド]     顔検出 + 3DMM パラメータ抽出 (GPU)
  → [推論スレッド]     LHG モデル推論 (GPU)
  → [レンダリングスレッド] 顔画像生成 (GPU)
  → [表示スレッド]     GUI 表示 (CPU)
```

各スレッド間は `PipelineBuffer` で非同期に接続され、`Latest-Frame-Wins` ポリシーにより処理遅延時は最新フレームを優先します。

## 前提条件

- FLARE 環境のセットアップ済み → [環境構築ガイド](guide_setup.md)
- CUDA 対応 GPU (RTX 2080 Ti 以上推奨)
- チェックポイントファイル:

| ルート | Extractor | チェックポイント |
|--------|-----------|-----------------|
| FLAME  | DECA      | `checkpoints/deca/deca_model.tar` |
| FLAME  | SMIRK     | `checkpoints/smirk/smirk_encoder.pt` |
| BFM    | Deep3D    | `checkpoints/deep3d/deep3d_epoch20.pth` |
| BFM    | 3DDFA     | `checkpoints/3ddfa/mb1_120x120.onnx` |

- レンダラチェックポイント:

| ルート | Renderer      | チェックポイント |
|--------|--------------|-----------------|
| FLAME  | FlashAvatar  | `checkpoints/flashavatar/` (対象人物ごとに学習済み) |
| BFM    | PIRender     | `checkpoints/pirender/epoch_00190_*.pt` |

各チェックポイントディレクトリの README.md に入手方法が記載されています。

## 設定ファイルの作成

YAML 設定ファイルでパイプライン全体を制御します。

### FLAME ルート (DECA + FlashAvatar)

```yaml
# configs/realtime_flame.yaml
pipeline:
  name: realtime_flame
  fps: 30
  device: cuda:0
  converter_chain:
    - type: deca_to_flame

extractor:
  type: deca
  model_path: ./checkpoints/deca/deca_model.tar
  input_size: 224

renderer:
  type: flash_avatar
  model_path: ./checkpoints/flashavatar/person01/
  source_image: null  # FlashAvatar は不要
  source_image: null  # FlashAvatar は不要
  output_size: [512, 512]

lhg_model:
  type: learning2listen
  model_path: ./checkpoints/l2l/l2l_vqvae.pth
  window_size: 64

audio:
  sample_rate: 16000
  feature_type: mel

buffer:
  max_size: 256
  timeout_sec: 0.5
  overflow_policy: drop_oldest

device_map:
  extractor: cuda:0
  lhg_model: cuda:0
  renderer: cuda:0
```

### BFM ルート (Deep3D + PIRender)

```yaml
# configs/realtime_bfm.yaml
pipeline:
  name: realtime_bfm
  fps: 30
  device: cuda:0
  converter_chain: []  # BFM → PIRender は変換不要

extractor:
  type: deep3d
  model_path: ./checkpoints/deep3d/deep3d_epoch20.pth
  input_size: 224

renderer:
  type: pirender
  model_path: ./checkpoints/pirender/epoch_00190_iteration_000400000_checkpoint.pt
  source_image: ./data/source_images/source_portrait.png  # PIRender はソース画像が必要
  output_size: [256, 256]

device_map:
  extractor: cuda:0
  lhg_model: cuda:0
  renderer: cuda:0
```

## 実行方法

### Python コードから直接

```python
from flare.config import PipelineConfig
from flare.pipeline.realtime import RealtimePipeline

# YAML 設定読み込み
config = PipelineConfig.from_yaml("configs/realtime_flame.yaml")

# Webカメラ入力 (デバイス 0)
pipeline = RealtimePipeline(source=0, display_backend="opencv")
pipeline.run(config)  # 'q' キーで停止
```

### 動画ファイル入力

```python
pipeline = RealtimePipeline(source="./data/test_video.mp4")
pipeline.run(config)
```

### PyQt6 GUI 表示

```python
pipeline = RealtimePipeline(source=0, display_backend="pyqt")
pipeline.run(config)
```

### サンプルスクリプト

すぐに実行できるサンプルスクリプトを用意しています:

```bash
python examples/realtime_extract.py \
    --config configs/realtime_flame.yaml \
    --source 0
```

詳細は [`examples/realtime_extract.py`](../examples/realtime_extract.py) を参照。

## マルチ GPU 構成

複数 GPU がある場合、各コンポーネントを異なるデバイスに配置できます:

```yaml
device_map:
  extractor: cuda:0   # GPU 0: DECA 推論
  lhg_model: cuda:1   # GPU 1: LHG モデル推論
  renderer: cuda:1     # GPU 1: レンダリング
```

## パフォーマンスチューニング

| パラメータ | 説明 | 推奨値 |
|-----------|------|--------|
| `buffer.max_size` | バッファサイズ (フレーム) | 128-256 |
| `buffer.timeout_sec` | バッファ読み取りタイムアウト | 0.3-0.5 |
| `buffer.overflow_policy` | オーバーフロー時の方針 | `drop_oldest` (リアルタイム用) |
| `pipeline.fps` | 目標 FPS | 30 |

- `drop_oldest`: 古いフレームを破棄（リアルタイム表示向け、遅延最小）
- `block`: 空くまで待機（全フレーム処理が必要な場合）
- `interpolate`: 補間挿入（実験的）

## LHG モデルとの連携

リアルタイムパイプラインでは、Extractor で抽出した話者の 3DMM パラメータを LHG モデルに入力し、聞き手の頭部動作パラメータを予測します。予測結果は Renderer で可視化されます。

```
[話者映像] → Extractor → 話者の 3DMM パラメータ
                              ↓
[話者音声] → Audio特徴量 →  LHG Model → 聞き手の 3DMM パラメータ
                                              ↓
                                         Renderer → 聞き手顔画像
```

対応 LHG モデル:
- **Learning2Listen (L2L)**: VQ-VAE ベース、window_size=64
- **ViCo**: Transformer ベース

## トラブルシューティング

| 症状 | 原因と対策 |
|------|-----------|
| CUDA out of memory | `device_map` で GPU を分散、または `buffer.max_size` を減らす |
| 映像が表示されない | `display_backend` を `"opencv"` に変更。SSH 経由の場合 X forwarding 要 |
| FPS が低い | `overflow_policy` を `drop_oldest` に。Extractor の `input_size` を 160 に |
| 顔が検出されない | 照明条件を確認。MediaPipe は暗所に弱い |
