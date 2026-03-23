# FLARE: Facial Landmark Analysis & Rendering Engine

LHG（Listening Head Generation）研究における特徴量抽出（エンコード）とフォトリアルレンダリング（デコード）の統合ツール。

## 概要

FLAREは、話者の音声・映像から聞き手の自然な頭部動作を生成するLHG研究パイプラインにおいて、3D Morphable Model（3DMM）のパラメータ抽出と、得られたパラメータからのフォトリアルな顔画像レンダリングを統合的に提供します。

主な機能:

- **2つの処理ルート**: BFMベース（ルートA）とFLAMEベース（ルートB）をサポート
- **リアルタイムモード**: Webカメラ入力からリアルタイムに処理・表示（目標30FPS以上）
- **バッチモード**: データセットの一括前処理とチェックポイントによる中断・再開
- **モジュラー設計**: Extractor / Renderer / Converter の独立交換が可能

## 処理ルート

### ルートB（FLAME） — **推奨**

最新研究（CVPR 2024/2025）の多くがFLAMEベースであり、L2L（Learning to Listen）との直接互換性を持ちます。

| コンポーネント | ツール | 速度 |
|---|---|---|
| Extractor | DECA (~119 FPS) / SMIRK | SIGGRAPH 2021 / CVPR 2024 |
| Renderer | FlashAvatar (300 FPS) / HeadGaS (250 FPS) | CVPR 2024 |
| Converter | DECAToFlameAdapter（ゼロパディング） | 同一PCA空間 |

DECA exp 50DとFlashAvatar expr 100Dは同一のFLAME expression PCA空間に属するため、ゼロパディング（`F.pad(exp, (0, 50), value=0.0)`）で正確に変換可能です。

### ルートA（BFM） — ViCo評価互換

ViCo Challenge Baselineとの互換性を維持するための並行サポートです。

| コンポーネント | ツール | 速度 |
|---|---|---|
| Extractor | Deep3DFaceRecon (~50 FPS) / 3DDFA V2 (~740 FPS) | |
| Renderer | PIRender (~20-30 FPS) | |
| Converter | FlameToPIRenderAdapter（近似変換） | |

## 環境構築

### 1. Python 3.11.0 セットアップ（pyenv）

```bash
pyenv install 3.11.0
pyenv local 3.11.0
python --version  # Python 3.11.0
```

### 2. PyTorch 2.9.0 + CUDA 12.8 インストール

```bash
bash install/python3.11.0/install_torch.sh --cuda 12.8
```

内容: `torch==2.9.0`, `torchvision==0.24.0`, `torchaudio==2.9.0` + 関連パッケージ（torchao, torchmetrics等）。

### 3. 全依存パッケージインストール

```bash
bash install/build_environment.sh --cuda 12.8
```

約120パッケージを固定バージョンで一括インストールします（numpy 2.2.6, scipy 1.16.3, pydantic 2.12.4, loguru, librosa, transformers 4.57.1 等）。

### 4. pytorch3d ソースビルド

PyTorch 2.9.0環境では公式wheelが未提供のため、ソースからビルドします。`axis_angle_to_matrix` / `matrix_to_rotation_6d` は純PyTorchテンソル演算であり、C++/CUDA拡張に依存しないため互換性は保証されます。

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
```

## CLIの使い方

### バッチ特徴量抽出

```bash
python -m flare.cli extract \
    --input-dir /data/videos/ \
    --output-dir /data/features/ \
    --route flame --extractor deca \
    --gpu 0 --batch-size 32 --resume
```

### バッチレンダリング

```bash
python -m flare.cli render \
    --input-dir /data/features/ \
    --output-dir /data/rendered/ \
    --route flame --renderer flashavatar \
    --avatar-model /models/avatar_001/ --resolution 512
```

### リアルタイムモード

```bash
python -m flare.cli realtime \
    --config config.yaml --camera-id 0 --gpu 0
```

リアルタイムモードでは「q」キーで終了、Ctrl+Cでもgraceful shutdownします。

## モジュール構成

```
flare/
├── __init__.py                    # パッケージ初期化・re-export
├── config.py                      # YAML + pydantic v2 設定管理
├── cli.py                         # Click CLIインターフェース
├── extractors/
│   ├── __init__.py
│   ├── base.py                    # BaseExtractor ABC (Tensor版)
│   ├── deca.py                    # DECA (FLAME, SIGGRAPH 2021)
│   ├── smirk.py                   # SMIRK (FLAME, CVPR 2024)
│   ├── deep3d.py                  # Deep3DFaceRecon (BFM)
│   └── tdddfa.py                  # 3DDFA V2 (BFM, CPU高速)
├── renderers/
│   ├── __init__.py
│   ├── base.py                    # BaseRenderer ABC (setup/render分離)
│   ├── flashavatar.py             # FlashAvatar (FLAME, 300 FPS)
│   ├── headgas.py                 # HeadGaS (FLAME, 250 FPS)
│   └── pirender.py                # PIRender (BFM, GAN系)
├── converters/
│   ├── __init__.py
│   ├── base.py                    # BaseAdapter ABC
│   ├── registry.py                # AdapterRegistry (自動選択)
│   ├── deca_to_flame.py           # DECA→FlashAvatar (ゼロパディング)
│   ├── flame_to_pirender.py       # FLAME→PIRender (近似変換)
│   └── identity.py                # IdentityAdapter (パススルー)
├── pipeline/
│   ├── __init__.py
│   ├── buffer.py                  # PipelineBuffer (queue.Queue)
│   ├── batch.py                   # バッチ処理パイプライン
│   └── realtime.py                # リアルタイムパイプライン (5スレッド)
├── model_interface/
│   ├── __init__.py
│   └── base.py                    # BaseLHGModel ABC (2引数版)
├── utils/
│   ├── __init__.py
│   ├── errors.py                  # カスタム例外 + ErrorPolicy
│   ├── logging.py                 # Loguru ロギング設定
│   ├── video.py                   # VideoReader / VideoWriter
│   ├── face_detect.py             # 顔検出 (MediaPipe / Haar)
│   ├── metrics.py                 # FPSCounter / PipelineMetrics
│   ├── visualization.py           # 描画ユーティリティ
│   ├── mediapipe_eyes.py          # MediaPipe eye pose推定
│   └── benchmark.py               # パイプラインベンチマーク
└── README.md
```

## 設定ファイル

`config.yaml` のサンプル:

```yaml
pipeline:
  name: "lhg_realtime_v1"
  fps: 30
  device: "cuda:0"
  converter_chain:
    - type: deca_to_flame
    - type: identity

extractor:
  type: "deca"
  model_path: "./checkpoints/deca_model.tar"
  input_size: 224
  return_keys: ["shape", "exp", "pose", "detail"]

lhg_model:
  type: "learning2listen"
  model_path: "./checkpoints/l2l_vqvae.pth"
  window_size: 64
  codebook_size: 256

renderer:
  type: "flash_avatar"
  model_path: "./checkpoints/flashavatar/"
  source_image: "./data/source_portrait.png"
  output_size: [512, 512]

audio:
  sample_rate: 16000
  feature_type: "mel"       # or "hubert", "wav2vec2"
  n_mels: 128

buffer:
  max_size: 256
  timeout_sec: 0.5
  overflow_policy: "drop_oldest"

device_map:
  extractor: "cuda:0"
  lhg_model: "cuda:0"
  renderer: "cuda:0"

logging:
  level: "INFO"
  file: "./logs/pipeline.log"
  rotation: "10 MB"

checkpoint:
  enabled: true
  save_dir: "./checkpoints/batch/"
  format: "json"
```

各セクションの詳細は `flare/config.py` のpydanticモデル定義を参照してください。

## 必要環境

| コンポーネント | バージョン |
|---|---|
| Python | 3.11.0 |
| PyTorch | 2.9.0 (CUDA 12.8) |
| pytorch3d | 0.7.8 (ソースビルド) |
| numpy | 2.2.6 |
| pydantic | 2.12.4 |
| opencv-python | 4.12.0.88 |
| mediapipe | 0.10.11 |
| loguru | 0.7+ |

GPU要件: RTX 2080 Ti (11GB VRAM) 以上。推奨: RTX 3090/4090 (24GB VRAM)。

## ライセンス

本プロジェクトは研究目的で開発されています。各コンポーネントのライセンスは以下の通りです:

- **DECA**: [SIGGRAPH 2021] — 非商用研究ライセンス
- **SMIRK**: [CVPR 2024] — 各リポジトリのライセンスに従う
- **FlashAvatar**: [CVPR 2024] — 各リポジトリのライセンスに従う
- **HeadGaS**: Gaussian Splatting系 — 各リポジトリのライセンスに従う
- **Deep3DFaceRecon**: — 非商用研究ライセンス
- **PIRender**: — 各リポジトリのライセンスに従う
- **3DDFA V2**: — 各リポジトリのライセンスに従う
- **FLAME**: — Max Planck Gesellschaft ライセンス
- **BFM**: — Basel Face Model 登録制ダウンロード

利用にあたっては各元リポジトリのライセンス条項を遵守してください。
