# FLARE: Facial Landmark Analysis & Rendering Engine

LHG (Listening Head Generation) 研究における特徴量抽出 (エンコード) とフォトリアルレンダリング (デコード) の統合ツール。

## 特徴

- **2 ルート対応**: Route A (BFM ベース) と Route B (FLAME ベース)
- **デュアルモード**: リアルタイムモード (Webカメラ入力) と前処理モード (バッチ処理)
- **モジュラー設計**: Extractor / Renderer / LHGModel / Converter を独立交換可能
- **YAML 設定**: pydantic バリデーション付き設定ファイル
- **堅牢性**: 3 段階エラーハンドリング (SKIP / RETRY / ABORT)

## タスク別ガイド

**「やりたいこと」から逆引き** — 各ガイドは独立した手順書です。

| やりたいこと | ガイド |
|---|---|
| LHG システムのリアルタイム動作で特徴抽出器として使う | [Guide A: リアルタイム特徴抽出](docs/guide_realtime.md) |
| LHG モデルの深層学習用に対面対話データから特徴量を一括抽出する | [Guide B: LHG 前処理 (バッチ特徴抽出)](docs/guide_lhg_preprocess.md) |
| BFM 系と FLAME 系を切り替える / 違いを理解する | [Guide C: BFM / FLAME ルート切り替え](docs/guide_route_switching.md) |
| FLAME 系のデコーダを学習して可視化する (DECA / SMIRK) | [Guide D: FLAME デコーダの学習と可視化](docs/guide_flame_decoder.md) |
| BFM 系で抽出したパラメータを可視化する (PIRender) | [Guide E: BFM 可視化 (PIRender)](docs/guide_bfm_visualization.md) |

## クイックスタート

### セットアップ

```bash
# 前提: Python 3.11, CUDA 12.8 対応 GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pydantic pyyaml click loguru opencv-python mediapipe scipy
```

### LHG 前処理 (最もよく使う操作)

```bash
# DECA (FLAME) ルート — 対面対話動画からバッチ特徴抽出
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --config configs/lhg_extract_deca.yaml

# 設定確認のみ (モデル読み込みなし)
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --config configs/lhg_extract_deca.yaml \
    --dry-run
```

### リアルタイムパイプライン

```bash
python examples/realtime_extract.py \
    --config configs/realtime_flame.yaml \
    --source 0   # Webカメラ
```

### 抽出結果の可視化

```bash
# FLAME メッシュ (学習不要)
python scripts/demo_visualize.py \
    --npz data/movements/data001/comp/deca_comp_00000_04499.npz \
    --mode mesh \
    --flame-model ./checkpoints/generic_model.pkl \
    --output demo.mp4

# BFM → PIRender (フォトリアル)
python examples/visualize_bfm_pirender.py \
    --npz data/movements/data001/comp/bfm_comp_00000_04499.npz \
    --source-image ./data/source_portrait.png \
    --pirender-model ./checkpoints/pirender/epoch_00190_*.pt \
    --output demo_bfm.mp4
```

## ルート構成

```
              ┌─────────────────────────────────────────────────┐
              │            Route B (FLAME)                      │
  動画入力 →  │ DECA/SMIRK → [Converter] → FlashAvatar         │ → 顔画像
              │  (shape 100, exp 50)         (3DGS, 300 FPS)   │
              ├─────────────────────────────────────────────────┤
              │            Route A (BFM)                        │
  動画入力 →  │ Deep3D/3DDFA        →        PIRender           │ → 顔画像
              │  (id 80, exp 64)             (NeRF, 100 FPS)   │
              └─────────────────────────────────────────────────┘
```

## ディレクトリ構造

```
FLARE_by_Claude/
├── README.md                    # 本ファイル
├── tool.py                      # CLI エントリポイント
├── configs/                     # YAML 設定ファイル
│   ├── lhg_extract_deca.yaml    # LHG 前処理 (DECA)
│   ├── lhg_extract_bfm.yaml    # LHG 前処理 (BFM)
│   └── train_face_decoder.yaml # デコーダ学習設定
├── scripts/                     # 実行スクリプト
│   ├── train_face_decoder.py   # FLAME デコーダ学習
│   └── demo_visualize.py       # 可視化デモ (mesh / neural)
├── examples/                    # サンプルスクリプト
│   ├── realtime_extract.py     # リアルタイムパイプライン起動
│   └── visualize_bfm_pirender.py # BFM + PIRender 可視化
├── docs/                        # ドキュメント
│   ├── guide_realtime.md        # Guide A
│   ├── guide_lhg_preprocess.md  # Guide B
│   ├── guide_route_switching.md # Guide C
│   ├── guide_flame_decoder.md   # Guide D
│   ├── guide_bfm_visualization.md # Guide E
│   └── design/                  # 設計ドキュメント
│       ├── lhg_extract_pipeline.md
│       ├── interpolation.md
│       └── rotation_interpolation.md
├── flare/                       # メインパッケージ
│   ├── cli.py                   # Click CLI (extract / render / lhg-extract)
│   ├── config.py                # YAML + pydantic 設定管理
│   ├── extractors/              # 3DMM 特徴抽出
│   │   ├── base.py              # BaseExtractor ABC
│   │   ├── deca.py              # DECA (FLAME)
│   │   ├── deep3d.py            # Deep3DFaceRecon (BFM)
│   │   ├── smirk.py             # SMIRK (FLAME)
│   │   └── tdddfa.py            # 3DDFA (BFM)
│   ├── renderers/               # 画像レンダリング
│   │   ├── base.py              # BaseRenderer ABC
│   │   ├── flashavatar.py       # FlashAvatar (FLAME, 3DGS)
│   │   ├── pirender.py          # PIRender (BFM, NeRF)
│   │   └── headgas.py           # HeadGaS (FLAME, 3DGS)
│   ├── decoders/                # デコーダ (可視化用)
│   │   ├── flame_mesh_renderer.py # FLAME メッシュ描画 (学習不要)
│   │   └── face_decoder_net.py  # ニューラル顔デコーダ (学習必要)
│   ├── converters/              # パラメータ変換
│   │   ├── deca_to_flame.py     # DECA → FlashAvatar
│   │   ├── bfm_to_flame.py      # BFM → FLAME
│   │   ├── flame_to_pirender.py # FLAME → PIRender
│   │   └── registry.py          # アダプタレジストリ
│   ├── model_interface/         # LHG モデルインターフェース
│   │   ├── base.py              # BaseLHGModel ABC
│   │   ├── l2l.py               # Learning to Listen
│   │   └── vico.py              # ViCo
│   ├── pipeline/                # パイプライン
│   │   ├── realtime.py          # リアルタイム (5 スレッド)
│   │   ├── batch.py             # バッチ処理
│   │   ├── lhg_batch.py         # LHG 特徴抽出バッチ
│   │   ├── buffer.py            # PipelineBuffer
│   │   └── frame_drop.py        # フレームドロップポリシー
│   └── utils/                   # ユーティリティ
│       ├── face_detect.py       # 顔検出 (MediaPipe)
│       ├── interp.py            # 線形 / PCHIP 補間
│       ├── rotation_interp.py   # SLERP 回転補間
│       ├── video.py             # 動画 I/O
│       ├── visualization.py     # 可視化ヘルパー
│       ├── metrics.py           # FPS 計測
│       ├── errors.py            # エラーポリシー
│       └── logging.py           # Loguru 設定
└── tests/                       # テストスイート (335 tests)
```

## 必要なチェックポイント

| ファイル | 用途 | 入手元 |
|----------|------|--------|
| `checkpoints/deca_model.tar` | DECA 推論 | [DECA repo](https://github.com/yfeng95/DECA) |
| `checkpoints/deep3d_epoch20.pth` | Deep3DFaceRecon 推論 | [Deep3DFaceRecon repo](https://github.com/sicxu/Deep3DFaceRecon_pytorch) |
| `checkpoints/generic_model.pkl` | FLAME メッシュ可視化 | [FLAME website](https://flame.is.tue.mpg.de/) |
| `checkpoints/pirender/*.pt` | PIRender (BFM 可視化) | [PIRender repo](https://github.com/RenYurui/PIRender) |
| `checkpoints/flashavatar/` | FlashAvatar (FLAME 可視化) | [FlashAvatar repo](https://github.com/MingSHTM/FlashAvatar) |

## CLI コマンド一覧

```bash
python tool.py --help          # 全コマンド表示
python tool.py extract --help  # バッチ特徴抽出
python tool.py render --help   # バッチレンダリング
python tool.py lhg-extract --help  # LHG 前処理
```

## テスト

```bash
pytest tests/ -v    # 全 335 テスト実行
```

## ライセンス

MIT License
