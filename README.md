# FLARE: Facial Landmark Analysis & Rendering Engine

LHG（Listening Head Generation）研究における特徴量抽出（エンコード）とフォトリアルレンダリング（デコード）の統合ツール。

## 特徴

- **2ルート対応**: Route A（BFMベース）と Route B（FLAMEベース）
- **デュアルモード**: リアルタイムモード（Webcam入力）と前処理モード（バッチ処理）
- **モジュラー設計**: Extractor / Renderer / LHGModel / Converter を独立交換可能
- **YAML設定**: pydantic バリデーション付き設定ファイル
- **堅牢性**: 3段階エラーハンドリング（SKIP / RETRY / ABORT）

## ディレクトリ構造

```
flare/
├── __init__.py              # パッケージ初期化、バージョン定義
├── config.py                # YAML設定 + pydantic バリデーション
├── cli.py                   # Click CLI（extract / render サブコマンド）
├── extractors/
│   ├── base.py              # BaseExtractor ABC
│   ├── deca.py              # DECA Extractor (Phase 2)
│   └── deep3d.py            # Deep3DFaceRecon Extractor (Phase 3)
├── renderers/
│   ├── base.py              # BaseRenderer ABC (setup/render分離)
│   ├── flashavatar.py       # FlashAvatar Renderer (Phase 2)
│   └── pirender.py          # PIRender Renderer (Phase 3)
├── model_interface/
│   └── base.py              # BaseLHGModel ABC (2引数predict)
├── converters/
│   ├── base.py              # BaseAdapter ABC
│   ├── registry.py          # AdapterRegistry (自動選択)
│   └── deca_to_flame.py     # DECA→FLAME変換 (ゼロパディング)
├── pipeline/
│   ├── buffer.py            # PipelineBuffer (queue.Queue)
│   ├── batch.py             # バッチ処理パイプライン
│   └── realtime.py          # リアルタイムパイプライン
└── utils/
    ├── errors.py            # ErrorPolicy + カスタム例外
    ├── logging.py           # Loguru ロギング設定
    ├── video.py             # 動画 I/O
    ├── face_detect.py       # 顔検出 (MediaPipe)
    ├── metrics.py           # FPS計測・統計
    └── visualization.py     # 可視化ユーティリティ
```

## セットアップ

### 前提条件

- Python 3.11.0（pyenv推奨）
- CUDA 12.8 対応GPU（RTX 2080 Ti以上）

### 環境構築

```bash
# PyTorch + 依存パッケージのインストール
bash install/build_environment.sh --cuda 12.8

# pytorch3d（ソースビルド）
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e . && cd ..
```

### テスト実行

```bash
pytest tests/ -v
```

## CLI 使用例

### バッチ特徴量抽出

```bash
python tool.py extract \
    --input-dir /data/videos/ \
    --output-dir /data/features/ \
    --route flame --extractor deca \
    --gpu 0 --batch-size 32
```

### バッチレンダリング

```bash
python tool.py render \
    --input-dir /data/features/ \
    --output-dir /data/rendered/ \
    --route flame --renderer flashavatar \
    --avatar-model /models/avatar_001/ --resolution 512,512
```

## 設定ファイル

`config.yaml` で全コンポーネントのパラメータを管理。詳細は仕様書§8.7を参照。

```yaml
pipeline:
  name: "lhg_realtime_v1"
  fps: 30
  device: "cuda:0"

extractor:
  type: "deca"
  model_path: "./checkpoints/deca_model.tar"

renderer:
  type: "flash_avatar"
  model_path: "./checkpoints/flashavatar/"
```

## ライセンス

研究用途限定。
