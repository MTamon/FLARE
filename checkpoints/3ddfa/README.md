# 3DDFA V2 チェックポイントの準備

## 必要ファイル

| ファイル | 説明 |
|----------|------|
| `mb1_120x120.onnx` | 3DDFA V2 MobileNetV1 ONNX モデル |
| `bfm_noneck_v3.pkl` | BFM 基底データ (3DDFA V2 用) |

## 入手手順

### 方法 1: FLARE 自動ダウンロードスクリプト

```bash
python scripts/download_checkpoints.py --model 3ddfa
# → リポジトリを一時クローンし、必要ファイルをコピー
```

### 方法 2: gdown で直接ダウンロード + リポジトリからコピー

```bash
# ONNX モデルのダウンロード (リポジトリに同梱されていないため別途必要)
pip install gdown
gdown 1YpO1KfXvJHRmCBkErNa62dHm-CUjsoIk -O checkpoints/3ddfa/mb1_120x120.onnx

# BFM データはリポジトリに同梱されている
git clone --depth 1 https://github.com/cleardusk/3DDFA_V2.git /tmp/3DDFA_V2
cp /tmp/3DDFA_V2/configs/bfm_noneck_v3.pkl checkpoints/3ddfa/
rm -rf /tmp/3DDFA_V2
```

### 方法 3: リポジトリの完全セットアップ

3DDFA_V2 リポジトリのフル機能を使いたい場合:

```bash
git clone https://github.com/cleardusk/3DDFA_V2.git
cd 3DDFA_V2

# 依存パッケージ
pip install -r requirements.txt

# Cython/C モジュールのビルド (Linux/Mac のみ)
sh ./build.sh
# → FaceBoxes (CPU NMS), Sim3DR (3D rendering), render.so がコンパイルされる
# Windows の場合: Visual Studio Build Tools が必要。ONNX 推論のみなら省略可。

# ONNX モデルのダウンロード
gdown 1YpO1KfXvJHRmCBkErNa62dHm-CUjsoIk -O weights/mb1_120x120.onnx

# 動作確認
python demo.py -f examples/inputs/emma.jpg --onnx

# FLARE 用にコピー
cp weights/mb1_120x120.onnx ../FLARE_by_Claude/checkpoints/3ddfa/
cp configs/bfm_noneck_v3.pkl ../FLARE_by_Claude/checkpoints/3ddfa/
```

## 配置後の構成

```
checkpoints/3ddfa/
├── mb1_120x120.onnx
└── bfm_noneck_v3.pkl
```

## リポジトリ内のファイル配置 (参考)

```
3DDFA_V2/
├── weights/
│   ├── mb1_120x120.pth     ← PyTorch 版 (同梱)
│   └── mb1_120x120.onnx    ← ONNX 版 (要ダウンロード)
├── configs/
│   ├── bfm_noneck_v3.pkl   ← BFM 基底データ (同梱)
│   ├── param_mean_std_62d_120x120.pkl
│   ├── tri.pkl
│   └── mb1_120x120.yml
├── FaceBoxes/               ← Cython NMS (要ビルド)
├── Sim3DR/                  ← 3D rendering (要ビルド)
└── build.sh                 ← ビルドスクリプト
```

## 注意事項

- `.pth` ファイルはリポジトリに同梱されていますが、`.onnx` ファイルは別途ダウンロードが必要です
- 3DDFA_V2 リポジトリを `sys.path` に追加する必要があります (`tddfa_dir` パラメータで指定)
- `build.sh` は Linux/Mac 向けです。Windows では WSL2 または Git Bash で実行してください
- macOS の場合: `brew install libomp` が必要です (ONNX Runtime の依存)
- 3DDFA V2 は CPU 推論でも 1.35ms/image と極めて高速です

## 関連ガイド

- [環境構築ガイド](../../docs/guide_setup.md#24-3ddfa_v2) — セットアップの詳細
- [Guide A: リアルタイム特徴抽出](../../docs/guide_realtime.md) — BFM ルートの軽量 Extractor
- [Guide C: ルート切り替え](../../docs/guide_route_switching.md) — Extractor の選択指針
