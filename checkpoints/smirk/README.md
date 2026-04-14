# SMIRK チェックポイントの準備

## 必要ファイル

| ファイル | 説明 |
|----------|------|
| `smirk_encoder.pt` | SMIRK エンコーダモデル (`SMIRK_em1.pt`) |

## 入手手順

### 方法 1: quick_install.sh で一括セットアップ (推奨)

```bash
git clone https://github.com/georgeretsi/smirk.git
cd smirk

# 環境構築
conda create -n smirk python=3.9 -y
conda activate smirk
pip install -r requirements.txt

# pytorch3d のインストール (CUDA 11.7 + PyTorch 2.0.1 の場合)
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html

# 全モデルの一括ダウンロード
bash quick_install.sh
# → FLAME ライセンスの username/password を聞かれる
#   (https://flame.is.tue.mpg.de/ で事前にアカウント作成が必要)

# FLARE 用にコピー
cp pretrained_models/SMIRK_em1.pt \
   ../FLARE_by_Claude/checkpoints/smirk/smirk_encoder.pt
```

### 方法 2: 手動ダウンロード

```bash
# SMIRK モデル本体
pip install gdown
gdown 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE -O checkpoints/smirk/smirk_encoder.pt
```

> **注意**: SMIRK の推論には SMIRK モデル以外にも以下のアセットが必要です。
> これらは SMIRK リポジトリの `assets/` に配置する必要があります。

| ファイル | 配置先 | ダウンロード元 | ライセンス |
|----------|--------|----------------|------------|
| `SMIRK_em1.pt` | `pretrained_models/` | [Google Drive](https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE/view) | MIT |
| `FLAME2020/` | `assets/` | https://flame.is.tue.mpg.de/ | 要登録 (MPI) |
| `face_landmarker.task` | `assets/` | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task | Apache 2.0 |
| `ResNet50/` (EMOCA) | `assets/` | https://emoca.is.tue.mpg.de/ | 要登録 (MPI) |
| `mica.tar` | `assets/` | https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c0f96de/?dl=1 | 要確認 |
| expression_templates | `assets/` | [Google Drive](https://drive.google.com/file/d/1wEL7KPHw2kl5DxP0UAB3h9QcQLXk7BM_/view) `gdown 1wEL7KPHw2kl5DxP0UAB3h9QcQLXk7BM_` | MIT |

## 配置後の構成

```
checkpoints/smirk/
└── smirk_encoder.pt
```

## 環境情報

| 項目 | 要件 |
|------|------|
| Python | 3.9 |
| PyTorch | 2.0.1 |
| CUDA | 11.7 |
| pytorch3d | 必要 (事前ビルド wheel を使用) |

> pytorch3d の wheel URL は CUDA / PyTorch バージョンに依存します。
> 他のバージョンの場合は https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/ で
> 適切な wheel を探してください。

## 注意事項

- SMIRK リポジトリ自体を `sys.path` に追加する必要があります (`smirk_dir` パラメータで指定)
- SMIRK は FLAME ベースのパラメータ (shape 300D, exp 50D, pose 6D) を出力します
- 非対称表情の抽出に強みがあります (2024 年発表)
- FLAME と EMOCA の両方にアカウント登録 (MPI) が必要です

## 関連ガイド

- [環境構築ガイド](../../docs/guide_setup.md#23-smirk) — セットアップの詳細
- [Guide A: リアルタイム特徴抽出](../../docs/guide_realtime.md) — FLAME ルートの代替 Extractor
- [Guide C: ルート切り替え](../../docs/guide_route_switching.md) — Extractor の選択指針
