# FlashAvatar モデルの準備

## 概要

FlashAvatar は **対象人物ごとに個別学習が必要** な 3D Gaussian Splatting ベースのレンダラです。
事前学習済みの汎用モデルは存在しません。

## 必要ファイル (人物ごと)

| ファイル | 説明 |
|----------|------|
| `point_cloud/iteration_30000/point_cloud.ply` | 学習済み 3D Gaussian 点群 |
| `cameras.json` | カメラパラメータ |
| その他学習設定ファイル | FlashAvatar 学習時の出力 |

## 学習手順

### Step 1: リポジトリのクローン

```bash
git clone https://github.com/MingZhongCodes/FlashAvatar.git
cd FlashAvatar
pip install -r requirements.txt
```

### Step 2: diff-gaussian-rasterization のインストール

3D Gaussian Splatting の描画に必要なカスタム CUDA カーネルです:

```bash
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
```

> **注意**: CUDA Toolkit がインストールされている必要があります。
> `nvcc --version` でバージョンを確認してください。

### Step 3: 学習データの準備

FlashAvatar の README に従い、対象人物の動画から学習データを準備します。

### Step 4: 学習の実行

```bash
# RTX 3090 で約 30 分
python train.py --config configs/<config>.yaml
```

### Step 5: FLARE 用に配置

```bash
# 学習済みモデルを FLARE にコピー
cp -r output/person01/ ../FLARE_by_Claude/checkpoints/flashavatar/person01/
```

## 配置後の構成

```bash
# 人物ごとにサブディレクトリ
checkpoints/flashavatar/
├── person01/
│   └── point_cloud/
│       └── iteration_30000/
│           └── point_cloud.ply
├── person02/
│   └── ...
└── ...
```

## 注意事項

- FlashAvatar は FLAME ルート (Route B) のレンダラです
- 対象人物ごとに個別学習が必要 (汎用モデルはありません)
- `diff-gaussian-rasterization` パッケージが必要です (CUDA カーネルのビルドあり)
- PIRender と異なり、ソース画像は不要です
- ~300 FPS @ 512x512 (RTX 3090)

## 関連ガイド

- [環境構築ガイド](../../docs/guide_setup.md#26-flashavatar) — セットアップの詳細
- [Guide A: リアルタイム特徴抽出](../../docs/guide_realtime.md) — FLAME ルートのレンダラとして使用
- [Guide C: ルート切り替え](../../docs/guide_route_switching.md) — FlashAvatar vs PIRender の比較
