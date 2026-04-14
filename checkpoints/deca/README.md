# DECA チェックポイントの準備

## 必要ファイル

| ファイル | 説明 |
|----------|------|
| `deca_model.tar` | DECA 学習済みモデル (ResNet-50 ベース) |

## 入手方法

### 方法 1: FLARE 自動ダウンロードスクリプト (推奨)

```bash
pip install gdown
python scripts/download_checkpoints.py --model deca
```

### 方法 2: gdown で直接ダウンロード

```bash
pip install gdown
gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O checkpoints/deca/deca_model.tar
```

### 方法 3: DECA リポジトリの fetch_data.sh を使用

```bash
git clone https://github.com/yfeng95/DECA.git
cd DECA
bash fetch_data.sh
# → data/deca_model.tar がダウンロードされる
cp data/deca_model.tar ../FLARE_by_Claude/checkpoints/deca/
```

### 方法 4: ブラウザから手動ダウンロード

Google Drive のダウンロードリンク:
https://drive.google.com/uc?export=download&id=1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje

> gdown / wget でクォータ制限に引っかかる場合はブラウザからダウンロードしてください。

## 配置後の構成

```
checkpoints/deca/
└── deca_model.tar
```

## 補足: DECA リポジトリ自体のセットアップ

FLARE の DECA Extractor はチェックポイントファイルのみを使用するため、
DECA リポジトリ自体のクローン・環境構築は不要です。

DECA リポジトリを別途使いたい場合の参考情報:

- リポジトリ: https://github.com/yfeng95/DECA
- Python 3.7, PyTorch 1.6.0 (古いバージョン)
- `generic_model.pkl` (FLAME モデル) が `data/` に同梱される場合あり

## 補足: FLAME モデル (generic_model.pkl)

`deca_model.tar` には FLAME の基底データが内包されているため、
FLARE の Extractor として使う場合は `generic_model.pkl` は不要です。

メッシュ可視化 (`demo_visualize.py --mode mesh`) で別途必要な場合は
[`checkpoints/flame/README.md`](../flame/README.md) を参照してください。

## 関連ガイド

- [環境構築ガイド](../../docs/guide_setup.md#21-deca) — DECA セットアップの詳細
- [Guide A: リアルタイム特徴抽出](../../docs/guide_realtime.md) — FLAME ルートの Extractor として使用
- [Guide B: LHG 前処理](../../docs/guide_lhg_preprocess.md) — バッチ特徴抽出
- [Guide D: FLAME デコーダ学習](../../docs/guide_flame_decoder.md) — 学習データ準備時に使用
