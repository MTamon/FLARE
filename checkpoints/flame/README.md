# FLAME モデルの準備

## 必要ファイル

| ファイル | 説明 |
|----------|------|
| `generic_model.pkl` | FLAME パラメトリック顔モデル |

## 入手方法

**学術ライセンスへの同意が必要です。自動ダウンロードには対応していません。**

### Step 1: アカウント登録

1. [FLAME 公式サイト](https://flame.is.tue.mpg.de/) にアクセス
2. 「Register」からアカウント作成
3. 学術ライセンス (Academic License) に同意

### Step 2: モデルのダウンロード

1. ログイン後、「Downloads」ページへ
2. 「FLAME 2020」セクションから `FLAME2020.zip` をダウンロード
3. ZIP を展開 → `generic_model.pkl` を取得

```bash
# ZIP 展開後
unzip FLAME2020.zip -d /tmp/FLAME2020
cp /tmp/FLAME2020/generic_model.pkl checkpoints/flame/
```

### 代替入手方法

DECA リポジトリのセットアップ時に `generic_model.pkl` が `data/` に配置されることがあります:

```bash
# DECA をセットアップ済みの場合
cp DECA/data/generic_model.pkl checkpoints/flame/
```

`fetch_data.sh` 実行時に FLAME 公式サイトの認証情報を求められるため、
DECA 経由でも FLAME ライセンスへの同意は必要です。

## 配置後の構成

```
checkpoints/flame/
└── generic_model.pkl
```

## 用途

- `scripts/demo_visualize.py --mode mesh` でのメッシュ可視化 (ワイヤフレーム描画)
- FLAME パラメータのサニティチェック
- 学習なしでパラメータの定性的な確認が可能

> **DECA Extractor の使用には不要**: `deca_model.tar` には FLAME の基底データが
> 内包されているため、DECA を Extractor として使う分には `generic_model.pkl` は不要です。
> メッシュ可視化 (`FLAMEMeshRenderer`) を使う場合にのみ必要です。

## 関連ガイド

- [環境構築ガイド](../../docs/guide_setup.md#21-deca) — FLAME モデルの取得は DECA セットアップに含まれる
- [Guide D: FLAME デコーダ学習と可視化](../../docs/guide_flame_decoder.md) — メッシュ可視化モード
