# Deep3DFaceRecon チェックポイントの準備

## 必要ファイル

| ファイル | 説明 |
|----------|------|
| `deep3d_epoch20.pth` | Deep3DFaceRecon 学習済みモデル |
| `BFM/BFMmodelfront.mat` | BFM 基底ベクトル (自動生成) |

## 入手手順

**自動ダウンロードには対応していません。** BFM モデルに学術ライセンスが必要なため、手動セットアップが必要です。

### Step 1: リポジトリのクローン

```bash
git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch.git
cd Deep3DFaceRecon_pytorch
```

### Step 2: nvdiffrast のインストール

```bash
git clone -b 0.3.0 https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
pip install .
cd ..
```

### Step 3: ArcFace モデルのセットアップ

```bash
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch ./models/
```

### Step 4: BFM モデルの準備 (学術ライセンス必要)

```bash
# (a) BFM2009 のダウンロード
# https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads
# → アカウント登録 → 01_MorphableModel.mat をダウンロード → BFM/ に配置

# (b) Expression Basis のダウンロード
pip install gdown
gdown 1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6 -O BFM/Exp_Pca.bin

# (c) BFM_model_front.mat は初回実行時に自動生成される
# util/load_mats.py の transferBFM09() が以下から生成:
#   BFM/01_MorphableModel.mat + BFM/Exp_Pca.bin + リポジトリ同梱の index ファイル
```

BFM/ に既に同梱されているファイル (ダウンロード不要):
- `BFM_front_idx.mat`, `BFM_exp_idx.mat`
- `facemodel_info.mat`, `select_vertex_id.mat`
- `similarity_Lm3D_all.mat`, `std_exp.txt`

### Step 5: 学習済みモデルのダウンロード

```bash
# Google Drive フォルダからダウンロード:
# https://drive.google.com/drive/folders/1liaIxn9smpudjjqMaWWRpP0mXRW_qRPP
# → epoch_20.pth をダウンロード
mkdir -p checkpoints/face_recon_feat0.2_augment
# (ブラウザでダウンロードした epoch_20.pth を上記ディレクトリに配置)
```

### Step 6: 追加チェックポイントのダウンロード

| ファイル | 配置先 | ダウンロード元 |
|----------|--------|----------------|
| `backbone.pth` | `checkpoints/recog_model/ms1mv3_arcface_r50_fp16/` | [OneDrive](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%21558&cid=4A83B6B633B029CC) |
| `resnet50-0676ba61.pth` | `checkpoints/init_model/` | https://download.pytorch.org/models/resnet50-0676ba61.pth |
| `68lm_detector.pb` | `checkpoints/lm_model/` | [Google Drive](https://drive.google.com/file/d/1Jl1yy2v7lIJLTRVIpgg2wvxYITI8Dkmw/view) |

### Step 7: FLARE 用にコピー

```bash
# モデルチェックポイント
cp checkpoints/face_recon_feat0.2_augment/epoch_20.pth \
   ../FLARE_by_Claude/checkpoints/deep3d/deep3d_epoch20.pth

# BFM モデル (初回実行後に生成される BFM_model_front.mat)
mkdir -p ../FLARE_by_Claude/checkpoints/deep3d/BFM
cp BFM/BFM_model_front.mat \
   ../FLARE_by_Claude/checkpoints/deep3d/BFM/BFMmodelfront.mat
```

## 配置後の構成

```
checkpoints/deep3d/
├── deep3d_epoch20.pth
└── BFM/
    └── BFMmodelfront.mat
```

## 注意事項

- **Linux のみ対応**: Deep3DFaceRecon は Ubuntu でのみ動作確認されています
- **Python 3.6 + PyTorch 1.6**: リポジトリ自体の依存は古いバージョンです。FLARE から呼ぶ場合は `deep3d_dir` パラメータでリポジトリパスを指定し、`sys.path` 経由でインポートされます
- **nvdiffrast**: CUDA Toolkit と PyTorch の CUDA バージョンが一致している必要があります

## 関連ガイド

- [環境構築ガイド](../../docs/guide_setup.md#22-deep3dfacerecon) — セットアップの詳細
- [Guide A: リアルタイム特徴抽出](../../docs/guide_realtime.md) — BFM ルートの Extractor として使用
- [Guide B: LHG 前処理](../../docs/guide_lhg_preprocess.md) — バッチ特徴抽出
- [Guide C: ルート切り替え](../../docs/guide_route_switching.md) — BFM/FLAME の選択
