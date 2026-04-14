# 環境構築ガイド

FLARE とその依存外部リポジトリのセットアップ手順。

## 1. FLARE 本体のインストール

### 前提条件

| 項目 | 要件 |
|------|------|
| OS | Windows 10/11, Ubuntu 20.04+ |
| Python | 3.11 推奨 (3.9 以上) |
| GPU | CUDA 対応 NVIDIA GPU (RTX 2080 Ti 以上推奨) |
| CUDA | 12.x 推奨 (11.7 以上) |
| VRAM | 8 GB 以上 (複数モデル同時使用時は 12 GB 以上) |

### インストール手順

```bash
# 1. リポジトリのクローン
git clone <FLARE_REPO_URL>
cd FLARE_by_Claude

# 2. Python 仮想環境の作成 (conda の場合)
conda create -n flare python=3.11 -y
conda activate flare

# 2'. Python 仮想環境の作成 (venv の場合)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 3. PyTorch のインストール (CUDA バージョンに合わせる)
# CUDA 12.x の場合:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# CUDA 11.8 の場合:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. FLARE の依存パッケージ
pip install numpy pydantic pyyaml click loguru opencv-python mediapipe scipy
```

### 動作確認

```bash
# CLI の確認
python tool.py --help

# テストの実行
pip install pytest
pytest tests/ -x -q
```

---

## 2. 外部リポジトリのセットアップ

FLARE は Extractor / Renderer の実装として外部リポジトリを利用します。
**使用するルート (BFM / FLAME) に応じて、必要なものだけセットアップしてください。**

### 一覧

| リポジトリ | FLARE での用途 | ルート | 必須度 |
|---|---|---|---|
| [DECA](#21-deca) | FLAME Extractor (推奨) | FLAME | ルート使用時は必須 |
| [Deep3DFaceRecon](#22-deep3dfacerecon) | BFM Extractor (推奨) | BFM | ルート使用時は必須 |
| [SMIRK](#23-smirk) | FLAME Extractor (高精度代替) | FLAME | 任意 |
| [3DDFA_V2](#24-3ddfa_v2) | BFM Extractor (軽量代替) | BFM | 任意 |
| [PIRender](#25-pirender) | BFM Renderer | BFM | 可視化時に必要 |
| [FlashAvatar](#26-flashavatar) | FLAME Renderer | FLAME | 可視化時に必要 |
| [Learning2Listen](#27-learning2listen) | LHG 推論モデル | 共通 | リアルタイムパイプラインで必要 |

---

### 2.1 DECA

**用途**: FLAME ルートの標準 Extractor。画像 → FLAME パラメータ (shape 100D, exp 50D, pose 6D)。

```bash
# クローン
git clone https://github.com/yfeng95/DECA.git
cd DECA

# 環境構築 (DECA は Python 3.7 + PyTorch 1.6 だが、FLARE から呼ぶ場合は
# FLARE 側の環境で動作させるため、DECA リポジトリのインストールは不要。
# FLARE は DECA のコードを直接インポートせず、チェックポイントのみ使用する。)

# チェックポイントのダウンロード
pip install gdown
gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O ../FLARE_by_Claude/checkpoints/deca/deca_model.tar

# または FLARE の自動ダウンロードスクリプトを使用
cd ../FLARE_by_Claude
python scripts/download_checkpoints.py --model deca
```

**FLAME モデル (学術ライセンス必要)**:

DECA チェックポイント (`deca_model.tar`) には FLAME の基底データが内包されているため、
FLARE の DECA Extractor を使う場合は別途 `generic_model.pkl` は不要です。
ただし、メッシュ可視化 (`demo_visualize.py --mode mesh`) を使う場合は別途必要です:

1. https://flame.is.tue.mpg.de/ でアカウント作成・ライセンス同意
2. 「FLAME 2020」をダウンロード → `FLAME2020.zip` を展開
3. `generic_model.pkl` を `checkpoints/flame/` に配置

**FLARE 設定での参照**:

```yaml
# configs/realtime_flame.yaml or configs/lhg_extract_deca.yaml
extractor:
  type: deca
  model_path: ./checkpoints/deca/deca_model.tar
```

---

### 2.2 Deep3DFaceRecon

**用途**: BFM ルートの標準 Extractor。画像 → BFM パラメータ (id 80D, exp 64D, pose 6D)。

```bash
# クローン
git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch.git
cd Deep3DFaceRecon_pytorch

# nvdiffrast のインストール (レンダリングに必要)
git clone -b 0.3.0 https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
pip install .
cd ..

# ArcFace モデルのセットアップ (顔認識特徴量に必要)
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch ./models/
```

**BFM モデル (学術ライセンス必要)**:

```bash
# Step 1: BFM2009 のダウンロード
# https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads で登録
# → 01_MorphableModel.mat をダウンロード → BFM/ に配置

# Step 2: Expression Basis のダウンロード (Google Drive)
# https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view
gdown 1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6 -O BFM/Exp_Pca.bin

# Step 3: BFM_model_front.mat は初回実行時に自動生成される
# (util/load_mats.py の transferBFM09() が 01_MorphableModel.mat + Exp_Pca.bin から生成)
```

**チェックポイントのダウンロード**:

```bash
# 学習済みモデル (Google Drive フォルダ)
# https://drive.google.com/drive/folders/1liaIxn9smpudjjqMaWWRpP0mXRW_qRPP
# → epoch_20.pth をダウンロード
mkdir -p checkpoints/face_recon_feat0.2_augment
# ダウンロードした epoch_20.pth を配置

# FLARE 用にコピー
cp checkpoints/face_recon_feat0.2_augment/epoch_20.pth \
   ../FLARE_by_Claude/checkpoints/deep3d/deep3d_epoch20.pth
cp BFM/BFM_model_front.mat \
   ../FLARE_by_Claude/checkpoints/deep3d/BFM/BFMmodelfront.mat
```

**追加チェックポイント (Deep3DFaceRecon 内部で使用)**:

| ファイル | 配置先 | ダウンロード元 |
|----------|--------|----------------|
| `backbone.pth` | `checkpoints/recog_model/ms1mv3_arcface_r50_fp16/` | [OneDrive](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%21558&cid=4A83B6B633B029CC) |
| `resnet50-0676ba61.pth` | `checkpoints/init_model/` | `https://download.pytorch.org/models/resnet50-0676ba61.pth` |
| `68lm_detector.pb` | `checkpoints/lm_model/` | [Google Drive](https://drive.google.com/file/d/1Jl1yy2v7lIJLTRVIpgg2wvxYITI8Dkmw/view) |

**FLARE 設定での参照**:

```yaml
# configs/realtime_bfm.yaml or configs/lhg_extract_bfm.yaml
extractor:
  type: deep3d
  model_path: ./checkpoints/deep3d/deep3d_epoch20.pth
  # deep3d_dir: リポジトリのクローン先パスを指定
```

> **注意**: Deep3DFaceRecon は Linux のみ動作確認されています。Windows では nvdiffrast のビルドに問題が生じる場合があります。

---

### 2.3 SMIRK

**用途**: FLAME ルートの高精度 Extractor。非対称表情の抽出に強み。

```bash
# クローン
git clone https://github.com/georgeretsi/smirk.git
cd smirk

# 環境構築
# (FLARE 環境とは別に作成することを推奨 — pytorch3d の依存が複雑)
conda create -n smirk python=3.9 -y
conda activate smirk
pip install -r requirements.txt

# pytorch3d のインストール (CUDA 11.7 + PyTorch 2.0.1 の場合)
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html

# 全モデルの一括ダウンロード (推奨)
bash quick_install.sh
# → FLAME ライセンスの username/password を聞かれる
```

**手動ダウンロードの場合**:

| ファイル | 配置先 | ダウンロード元 |
|----------|--------|----------------|
| `SMIRK_em1.pt` | `pretrained_models/` | [Google Drive](https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE/view) (`gdown 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE`) |
| `FLAME2020/` | `assets/` | https://flame.is.tue.mpg.de/ (要登録) |
| `face_landmarker.task` | `assets/` | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task |
| `ResNet50/` (EMOCA) | `assets/` | https://emoca.is.tue.mpg.de/ (要登録) |
| `mica.tar` | `assets/` | https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c0f96de/?dl=1 |

```bash
# FLARE 用にチェックポイントをコピー
cp pretrained_models/SMIRK_em1.pt \
   ../FLARE_by_Claude/checkpoints/smirk/smirk_encoder.pt
```

**FLARE 設定での参照**:

```yaml
extractor:
  type: smirk
  model_path: ./checkpoints/smirk/smirk_encoder.pt
  # smirk_dir: リポジトリのクローン先パスを指定
```

---

### 2.4 3DDFA_V2

**用途**: BFM ルートの軽量 Extractor。ONNX ベースで CPU でも高速 (1.35ms/image)。

```bash
# クローン
git clone https://github.com/cleardusk/3DDFA_V2.git
cd 3DDFA_V2

# 依存パッケージ
pip install -r requirements.txt

# Cython/C モジュールのビルド (Linux/Mac)
sh ./build.sh
# → FaceBoxes (NMS), Sim3DR (3D rendering), render.so がコンパイルされる
```

> **Windows の場合**: `build.sh` は Linux 向けです。Windows では Cython ビルドに
> Visual Studio Build Tools が必要です。ONNX 推論のみなら NMS/Sim3DR のビルドは
> 省略可能ですが、一部機能が制限されます。

**ONNX モデルのダウンロード**:

リポジトリに含まれる `.pth` ファイルは ONNX 形式ではありません。ONNX モデルは別途ダウンロードが必要です:

```bash
# mb1_120x120.onnx (MobileNetV1, 標準モデル)
gdown 1YpO1KfXvJHRmCBkErNa62dHm-CUjsoIk -O weights/mb1_120x120.onnx

# FLARE 用にコピー
cp weights/mb1_120x120.onnx ../FLARE_by_Claude/checkpoints/3ddfa/mb1_120x120.onnx
cp configs/bfm_noneck_v3.pkl ../FLARE_by_Claude/checkpoints/3ddfa/bfm_noneck_v3.pkl

# または FLARE の自動ダウンロードスクリプトを使用
cd ../FLARE_by_Claude
python scripts/download_checkpoints.py --model 3ddfa
```

**FLARE 設定での参照**:

```yaml
extractor:
  type: 3ddfa
  model_path: ./checkpoints/3ddfa/mb1_120x120.onnx
  # tddfa_dir: リポジトリのクローン先パスを指定
```

---

### 2.5 PIRender

**用途**: BFM ルートのレンダラ。NeRF ベースのフローフィールド推定で顔画像を生成。

```bash
# クローン
git clone https://github.com/RenYurui/PIRender.git
cd PIRender

# 依存パッケージ
pip install -r requirements.txt
```

**チェックポイントのダウンロード**:

```bash
# Google Drive からダウンロード
# https://drive.google.com/file/d/1-0xOf6g58OmtKtEWJlU3VlnfRqPN9Uq7/view
gdown 1-0xOf6g58OmtKtEWJlU3VlnfRqPN9Uq7 -O pirender_pretrained.zip
unzip pirender_pretrained.zip

# チェックポイントを FLARE にコピー
cp result/face/epoch_00190_iteration_000400000_checkpoint.pt \
   ../FLARE_by_Claude/checkpoints/pirender/

# Baidu Netdisk の場合 (中国からのアクセス):
# https://pan.baidu.com/s/18B3xfKMXnm4tOqlFSB8ntg (抽出コード: 4sy1)
```

**ソース画像の準備**:

PIRender はレンダリング時に対象人物の正面顔写真が必要です。
詳細は [`data/source_images/README.md`](../data/source_images/README.md) を参照してください。

**FLARE 設定での参照**:

```yaml
renderer:
  type: pirender
  model_path: ./checkpoints/pirender/epoch_00190_iteration_000400000_checkpoint.pt
  source_image: ./data/source_images/source_portrait.png
  # pirender_dir: リポジトリのクローン先パスを指定 (デフォルト: ./PIRender)
```

---

### 2.6 FlashAvatar

**用途**: FLAME ルートのレンダラ。3D Gaussian Splatting ベースで高品質 (300 FPS)。

```bash
# クローン
git clone https://github.com/MingZhongCodes/FlashAvatar.git
cd FlashAvatar

# 依存パッケージ
pip install -r requirements.txt

# diff-gaussian-rasterization のインストール (3DGS 描画に必要)
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
```

**FlashAvatar は対象人物ごとに個別学習が必要**です。汎用の事前学習済みモデルはありません。
FlashAvatar リポジトリの README に従って、対象人物の動画から学習を実行してください。

学習完了後:

```bash
# 学習済みモデルを FLARE にコピー
cp -r output/person01/ ../FLARE_by_Claude/checkpoints/flashavatar/person01/
# → point_cloud/iteration_30000/point_cloud.ply が含まれること
```

**FLARE 設定での参照**:

```yaml
renderer:
  type: flash_avatar
  model_path: ./checkpoints/flashavatar/person01/
  source_image: null  # FlashAvatar はソース画像不要
```

---

### 2.7 Learning2Listen

**用途**: LHG 推論モデル。話者の音声 + 動作から聞き手の頭部動作を予測。

```bash
# クローン
git clone https://github.com/evonneng/learning2listen.git
cd learning2listen/src
```

**環境構築**:

```bash
# L2L は Python 3.6 / CUDA 9.0 の古い環境を要求する。
# FLARE から呼ぶ場合は FLARE 環境で動作させるため、
# L2L リポジトリのインストール自体は不要。チェックポイントのみ使用。

# 環境変数の設定
export L2L_PATH=$(pwd)
```

**学習済みモデルのダウンロード**:

Google Drive から話者ごとのモデルをダウンロード:

| 話者 | Google Drive ID | URL |
|------|----------------|-----|
| Conan | `1HlGLMPcshqwdmQvryKPVsYvd9oL2yGML` | [ダウンロード](https://drive.google.com/file/d/1HlGLMPcshqwdmQvryKPVsYvd9oL2yGML/view) |
| Fallon | `1_d4D6T9qflgd15uA3xhtp9wvchWbg9Da` | [ダウンロード](https://drive.google.com/file/d/1_d4D6T9qflgd15uA3xhtp9wvchWbg9Da/view) |
| Stephen | `1gXt2pjpnPItCIfINKCToBacoTI-cVs0W` | [ダウンロード](https://drive.google.com/file/d/1gXt2pjpnPItCIfINKCToBacoTI-cVs0W/view) |
| Trevor | `1M5y5J3NKhMbzIaU58_Gz8yOFQZuOVhCn` | [ダウンロード](https://drive.google.com/file/d/1M5y5J3NKhMbzIaU58_Gz8yOFQZuOVhCn/view) |

```bash
# 例: Conan モデルのダウンロード
gdown 1HlGLMPcshqwdmQvryKPVsYvd9oL2yGML -O models/conan.tar
bash scripts/unpack_models.sh

# FLARE 用にコピー (使用する話者のモデルを選択)
cp models/<展開後のチェックポイント> \
   ../FLARE_by_Claude/checkpoints/l2l/l2l_vqvae.pth
```

**学習データのダウンロード (自分で学習する場合)**:

```bash
# Berkeley Vision からダウンロード
wget http://learning2listen.berkeleyvision.org/conan_data.tar -P data/
wget http://learning2listen.berkeleyvision.org/fallon_data.tar -P data/
wget http://learning2listen.berkeleyvision.org/stephen_data.tar -P data/
wget http://learning2listen.berkeleyvision.org/trevor_data.tar -P data/
bash scripts/unpack_data.sh
```

データ形式: 各話者ごとに `*_list_faces_clean_deca.npy` (N x 64 x 184) 等の `.npy` ファイル。

**FLARE 設定での参照**:

```yaml
lhg_model:
  type: learning2listen
  model_path: ./checkpoints/l2l/l2l_vqvae.pth
  window_size: 64
  codebook_size: 256
  # l2l_dir: リポジトリのクローン先パスを指定
```

---

## 3. 最小構成の例

### FLAME ルートだけ使う場合

```bash
# 必要なもの:
# 1. FLARE 本体
# 2. DECA チェックポイント (deca_model.tar)
# 3. (可視化する場合) FLAME モデル (generic_model.pkl)

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pydantic pyyaml click loguru opencv-python mediapipe scipy gdown

# DECA モデルダウンロード
python scripts/download_checkpoints.py --model deca

# バッチ特徴抽出
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --config configs/lhg_extract_deca.yaml
```

### BFM ルートだけ使う場合

```bash
# 必要なもの:
# 1. FLARE 本体
# 2. Deep3DFaceRecon チェックポイント + BFM モデル
# 3. (可視化する場合) PIRender + ソース画像

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pydantic pyyaml click loguru opencv-python mediapipe scipy

# Deep3D は自動ダウンロード非対応 → checkpoints/deep3d/README.md の手順に従う

# バッチ特徴抽出
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --config configs/lhg_extract_bfm.yaml
```

---

## 4. トラブルシューティング

| 症状 | 原因と対策 |
|------|-----------|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch 未インストール。CUDA バージョンに合わせてインストール |
| `CUDA out of memory` | `device_map` で GPU 分散、または `batch_size` を減らす |
| `nvdiffrast` ビルドエラー | CUDA Toolkit のバージョンと PyTorch の CUDA バージョンを一致させる |
| `pytorch3d` インストールエラー | [公式ガイド](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) 参照。事前ビルド wheel を使う |
| Google Drive ダウンロードがクォータ制限 | ブラウザからダウンロードするか、翌日再試行 |
| `build.sh` が Windows で動作しない | WSL2 または Git Bash で実行。Cython ビルドには Visual Studio Build Tools 必要 |
| FLAME ダウンロードでエラー | https://flame.is.tue.mpg.de/ でアカウント登録・ライセンス同意が必要 |
