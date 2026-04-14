# Learning2Listen (L2L) モデルの準備

## 必要ファイル

| ファイル | 説明 |
|----------|------|
| `l2l_vqvae.pth` | Learning2Listen VQ-VAE 学習済みモデル |

## 入手手順

### Step 1: リポジトリのクローン

```bash
git clone https://github.com/evonneng/learning2listen.git
cd learning2listen/src
export L2L_PATH=$(pwd)
```

### Step 2: 学習済みモデルのダウンロード

Google Drive から話者ごとのモデルをダウンロード:

| 話者 | Google Drive ID | gdown コマンド |
|------|----------------|----------------|
| Conan | `1HlGLMPcshqwdmQvryKPVsYvd9oL2yGML` | `gdown 1HlGLMPcshqwdmQvryKPVsYvd9oL2yGML -O models/conan.tar` |
| Fallon | `1_d4D6T9qflgd15uA3xhtp9wvchWbg9Da` | `gdown 1_d4D6T9qflgd15uA3xhtp9wvchWbg9Da -O models/fallon.tar` |
| Stephen | `1gXt2pjpnPItCIfINKCToBacoTI-cVs0W` | `gdown 1gXt2pjpnPItCIfINKCToBacoTI-cVs0W -O models/stephen.tar` |
| Trevor | `1M5y5J3NKhMbzIaU58_Gz8yOFQZuOVhCn` | `gdown 1M5y5J3NKhMbzIaU58_Gz8yOFQZuOVhCn -O models/trevor.tar` |

```bash
pip install gdown

# 例: Conan モデルのダウンロード
gdown 1HlGLMPcshqwdmQvryKPVsYvd9oL2yGML -O models/conan.tar

# モデルの展開
bash scripts/unpack_models.sh

# FLARE 用にコピー (使用する話者のチェックポイントを選択)
cp models/<展開後のチェックポイントファイル> \
   ../FLARE_by_Claude/checkpoints/l2l/l2l_vqvae.pth
```

### Step 3: 学習データのダウンロード (自分で学習する場合のみ)

```bash
# Berkeley Vision がホストするデータセット
wget http://learning2listen.berkeleyvision.org/conan_data.tar -P data/
wget http://learning2listen.berkeleyvision.org/fallon_data.tar -P data/
wget http://learning2listen.berkeleyvision.org/stephen_data.tar -P data/
wget http://learning2listen.berkeleyvision.org/trevor_data.tar -P data/
wget http://learning2listen.berkeleyvision.org/devi2_data.tar -P data/
wget http://learning2listen.berkeleyvision.org/kimmel_data.tar -P data/

bash scripts/unpack_data.sh
```

データ形式: 各話者ごとに `.npy` ファイル:
- `*_list_faces_clean_deca.npy` — リスナー頭部動作 (N x 64 x 184)
- `*_speak_audio_clean_deca.npy` — 話者音声特徴 (N x 256 x 128)
- `*_speak_faces_clean_deca.npy` — 話者頭部動作 (N x 64 x 184)
- `*_speak_files_clean_deca.npy` — 話者ファイル情報 (N x 64 x 3)

## 配置後の構成

```
checkpoints/l2l/
└── l2l_vqvae.pth
```

## 環境情報 (参考: L2L リポジトリの要件)

| 項目 | 要件 |
|------|------|
| Python | 3.6.11 |
| PyTorch | (CUDA 9.0 対応版) |
| CUDA | 9.0, cuDNN 7.0 |

> **FLARE との互換性**: L2L リポジトリは古い環境 (Python 3.6 / CUDA 9.0) を要求しますが、
> FLARE から呼ぶ場合はチェックポイントのみを使用するため、FLARE 側の環境 (Python 3.11) で
> 動作します。L2L リポジトリの完全セットアップは自分で学習する場合のみ必要です。

## L2L リポジトリ内の依存パッケージ (参考)

```
einops, matplotlib, numpy, opencv_contrib_python, scipy, seaborn, six, torch, torchvision
```

> **既知の問題**: L2L の README には、`torch/nn/modules/conv.py` の `self.padding_mode != 'zeros'`
> アサーションをコメントアウトする必要があると記載されています (ConvTranspose1d の replicated padding 用)。
> FLARE 環境での使用時には影響しません。

## 用途

Learning2Listen は LHG (Listening Head Generation) の推論モデルです。
話者の音声 + 3DMM パラメータから、聞き手の頭部動作パラメータを予測します。

```
[話者音声] + [話者 3DMM] → L2L VQ-VAE → [聞き手 3DMM 予測]
```

- リアルタイムパイプラインの推論スレッドで使用されます
- window_size=64 フレームの時系列入力を受け取ります

## 関連ガイド

- [環境構築ガイド](../../docs/guide_setup.md#27-learning2listen) — セットアップの詳細
- [Guide A: リアルタイム特徴抽出](../../docs/guide_realtime.md) — LHG モデルとの連携
