# FLARE デモ操作ガイド

## やりたいことから探す (逆引き)

| やりたいこと | 使うデモ |
|---|---|
| 大量の対話動画から特徴量を一括抽出したい | Demo a (バッチ特徴抽出) |
| 特定の動画を FlashAvatar でレンダリングして確認したい | Demo b (動画レンダリング) |
| Webカメラで SMIRK + FlashAvatar をリアルタイム確認したい | Demo c (Webカメラ / SMIRK) |
| Webカメラで DECA + FlashAvatar をリアルタイム確認したい | Demo c-DECA (Webカメラ / DECA) |
| FlashAvatar を新しい人物向けに学習したい | FlashAvatar 学習手順 |

---

## ディレクトリ構成

```
demos/
  _env.sh                  # 全デモ共通: EGL vendor pinning + PYTHONPATH 設定
  run_demo_batch.sh        # Demo a ランチャ (SMIRK / DECA 両対応)
  run_demo_video.sh        # Demo b ランチャ
  run_demo_webcam.sh       # Demo c ランチャ (SMIRK)
  run_demo_webcam_deca.sh  # Demo c-DECA ランチャ (DECA)
  demo_batch_extract.py    # Demo a 本体
  demo_video_render.py     # Demo b 本体
  demo_webcam.py           # Demo c 本体 (SMIRK)
  demo_webcam_deca.py      # Demo c-DECA 本体 (DECA)
  DEMOS.md                 # 本ファイル
```

---

## 事前準備

### 必要なチェックポイント

各デモを実行する前に以下のファイルを配置してください。

**SMIRK (Demo a/c-SMIRK):**
```
checkpoints/smirk/SMIRK_em1.pt
```
セットアップ: `bash install/setup_smirk.sh`

SMIRK_em1.pt の手動取得:
Google Drive ID `1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE`
または `gdown 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE -O checkpoints/smirk/SMIRK_em1.pt`

**DECA (Demo a/b/c-DECA):**
```
checkpoints/deca/deca_model.tar
```
セットアップ: `bash install/setup_deca.sh`

deca_model.tar の手動取得:
Google Drive ID `1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje`
または `gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O checkpoints/deca/deca_model.tar`

**FlashAvatar (Demo b/c):**
```
checkpoints/flashavatar/<person_id>/point_cloud/iteration_30000/point_cloud.ply
```
FlashAvatar は対象人物ごとに個別学習が必要です。
学習手順は後述の「FlashAvatar 学習手順」を参照してください。

### 環境変数 (_env.sh)

各 `run_*.sh` は実行前に `demos/_env.sh` を source します。
このスクリプトが行うこと:

- **EGL vendor pinning**: MediaPipe の GPU delegate が MESA 代わりに NVIDIA EGL
  を使うよう `__EGL_VENDOR_LIBRARY_FILENAMES` を設定します。
  NVIDIA EGL JSON ファイル (通常 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`)
  が見つからない場合は警告を表示し、MediaPipe は CPU で動作します。
- **PYTHONPATH 設定**: FLARE ルートを `PYTHONPATH` に追加します。
- **PYTHONDONTWRITEBYTECODE=1**: `.pyc` ファイルの生成を抑制します。

WSL2 環境では NVIDIA EGL が利用できないため、GPU delegate は機能しません。
各デモスクリプトが起動時に自動検出して CPU にフォールバックします。

### FlashAvatar 学習手順

FlashAvatar は対象人物ごとに 3DGS (3D Gaussian Splatting) を個別学習する必要があります。
汎用の事前学習済みモデルはありません。

**必要なもの:**
- 対象人物の正面向きモノクラー動画 (最低 600 フレーム推奨、target_fps=25 なら 24 秒以上)
- セットアップ済みの環境 (`bash install/setup_flashavatar.sh` 実行済み)
- DECA または SMIRK のチェックポイント

**学習コマンド:**
```bash
python scripts/train_flashavatar.py \
    --id_name person01 \
    --video data/raw/person01.mp4 \
    --config configs/train_flashavatar.yaml
```

**設定ファイル `configs/train_flashavatar.yaml` の主要パラメータ:**

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `pipeline.extractor` | `deca` | 特徴抽出器 (`deca` または `smirk`) |
| `video.target_fps` | `25` | 学習用 FPS に正規化 |
| `video.img_size` | `512` | フレーム解像度 (px) |
| `flashavatar.iterations` | `30000` | 学習イテレーション数 |

**想定学習時間 (RTX 3090):**
- `iterations=30000` + `img_size=512`: 約 30 分
- `iterations=60000` + `img_size=512`: 約 60 分 (高品質)
- VRAM 不足 (< 12 GB): `img_size=256` + `iterations=15000` を試す

**学習済みモデルの配置:**
学習完了後に `checkpoints/flashavatar/person01/` を指定してデモを実行します。
スクリプトが自動的に正しいパスに配置します。

---

## Demo a: バッチ特徴抽出

`data/multimodal_dialogue_formed/` 以下の動画ファイルを一括処理し、
FLAME パラメータを `data/movements/` に `.npz` 形式で保存します。

### 基本実行 (SMIRK)

```bash
bash demos/run_demo_batch.sh \
    --input_dir data/multimodal_dialogue_formed \
    --output_dir data/movements \
    --extractor smirk
```

### DECA ルート

```bash
bash demos/run_demo_batch.sh \
    --input_dir data/multimodal_dialogue_formed \
    --output_dir data/movements \
    --extractor deca
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--extractor` | `smirk` | `smirk` または `deca` |
| `--mp_delegate` | `cpu` | MediaPipe 顔検出デバイス (`cpu` または `gpu`) |
| `--overwrite` | (省略時スキップ) | 既存 .npz を上書きする |
| `--log_path` | `output_dir/batch_extract.log` | ログファイルパス |
| `--max_frames` | (全フレーム) | 1 動画あたりの最大フレーム数 (デバッグ用) |

### 出力ファイル

```
data/movements/data001/comp/smirk_host_0_1000.npz
data/movements/data001/host/smirk_comp_0_1000.npz
```

認識失敗フレームは直前フレームの値で補完し、
`batch_extract.fail.jsonl` にフレームインデックスと失敗理由を記録します。

---

## Demo b: 動画ファイルでのレンダリングテスト

指定の動画から特徴抽出し、学習済み FlashAvatar でレンダリングして
サイド・バイ・サイド動画を生成します。

### SMIRK ルート

```bash
bash demos/run_demo_video.sh \
    --input_video data/raw/sample.mp4 \
    --output_video output/rendered.mp4 \
    --checkpoint_dir checkpoints/flashavatar/person01
```

### DECA ルート

```bash
bash demos/run_demo_video.sh \
    --input_video data/raw/sample.mp4 \
    --extractor deca \
    --checkpoint_dir checkpoints/flashavatar/person01
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--input_video` | (必須) | 入力動画パス |
| `--output_video` | (入力と同ディレクトリ) | 出力動画パス |
| `--checkpoint_dir` | `./checkpoints/flashavatar/` | FlashAvatar チェックポイントディレクトリ |
| `--extractor` | `smirk` | `smirk` または `deca` |
| `--max_frames` | (全フレーム) | 最大処理フレーム数 |
| `--display_width` | `512` | 1 ペインの幅 (px) |
| `--no_eye_supplement` | (省略時は補完有効) | DECA 使用時に eyes_pose 補完を無効化 |

出力動画は「入力フレーム (左) | FlashAvatar レンダリング (右)」の横並びです。

---

## Demo c: Webカメラリアルタイム (SMIRK)

Webカメラまたは動画ファイルから SMIRK で特徴抽出し、
FlashAvatar でリアルタイムレンダリングして 3 ペイン表示します。

```
左ペイン: 元カメラ映像
中ペイン: 顔検出 bbox overlay
右ペイン: FlashAvatar レンダリング結果
```

### 基本実行

```bash
bash demos/run_demo_webcam.sh \
    --checkpoint_dir checkpoints/flashavatar/person01
```

### GPU MediaPipe を使用する場合

```bash
bash demos/run_demo_webcam.sh \
    --mp_delegate gpu \
    --checkpoint_dir checkpoints/flashavatar/person01
```

### 動画ファイルを入力とする場合 (ideal-source モード)

```bash
bash demos/run_demo_webcam.sh \
    --source data/raw/sample.mp4 \
    --checkpoint_dir checkpoints/flashavatar/person01
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--source` | `0` | カメラインデックス (整数) または動画ファイルパス |
| `--checkpoint_dir` | `./checkpoints/flashavatar/` | FlashAvatar チェックポイントディレクトリ |
| `--smirk_model` | `./checkpoints/smirk/SMIRK_em1.pt` | SMIRK チェックポイントパス |
| `--mp_delegate` | `cpu` | MediaPipe 推論デバイス |
| `--width` / `--height` | `1280` / `720` | Webカメラ解像度 |
| `--fourcc` | `MJPG` | Webカメラ FOURCC |
| `--no_render` | (省略時レンダリング有効) | FlashAvatar をスキップ (SMIRK のみ測定) |

キーボード操作:
- `q`: 終了
- `s`: スナップショット保存 (`output/` ディレクトリ)

終了時に per-stage の平均処理時間サマリを表示します
(`cap / mp / smirk / cvt / render / disp`)。

---

## Demo c-DECA: Webカメラリアルタイム (DECA)

Demo c の DECA 版。DECA は eyes_pose / eyelids を出力しないため、
MediaPipe Face Landmarker で眼球ポーズを自動補完します
(`--no_eye_supplement` で無効化可能)。

### 基本実行

```bash
bash demos/run_demo_webcam_deca.sh \
    --checkpoint_dir checkpoints/flashavatar/person01
```

### オプション一覧 (SMIRK 版との差分)

| オプション | 説明 |
|---|---|
| `--deca_model` | DECA チェックポイントパス (デフォルト: `./checkpoints/deca/deca_model.tar`) |
| `--no_eye_supplement` | MediaPipe による eyes_pose / eyelids 補完を無効化 |

その他のオプションは Demo c (SMIRK 版) と同じです。

per-stage タイマには `eye` (MediaPipe 補完処理) が追加されます
(`cap / mp / deca / eye / cvt / render / disp`)。

---

## トラブルシューティング

### MediaPipe GPU delegate が機能しない

**症状:** `--mp_delegate gpu` を指定してもCPU で動作している、または
`Unable to initialize EGL` というエラーが出る。

**原因と対処:**

1. NVIDIA EGL JSON ファイルがない:
   ```bash
   ls /usr/share/glvnd/egl_vendor.d/
   # 10_nvidia.json が存在するか確認
   ```
   存在しない場合は `nvidia-egl-external-platform-dev` パッケージをインストールしてください。

2. WSL2 環境:
   WSL2 では NVIDIA EGL は利用できません。`--mp_delegate cpu` を使用してください。
   各デモスクリプトが WSL2 を自動検出して警告を表示します。

3. MESA ドライバが NVIDIA より優先されている:
   `demos/_env.sh` が自動的に NVIDIA EGL JSON を直接指定することで
   この問題を回避しています。`run_*.sh` 経由で実行してください
   (`python demos/demo_webcam.py` を直接実行した場合は設定されません)。

### Webカメラの FPS が出ない

**症状:** カメラは動作しているが、フレームレートが 5-10 FPS 程度に落ちる。

**原因:** USB 2.0 帯域の制約で YUYV フォーマットでは 720p@30fps が出ない。

**対処:**
```bash
# FOURCC が MJPG になっているか確認 (起動ログに表示される)
# [INFO] fourcc: MJPG  size: 1280x720  reported_fps: 30.0

# MJPG が使えない場合は解像度を下げる
bash demos/run_demo_webcam.sh \
    --fourcc MJPG --width 640 --height 480 \
    --checkpoint_dir checkpoints/flashavatar/person01
```

`BUFFERSIZE` や `AUTO_EXPOSURE` の変更は FPS 低下の原因になるため、
本デモでは変更しません。

### FlashAvatar のレンダリングが崩れる / 顔が歪む

**症状:** レンダリング結果が大きく崩れている、顔の形状がおかしい。

**原因と対処:**

1. チェックポイントの person_id が一致していない:
   FlashAvatar は対象人物ごとに学習するため、
   学習した人物と異なる顔を入力すると崩れます。
   `--checkpoint_dir` に指定している人物の動画を入力してください。

2. チェックポイントのパスが間違っている:
   ```bash
   ls checkpoints/flashavatar/<person_id>/point_cloud/iteration_30000/
   # point_cloud.ply が存在するか確認
   ```

3. 学習が不十分:
   `iterations=30000` 未満で学習を停止した場合は崩れることがあります。
   `configs/train_flashavatar.yaml` の `iterations` を確認してください。

### SMIRK モデルのロードに失敗する

**症状:** `Failed to import SMIRK modules` というエラーが出る。

**対処:**
```bash
bash install/setup_smirk.sh
# third_party/smirk サブモジュールの初期化と依存パッケージのインストールを行います
```

### DECA モデルのロードに失敗する

**症状:** `Failed to import DECA modules` というエラーが出る。

**対処:**
```bash
bash install/setup_deca.sh
# third_party/DECA サブモジュールの初期化と依存パッケージのインストールを行います
```

### CUDA が見つからない / CPU で実行される

**症状:** `CUDA が利用できません。CPU に切り替えます。` という警告が出る。

**対処:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# False の場合は CUDA 版 PyTorch をインストールしてください
# install/setup_smirk.sh または install/setup_deca.sh が CUDA 12.8 用を設定します
```
