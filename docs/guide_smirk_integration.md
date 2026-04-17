# Guide: SMIRK 統合計画メモ

FLARE の real-time パイプラインに SMIRK（CVPR 2024）を DECA の代替として統合するための設計メモ。**本ファイルは将来実装時の参照資料であり、SMIRK の cuda128 版作成後に再参照することを想定している。**

## 背景

- FLARE は **real-time 処理**（≥25 FPS）が目的
- 現状は DECA で FLAME パラメータを抽出（~119 FPS、精度は中程度）
- SMIRK は DECA より表情精度が高く、かつ `eyelid`（瞼開閉 2D）を出力できる
- SMIRK の cuda128 版は user が今後作成予定

## FlashAvatar のカメラについて

FlashAvatar は `.frame` 形式に **world-space カメラ外部パラメータ (K, R, t)** を期待する
（`flame_converter.py` の `opencv.{K, R, t}` フィールド）。

| パラメータ | 意味 |
|-----------|------|
| **K** (3×3) | カメラ固有行列（焦点距離・主点） |
| **R** (3×3) | 回転行列 = 頭部の**姿勢（Pose）** |
| **t** (3,)  | 並進ベクトル = 頭部の**3D 位置（Position）** = カメラ座標系での (x, y, z) |

現在の DECA ルートは `weak_perspective_to_full(focal_scale=5.0)` で近似しているが、  
`MediaPipePnPTracker` を使うと真の K/R/t を取得できる。

## 最終的に採用する構成

### 構成 A: SMIRK のみ（DECA 代替、カメラは近似）
SMIRK cuda128 完成時点でのミニマム構成。

```text
Webcam/Video Frame (real-time)
  │
  ├── SMIRKExtractor           (flare/extractors/smirk.py)
  │     ├── exp     (50D)        ← FLAME expression
  │     ├── pose    (6D)         ← pose[:, 0:3]=姿勢(回転), pose[:, 3:6]=顎回転
  │     ├── cam     (3D)         ← weak-perspective [scale, tx, ty]
  │     └── eyelid  (2D)         ← 瞼開閉量
  │
  ├── SMIRKToFlameAdapter      (flare/converters/smirk_to_flame.py)
  │     ├── expr      (100D)     ← F.pad(exp, (0, 50))
  │     ├── jaw_pose  (6D)       ← axis_angle → rotation_6d
  │     ├── eyes_pose (12D)      ← identity_6d × 2  (★ 眼球 tracking なしの限界)
  │     └── eyelids   (2D)       ← SMIRK の eyelid をそのまま使用
  │
  └── FlashAvatar Renderer     (120D condition + 近似 K/R/t → 3DGS head)
```

### 構成 B: SMIRK + MediaPipePnPTracker（カメラ精度向上版）
カメラ外部パラメータが重要な場合に追加する。

```text
Webcam/Video Frame (real-time)
  │
  ├── SMIRKExtractor           (flare/extractors/smirk.py)
  │     └── ... (上記と同じ)
  │
  ├── MediaPipePnPTracker      (flare/extractors/mediapipe_pnp.py)  ← 新規実装済み
  │     ├── K   (3×3)           ← カメラ固有行列（校正値 or 自動推定）
  │     ├── R   (3×3)           ← 頭部の回転行列（姿勢）
  │     └── t   (3,)            ← 頭部の 3D 位置  ← ★ ここで位置が取れる
  │
  ├── SMIRKToFlameAdapter      (flare/converters/smirk_to_flame.py)
  │
  └── FlashAvatar Renderer     (120D condition + 実測 K/R/t → 3DGS head)
```

### 頭部の姿勢（向き）と位置の取得

| 情報 | DECA / SMIRK 単体 | + MediaPipePnPTracker |
|------|------------------|-----------------------|
| **姿勢（向き、Pose）** | `pose[:, 0:3]` axis-angle（近似） | `R` (3×3 回転行列、精確） |
| **位置 (Position)** | `t ≈ [tx, ty, 1/scale]`（z は heuristic） | `t` (3D 並進、精確） |
| **カメラ K** | `focal ≈ focal_scale × W × scale`（heuristic） | 実測校正値 or 自動推定 |

SMIRK の `pose[:, 0:3]` が「頭がどちらを向いているか」の axis-angle 表現。  
jaw_pose は `pose[:, 3:6]`。

`MediaPipePnPTracker.track()` の返り値 `t` が**真の 3D 位置**（カメラ座標系での x, y, z）。  
特に `t[2]`（z 方向）がカメラからの距離（奥行き）に対応する。

## 統合時の作業手順

### Step 1: SMIRK cuda128 版を作成（user 側作業）
- third_party/SMIRK を cuda128 環境でビルド可能にする
- `flare/extractors/smirk.py` の `_load_model()` が想定する `src.smirk_encoder.SmirkEncoder` が動くようにする

### Step 2: 検証（既存コード変更なし）
```python
from flare.extractors.smirk import SMIRKExtractor
from flare.converters.smirk_to_flame import SMIRKToFlameAdapter

extractor = SMIRKExtractor(
    model_path="./checkpoints/smirk/smirk_encoder.pt",
    device="cuda:0",
    smirk_dir="./third_party/SMIRK",
)
adapter = SMIRKToFlameAdapter()

params = extractor.extract(image_tensor)  # dict with "exp", "pose", "cam", "eyelid", ...
flash_params = adapter.convert(params)
# flash_params["expr"].shape      == (1, 100)
# flash_params["jaw_pose"].shape  == (1, 6)
# flash_params["eyes_pose"].shape == (1, 12)
# flash_params["eyelids"].shape   == (1, 2)   ← DECA と違い実測値
```

### Step 3: パイプライン統合
1. `scripts/extract_deca_frames.py` を流用し `extract_smirk_frames.py` を作成
   - `DECAExtractor` を `SMIRKExtractor` に差し替え
   - 出力キーの保存形式が変わるため `.pt` フォーマット確認（`shape` / `exp` / `pose` / `cam` / `eyelid`）
2. FlashAvatar 側で読み込む `FlameConverter` / `FrameLoader` が SMIRK 形式を受け付けるよう `.frame` 生成ロジックを調整
3. `lhg_batch.py` の extractor factory に SMIRK ルートを追加

### Step 4（オプション）: real-time カメラ tracker の追加
SMIRK は弱透視カメラしか出さないため、より精度の高い world-space カメラ外部パラメータ（K / R / t）が必要な場合は `MediaPipePnPTracker` を組み合わせる。

**`MediaPipePnPTracker` は実装済み** (`flare/extractors/mediapipe_pnp.py`)。

使用例:
```python
from flare.extractors.mediapipe_pnp import MediaPipePnPTracker
from third_party.FlashAvatar.utils.flame_converter import FlameConverter

tracker = MediaPipePnPTracker()  # カメラ校正なし（自動推定）
# または: tracker = MediaPipePnPTracker.from_calibration("calib.yaml")

converter = FlameConverter(tracker="smirk")

while True:
    frame_bgr = camera.read()

    # SMIRK で FLAME パラメータを抽出
    smirk_params = smirk_extractor.extract(to_tensor(crop(frame_bgr)))

    # MediaPipe で頭部位置・姿勢を取得
    cam = tracker.track(frame_bgr, device="cuda:0")  # full frame!

    if cam is not None:
        # 実測 K/R/t で FlashAvatar frame を生成
        frame_dict = converter.convert(
            smirk_params,
            img_size=(512, 512),
            camera_K=cam["K"].unsqueeze(0),
            camera_R=cam["R"].unsqueeze(0),
            camera_t=cam["t"].unsqueeze(0),
        )
    else:
        # フォールバック: 弱透視近似
        frame_dict = converter.convert(smirk_params, img_size=(512, 512))
```

カメラを校正する場合は OpenCV のチェッカーボード校正を使い `calib.yaml` を作成する。  
校正なしの場合は `焦点距離 = max(W, H) px` で自動推定される（精度は落ちるが動く）。

## 調査で除外した候補

リアルタイム処理に不適合のため、FLARE への統合対象から外した:

| ツール | 除外理由 |
|-------|---------|
| **flame-head-tracker** | Photometric optimization で**数秒/フレーム**、完全オフライン専用。FLARE の real-time 目標に不適合。 |
| **SPECTRE** | 双方向時系列 CNN で**±2 フレーム遅延（~80ms）**、かつカメラ外部パラメータを出さない。FLARE の課題（real-time + camera）どちらも解決しない。 |

両者とも一度はスタブを作成したが、real-time 不適合のため削除した（2026-04-17 の cleanup コミット参照）。学習データ作成等のオフライン前処理が必要になった時点で、`scripts/` 配下にオフライン専用スクリプトとして再導入することは可能。

## 現在のファイル構成

```
flare/
├── extractors/
│   ├── deca.py              ← 既存、real-time
│   ├── smirk.py             ← 既存、real-time（cuda128 化待ち）
│   ├── mediapipe_pnp.py     ← 新規実装済み（real-time カメラ tracker）
│   ├── deep3d.py            ← 既存
│   └── tdddfa.py            ← 既存スタブ
│
└── converters/
    ├── deca_to_flame.py       ← 既存
    └── smirk_to_flame.py      ← 新規実装済み
```

## 関連コミット

- `b8f34c6` : H1/H2/M1/M2 修正 + SMIRK / flame-head-tracker / SPECTRE スタブ追加
- （このメモのコミット）: flame-head-tracker / SPECTRE スタブ削除 + 本メモ追加

## TL;DR

- SMIRK cuda128 が完成したら、`SMIRKExtractor` + `SMIRKToFlameAdapter` で即 FlashAvatar を駆動可能
- 頭部の**姿勢（向き）**: SMIRK の `pose[:, 0:3]`（axis-angle）または MediaPipePnP の `R`
- 頭部の**位置（3D）**: MediaPipePnP の `t`（x, y, z）、特に `t[2]` がカメラからの距離
- DECA/SMIRK のカメラは弱透視近似（位置は z が heuristic）、真の位置は PnP が必要
- `MediaPipePnPTracker` は実装済み (`flare/extractors/mediapipe_pnp.py`)
- flame-head-tracker / SPECTRE はリアルタイム不適合のため不採用
