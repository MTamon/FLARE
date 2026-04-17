# Guide: SMIRK 統合計画メモ

FLARE の real-time パイプラインに SMIRK（CVPR 2024）を DECA の代替として統合するための設計メモ。**本ファイルは将来実装時の参照資料であり、SMIRK の cuda128 版作成後に再参照することを想定している。**

## 背景

- FLARE は **real-time 処理**（≥25 FPS）が目的
- 現状は DECA で FLAME パラメータを抽出（~119 FPS、精度は中程度）
- SMIRK は DECA より表情精度が高く、かつ `eyelid`（瞼開閉 2D）を出力できる
- SMIRK の cuda128 版は user が今後作成予定

## 最終的に採用する構成

```text
Webcam/Video Frame (real-time)
  │
  ├── SMIRKExtractor           (flare/extractors/smirk.py)
  │     ├── exp     (50D)        ← FLAME expression
  │     ├── pose    (6D)         ← global_rotation (3D) + jaw_pose (3D)
  │     ├── cam     (3D)         ← weak-perspective [scale, tx, ty]
  │     └── eyelid  (2D)         ← 瞼開閉量
  │
  ├── SMIRKToFlameAdapter      (flare/converters/smirk_to_flame.py)
  │     ├── expr      (100D)     ← F.pad(exp, (0, 50))
  │     ├── jaw_pose  (6D)       ← axis_angle → rotation_6d
  │     ├── eyes_pose (12D)      ← identity_6d × 2  (★ 眼球 tracking なしの限界)
  │     └── eyelids   (2D)       ← SMIRK の eyelid をそのまま使用
  │
  └── FlashAvatar Renderer     (120D condition → 3DGS head)
```

### ポーズ（頭部回転）の取得について

**頭部回転（ポーズ）は取得できます。** 具体的には:

- `SMIRKExtractor` が出力する `pose` (6D) の先頭 3 次元 `pose[:, 0:3]` が **global_rotation**（頭部全体の axis-angle 回転）
- これが「頭がどちらを向いているか」の情報
- jaw_pose (`pose[:, 3:6]`) は顎の開閉回転
- 現在の `DECAToFlameAdapter` はこれを読み取って jaw_pose のみ変換する設計で、global_rotation は FlashAvatar に直接渡していない（FlashAvatar が内部で使用するカメラ R/t とは別系統）
- `SMIRKToFlameAdapter` も同じ方針で、頭部回転情報そのものは `pose[:, 0:3]` で取得可能

ただし、SMIRK のカメラは DECA と同じ **弱透視 [scale, tx, ty]** のみで、world-space の完全なカメラ外部パラメータ (K, R, t) は出力しない点に注意。FlashAvatar 学習時にはこの弱透視を `flame_converter.weak_perspective_to_full()` で擬似的に拡張して使っている（`focal_scale=5.0` のヒューリスティック）。

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
SMIRK は弱透視カメラしか出さないため、より精度の高い world-space カメラ外部パラメータ（K / R / t）が必要な場合は以下を追加:

| 候補 | 備考 |
|------|------|
| **MediaPipe FaceMesh + cv2.solvePnP** | GPU 200+ FPS、K は固定 / 要校正、R/t は PnP で取得。**推奨。** 既に MediaPipe は FLARE の依存。 |
| 3DDFA_V2（`tdddfa.py` 既存スタブ） | CPU ~740 FPS、12D affine 出力、ただし BFM 基底なので FLAME との座標整合が要 |

この段階は SMIRK 統合後に別タスクとして検討。

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
│   ├── deca.py          ← 既存、real-time
│   ├── smirk.py         ← 既存、real-time（cuda128 化待ち）
│   ├── deep3d.py        ← 既存
│   └── tdddfa.py        ← 既存スタブ
│
└── converters/
    ├── deca_to_flame.py       ← 既存
    └── smirk_to_flame.py      ← 新規（今回作成）
```

## 関連コミット

- `b8f34c6` : H1/H2/M1/M2 修正 + SMIRK / flame-head-tracker / SPECTRE スタブ追加
- （このメモのコミット）: flame-head-tracker / SPECTRE スタブ削除 + 本メモ追加

## TL;DR

- SMIRK cuda128 が完成したら、`SMIRKExtractor` + `SMIRKToFlameAdapter` で即 FlashAvatar を駆動可能
- 頭部ポーズは SMIRK の `pose[:, 0:3]` で取得可能（axis-angle 3D）
- カメラ外部パラメータがさらに必要なら MediaPipe FaceMesh + PnP を後で追加
- flame-head-tracker / SPECTRE はリアルタイム不適合のため不採用
