# FlashAvatar サブモジュール適用指示書

本ドキュメントは FLARE 本体の学習パイプライン修正に伴う **FlashAvatar サブモジュール側**
(`third_party/FlashAvatar` が指す `MTamon/FlashAvatar:release/cuda128-fixed` など)
への変更手順をまとめたものです。

FLARE リポジトリの claude セッションからはサブモジュールリポジトリに push できない
ため、同梱パッチ `docs/patches/flashavatar-head-pose-aware.patch` を手元で適用して
下さい。

## 適用対象

| ファイル | 目的 |
| --- | --- |
| `utils/flame_converter.py` | SMIRK の eyelid/shape_dim 修正 + `.frame` への bbox メタ埋め込み |
| `scene/__init__.py`         | `head_pose_aware` フラグで T[0]/T[1] を per-frame 補正 |
| `train.py`                  | `--head_pose_aware` CLI フラグと `Scene_mica` への引数伝播 |

## 適用手順

```bash
# 1) FlashAvatar サブモジュールのチェックアウト
cd third_party/FlashAvatar

# 2) 作業ブランチ作成 (release/cuda128-fixed を起点に)
git fetch origin
git checkout -b feat/head-pose-aware origin/release/cuda128-fixed

# 3) パッチ適用 (FLARE ルートからの相対パス)
git apply ../../docs/patches/flashavatar-head-pose-aware.patch

# 4) 差分確認
git diff --stat
# 期待: scene/__init__.py, train.py, utils/flame_converter.py の 3 ファイル

# 5) コミット & プッシュ
git add scene/__init__.py train.py utils/flame_converter.py
git commit -m "feat: SMIRK eyelid/shape_dim 修正 + head_pose_aware 追加"
git push -u origin feat/head-pose-aware

# 6) release/cuda128-fixed への PR を作成しマージ

# 7) FLARE 側で submodule ポインタ更新
cd ../..
git submodule update --remote third_party/FlashAvatar
git add third_party/FlashAvatar
git commit -m "chore: bump FlashAvatar submodule to head_pose_aware"
```

## 変更の意図

### 1. `utils/flame_converter.py` — SMIRK eyelid 欠落バグ修正

SMIRK は `eyelid` キー (2D) を tracker 出力に含むが、`TRACKER_CONFIGS["smirk"]` が
`has_eyelids=False` の既定値だったため silent drop されていた。

```python
# BEFORE
"smirk": TrackerConfig(name="SMIRK"),
# AFTER
"smirk": TrackerConfig(
    name="SMIRK",
    shape_dim=300,      # SMIRK は shape 300 次元
    has_eyelids=True,
    eyelids_key="eyelid",
),
```

加えて `convert()` に `head_pose_aware: bool = False` 引数と、bbox メタを
`.frame` 辞書に含める `_extract_bbox_payload()` / `_shift_K_to_bbox()` を追加。

### 2. `scene/__init__.py` — `head_pose_aware` per-frame 補正

3DGS レンダラは `FoVx`/`FoVy` ベースで動作し `K` の principal point
(cx, cy) を参照しない。そこで `head_pose_aware=True` のとき `.frame` の
`bbox` メタから weak-perspective translation の `T[0]`, `T[1]` を per-frame
に補正し、検出された頭部中心が描画画像中央からズレていても Gaussian が
その位置に描画されるようにする。

```python
# 画像空間のシフト Δpx を world 空間の Δ (depth / focal) に戻して T に加算
dx_world = (cx_crop - img_w_render / 2.0) * t_z / max(fl_x_render, 1e-6)
dy_world = (cy_crop - img_h_render / 2.0) * t_z / max(fl_y_render, 1e-6)
T[0] += dx_world
T[1] += dy_world
```

bbox メタが欠けているフレームは従来通り無補正で扱う後方互換構造。

### 3. `train.py` — CLI フラグ追加

```python
parser.add_argument("--head_pose_aware", action="store_true", ...)
# ...
scene = Scene_mica(..., head_pose_aware=args.head_pose_aware)
```

既定 off のオプトイン機能なので既存学習ジョブへの影響なし。

## 動作検証

FLARE 側テスト (`uv run pytest`) で以下を確認済み:

- SMIRK `.pt` → `.frame` 変換で eyelid が `[B, 2]` 形状で保存される
- `head_pose_aware=True` のとき K の `cx/cy` が bbox 中心に写像される
  (例: 元画像 (640,640) で crop (100,160)-(180,240)、bbox_center (110,210)
  を img_size (512,512) に写像 → K.cx, K.cy ≈ (320,320))

## 既知の制約

- `Scene_mica` 側の補正は x/y のみ。z 方向 (スケール) には介入しない
  想定で、FlashAvatar が学習中に scale を吸収する前提。
- fixed crop が square であることを前提にした scale 計算。非正方形クロップ
  が必要な場合は `_shift_K_to_bbox` の `scale_x/y` 計算を見直す必要あり。

## FLARE 側で既に実装済みの連携

- `configs/train_flashavatar.yaml`: `flashavatar.head_pose_aware: false`
- `flare/training.py`: `FlashAvatarSettings.head_pose_aware: bool`
- `scripts/train_flashavatar.py`: `--head_pose_aware` を Step 3
  (`flame_converter.py`) と Step 4 (`FlashAvatar/train.py`) に伝播
- `scripts/extract_{deca,smirk}_frames.py`: per-frame `.pt` に
  `bbox_center / bbox_scale / img_size / crop_bbox / crop_img_size` を保存
