# Guide: FLAME ルート統合 ハンドオーバー資料

**対象日**: 2026-04-18（翌日作業用の引き継ぎ + レビュー資料）
**作成背景**: 2026-04-17 に DECA → FlashAvatar パイプラインの完成、SMIRK 統合準備、
MediaPipePnPTracker (GPU 対応) 実装を完了した。翌日は実機検証で FLAME ルートの
完成確認を行う予定であり、そのための手順と構成を集約する。

---

## 0. 現状確認（認識のすり合わせ）

ユーザが述べた「DECA→FlashAvatar / SMIRK(+gpu mediapipe)→FlashAvatar /
FlashAvatar 学習 が実装された」という認識の正確な対応関係:

| フロー | 実装状況 | 備考 |
|--------|---------|------|
| **DECA → FlashAvatar 学習** | ✅ 完成 | `scripts/train_flashavatar.sh` で 5 Step が完結 |
| **DECA → FlashAvatar リアルタイム推論** | ✅ 完成 | `configs/realtime_flame.yaml` + `FlameConverter` |
| **SMIRK → FlashAvatar 変換コード** | ✅ 完成 | `SMIRKExtractor` + `SMIRKToFlameAdapter` |
| **SMIRK モデル cuda128 ビルド** | ⚠️ 未完了 | `third_party/SMIRK` の cuda128 対応は user 作業待ち |
| **MediaPipePnPTracker（カメラ外部パラメータ）** | ✅ 完成 | Solutions + Tasks API 両対応、GPU delegate 実装済み |
| **MediaPipe GPU の実機動作確認** | ⚠️ 未実施 | `.task` モデルは取得済み、環境検証が翌日作業 |

**結論**: 「DECA → FlashAvatar」は学習・推論とも動作可能。
「SMIRK → FlashAvatar」は **FLARE 側のコードは完成**しているが、
SMIRK 本体の cuda128 ビルドが残っているため実走行は未検証。
「MediaPipe GPU」はコードレベルで完成、動作確認は翌日。

---

## 1. 実行手順

### 1.1. DECA → FlashAvatar: 学習パイプライン

**目的**: 入力動画から FlashAvatar 用の 3D Gaussian Splatting (3DGS) 個人化モデルを学習する。

**前提**:
- FLAME モデルアセット (`generic_model.pkl`, `FLAME_masks.pkl`) が `third_party/FlashAvatar/flame/` 配下に配置されていること
- DECA チェックポイント (`checkpoints/deca/deca_model.tar`) が存在すること
- 入力動画 ≥ 600 フレーム（論文推奨）、25 FPS 相当に間引きすることを推奨

**実行コマンド（ワンライナー）**:

```bash
bash scripts/train_flashavatar.sh \
    --id_name person01 \
    --video /path/to/person01.mp4 \
    --device cuda:0 \
    --img_size 512 \
    --fps 25 \
    --iterations 30000
```

**内部で実行される 5 Step**:

| Step | スクリプト | 処理内容 | 出力先 |
|------|-----------|---------|--------|
| 1 | `scripts/extract_deca_frames.py` | 動画 → フレーム画像 + DECA per-frame `.pt` | `data/flashavatar_training/<id>/imgs/*.jpg`, `deca_outputs/*.pt` |
| 2 | `scripts/generate_masks_mediapipe.py` | MediaPipe Selfie Segmentation で前景マスク生成 | `<id>/parsing/`, `<id>/alpha/` |
| 3 | `third_party/FlashAvatar/utils/flame_converter.py` (CLI) | DECA `.pt` → FlashAvatar `.frame` 変換 | `third_party/FlashAvatar/metrical-tracker/output/<id>/checkpoint/*.frame` |
| 4 | `third_party/FlashAvatar/train.py` | 3DGS 学習 (iterations=30000) | `<id>/log/ckpt/*.pth`, `<id>/log/point_cloud/` |
| 5 | `third_party/FlashAvatar/test.py` | 検証動画生成 | `<id>/log/test.avi` |

**再開・部分実行**:

```bash
# 再開（抽出済みから Step 2 以降）
bash scripts/train_flashavatar.sh --id_name person01 --video ... --skip_extract

# 学習のみ再開
bash scripts/train_flashavatar.sh --id_name person01 --skip_extract --skip_masks --skip_convert

# テスト動画生成のみ
bash scripts/train_flashavatar.sh --id_name person01 --test_only
```

**完了後の成果物**:
- 学習済みモデル: `checkpoints/flashavatar/<id>/point_cloud/`（シンボリックリンク）
- 検証動画: `data/flashavatar_training/<id>/log/test.avi`

---

### 1.2. DECA → FlashAvatar: リアルタイム推論

**目的**: Webcam / 動画ファイルから real-time に FLAME パラメータを抽出し、
学習済み FlashAvatar で 3DGS 顔レンダリングを行う。

**前提**:
- Step 1.1 で学習済み FlashAvatar モデルがあること
- Route B (FLAME) 用設定 `configs/realtime_flame.yaml` の `renderer.model_path` が
  対象話者のモデルを指していること

**コマンド**:

```bash
# 設定ファイルを対象話者に合わせる
sed -i "s|model_path: ./checkpoints/flashavatar/.*|model_path: ./checkpoints/flashavatar/person01/|" \
    configs/realtime_flame.yaml

# Webcam (source=0) で実行
python examples/realtime_extract.py \
    --config configs/realtime_flame.yaml \
    --source 0
```

**内部処理フロー**:

```
frame (H×W BGR)
  │
  ├─ face_detect.py: 顔検出 + crop(224×224)
  │
  ├─ DECAExtractor.extract(image_224): {shape, exp, pose, cam, light, detail}
  │
  ├─ DECAToFlameAdapter.convert(): {expr(100), jaw_pose(6), eyes_pose(12), eyelids(2)}
  │
  ├─ (内部) weak_perspective_to_full(cam): cam(scale,tx,ty) → 近似 K/R/t
  │
  └─ FlashAvatarRenderer.render(condition_120D, K, R, t) → rendered frame
```

---

### 1.3. SMIRK → FlashAvatar（SMIRK cuda128 ビルド完了後）

**目的**: DECA の代わりに SMIRK を用い、より精度の高い表情 + 瞼開閉量を抽出する。

**前提条件（user 作業）**:
- `third_party/SMIRK` が cuda128 環境でビルド可能であること
- `checkpoints/smirk/smirk_encoder.pt` が配置済み
- `src.smirk_encoder.SmirkEncoder` が import 可能であること

**動作確認コード**:

```python
from flare.extractors.smirk import SMIRKExtractor
from flare.converters.smirk_to_flame import SMIRKToFlameAdapter

extractor = SMIRKExtractor(
    model_path="./checkpoints/smirk/smirk_encoder.pt",
    device="cuda:0",
    smirk_dir="./third_party/SMIRK",
)
adapter = SMIRKToFlameAdapter()

# image_tensor: (1, 3, H, W), values in [0, 1], face-cropped
params = extractor.extract(image_tensor)
# params: {shape(300), exp(50), pose(6), cam(3), eyelid(2)}

flash_params = adapter.convert(params)
# flash_params: {expr(100), jaw_pose(6), eyes_pose(12), eyelids(2)}
assert flash_params["expr"].shape == (1, 100)
assert flash_params["jaw_pose"].shape == (1, 6)
assert flash_params["eyelids"].shape == (1, 2)  # SMIRK 実測値（DECA と違いゼロでない）
```

**学習用スクリプトへの組み込み**:
`scripts/train_flashavatar.sh` の Step 1 / Step 3 を SMIRK 用に差し替える（未実装）:

1. `scripts/extract_smirk_frames.py` を作成（`extract_deca_frames.py` を流用、
   `DECAExtractor` → `SMIRKExtractor` に差し替え）
2. Step 3 の CLI 引数を `--tracker smirk` に変更（`FlameConverter` は既に
   `TRACKER_CONFIGS["smirk"]` をサポート済）

---

### 1.4. MediaPipe GPU (Tasks API) の動作確認

**目的**: `MediaPipePnPTracker` の GPU バックエンドが
CUDA 12.8 + NVIDIA EGL 環境で実際に動作するか検証する。

**前提**:
- `face_landmarker.task` モデルファイルが取得済み（ユーザ確認済）
- 配置推奨先: `checkpoints/mediapipe/face_landmarker.task`

**Step 1: CPU Tasks backend で動作確認**（GPU より先にこちらから）:

```python
import cv2
from flare.extractors.mediapipe_pnp import MediaPipePnPTracker

tracker = MediaPipePnPTracker(
    backend="tasks",
    model_path="./checkpoints/mediapipe/face_landmarker.task",
    gpu=False,
)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = tracker.track(frame)
    if result is not None:
        print("K:", result["K"].shape, "R:", result["R"].shape, "t:", result["t"])
        print("t (3D position):", result["t"].cpu().numpy())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
tracker.release()
```

**Step 2: GPU delegate で動作確認**:

```bash
# NVIDIA EGL の可視化（必要な場合）
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

python -c "
from flare.extractors.mediapipe_pnp import MediaPipePnPTracker
tracker = MediaPipePnPTracker(
    backend='tasks',
    model_path='./checkpoints/mediapipe/face_landmarker.task',
    gpu=True,
)
print('GPU backend initialized OK')
print('backend:', tracker.backend)
tracker.release()
"
```

**失敗時の確認事項**:
- `RuntimeError: Unable to initialize EGL` → libEGL / NVIDIA driver 未インストール
- `ImportError: mediapipe not found` → `pip install mediapipe>=0.10`
- Silent fallback は**しない**（Opus 4.7 検証で確認済）: EGL 不在時は明示的に RuntimeError

**Step 3: FPS ベンチマーク**:

```python
import time, cv2
from flare.extractors.mediapipe_pnp import MediaPipePnPTracker

for backend, gpu, label in [
    ("solutions", False, "Solutions (CPU)"),
    ("tasks", False, "Tasks (CPU)"),
    ("tasks", True, "Tasks (GPU)"),
]:
    tracker = MediaPipePnPTracker(
        backend=backend,
        model_path="./checkpoints/mediapipe/face_landmarker.task" if backend == "tasks" else None,
        gpu=gpu,
    )
    cap = cv2.VideoCapture(0)
    n_frames = 100
    t0 = time.time()
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        tracker.track(frame)
    dt = time.time() - t0
    print(f"{label}: {n_frames / dt:.1f} FPS")
    cap.release()
    tracker.release()
```

**期待値（RTX 3090）**:
- Solutions (CPU): 80-120 FPS
- Tasks (CPU): 80-120 FPS
- Tasks (GPU): 150-200 FPS

**Step 4: DECA と組み合わせてリアルタイム動作確認**:

```python
# DECA + MediaPipePnPTracker の並列実行で、
# FlashAvatar に真の K/R/t を供給する
from flare.extractors.deca import DECAExtractor
from flare.extractors.mediapipe_pnp import MediaPipePnPTracker
from flare.converters.deca_to_flame import DECAToFlameAdapter
from third_party.FlashAvatar.utils.flame_converter import FlameConverter

deca = DECAExtractor(...)
tracker = MediaPipePnPTracker(backend="tasks", model_path="...", gpu=True)
adapter = DECAToFlameAdapter()
converter = FlameConverter(tracker="deca")

while True:
    frame = cam.read()
    cam_result = tracker.track(frame)  # full-res frame
    cropped = face_detect.crop(frame)   # 224x224 for DECA
    deca_out = deca.extract(cropped)

    if cam_result is not None:
        frame_dict = converter.convert(
            deca_out,
            img_size=(512, 512),
            camera_K=cam_result["K"],
            camera_R=cam_result["R"],
            camera_t=cam_result["t"],
        )
    else:
        frame_dict = converter.convert(deca_out, img_size=(512, 512))

    # frame_dict を FlashAvatar renderer に渡す
```

---

## 2. ファイルの配置

### 2.1. リポジトリ全体構造（FLAME ルート関連のみ）

```
/home/user/FLARE/
├── configs/
│   ├── realtime_flame.yaml              # Route B (FLAME) リアルタイム設定
│   ├── realtime_bfm.yaml                # Route A (BFM) ※FLAME ルート外
│   ├── lhg_extract_deca.yaml            # DECA 使用の LHG 前処理設定
│   └── train_face_decoder.yaml
│
├── docs/
│   ├── guide_deca_flashavatar.md        # DECA → FlashAvatar 統合ガイド（既存）
│   ├── guide_smirk_integration.md       # SMIRK 統合計画メモ（本日更新）
│   ├── guide_route_switching.md
│   ├── guide_realtime.md
│   └── guide_flame_route_handover.md    # ★本資料
│
├── flare/
│   ├── extractors/
│   │   ├── base.py                      # BaseExtractor 抽象クラス
│   │   ├── deca.py                      # DECAExtractor（cuda128）
│   │   ├── smirk.py                     # SMIRKExtractor（SMIRK cuda128 待ち）
│   │   ├── mediapipe_pnp.py             # ★本日実装: GPU 対応 K/R/t トラッカ
│   │   ├── deep3d.py                    # Deep3DFaceRecon（BFM ルート）
│   │   └── tdddfa.py                    # （スタブ）
│   │
│   ├── converters/
│   │   ├── base.py                      # BaseAdapter 抽象クラス
│   │   ├── deca_to_flame.py             # DECA → FlashAvatar 120D 変換
│   │   ├── smirk_to_flame.py            # ★本日作業: SMIRK → FlashAvatar 120D 変換
│   │   ├── bfm_to_flame.py              # （BFM ルート）
│   │   └── registry.py
│   │
│   ├── renderers/
│   │   ├── flashavatar.py               # FlashAvatarRenderer（Route B）
│   │   ├── pirender.py                  # PIRenderer（Route A）
│   │   └── headgas.py
│   │
│   └── pipeline/
│       ├── realtime.py                  # リアルタイム実行エンジン
│       └── lhg_batch.py                 # LHG 前処理バッチ
│
├── scripts/
│   ├── extract_deca_frames.py           # 動画 → DECA per-frame 抽出
│   ├── generate_masks_mediapipe.py      # MediaPipe で前景マスク生成
│   ├── train_flashavatar.sh             # 5 Step 学習パイプライン
│   └── extract_smirk_frames.py          # ※未作成（SMIRK 稼働後に作る）
│
├── third_party/
│   ├── DECA/                            # MTamon/DECA@cuda128 サブモジュール
│   ├── FlashAvatar/                     # MTamon/FlashAvatar@release/cuda128-fixed
│   │   ├── train.py                     # FlashAvatar 学習本体
│   │   ├── test.py                      # FlashAvatar 検証動画生成
│   │   ├── utils/flame_converter.py     # .pt → .frame 変換（DECA/SMIRK 共用）
│   │   ├── flame/generic_model.pkl      # ※FLAME 公式から取得
│   │   ├── flame/FLAME_masks/           # ※FLAME 公式から取得
│   │   └── metrical-tracker/output/<id>/checkpoint/*.frame
│   └── SMIRK/                           # ※cuda128 ビルド未完了
│
├── checkpoints/
│   ├── deca/deca_model.tar              # DECA 学習済みモデル
│   ├── smirk/smirk_encoder.pt           # ※SMIRK cuda128 稼働後に配置
│   ├── mediapipe/face_landmarker.task   # ★取得済、配置を明日確認
│   ├── flashavatar/<id>/point_cloud/    # 学習完了後の個人化 3DGS
│   └── l2l/l2l_vqvae.pth
│
└── data/
    └── flashavatar_training/<id>/        # 学習データ（Step 1-3 出力）
        ├── imgs/                         # 1-indexed フレーム画像
        ├── deca_outputs/                 # 0-indexed DECA per-frame .pt
        ├── parsing/                      # neckhead / mouth マスク
        ├── alpha/                        # アルファマスク
        ├── log/ckpt/*.pth                # FlashAvatar 学習チェックポイント
        ├── log/point_cloud/              # 個人化 3DGS
        └── log/test.avi                  # 検証動画
```

### 2.2. 本日（2026-04-17）作成 / 更新したファイル

| ファイル | 状態 | 目的 |
|---------|------|------|
| `flare/extractors/mediapipe_pnp.py` | 新規 → 修正 | GPU 対応 K/R/t トラッカ（Solutions + Tasks dual backend） |
| `flare/converters/smirk_to_flame.py` | 新規 | SMIRK → FlashAvatar 変換（eyelid 対応） |
| `flare/extractors/smirk.py` | 既存（未変更） | SMIRK cuda128 ビルド後に稼働 |
| `docs/guide_smirk_integration.md` | 更新 | FlashAvatar 世界座標 / MediaPipePnP 組み合わせ記述追加 |
| `docs/guide_flame_route_handover.md` | 新規 | ★本資料 |

### 2.3. 本日削除したファイル

| ファイル | 削除理由 |
|---------|---------|
| `flare/extractors/flame_head_tracker.py` | photometric optimization で数秒/フレーム、real-time 不適合 |
| `flare/extractors/spectre.py` | 双方向時系列 CNN で ±2 フレーム遅延 (~80 ms)、real-time 不適合 |

詳細: `docs/guide_smirk_integration.md` 末尾「調査で除外した候補」節。
学習データ作成など offline 用途が必要になった時点で再導入は可能。

### 2.4. 本日の関連コミット

```
c46f426 fix(mediapipe_pnp): guard against NaN solvePnP output and document thread safety
78687f7 Refactor MediaPipePnPTracker: dual-backend (solutions/tasks) with GPU support
0267b8a Fix MediaPipePnPTracker verification findings
6e9ba1e Add MediaPipePnPTracker for real-time camera extrinsics (Plan C)
fe66c47 Remove non-realtime extractor stubs; add SMIRK integration memo
b8f34c6 Fix crop margin, fixed bbox, mouth mask, jaw_pose; add SMIRK/FHT/SPECTRE stubs
```

すべて `claude/integrate-deca-flashavatar-7MPo0` ブランチに push 済。

### 2.5. 依存する外部アセット（DL 必要）

| アセット | 配置先 | 取得元 |
|---------|-------|--------|
| FLAME 汎用モデル | `third_party/FlashAvatar/flame/generic_model.pkl` | https://flame.is.tue.mpg.de/ |
| FLAME マスク | `third_party/FlashAvatar/flame/FLAME_masks/FLAME_masks.pkl` | 同上 |
| DECA モデル | `checkpoints/deca/deca_model.tar` | `install/setup_deca.sh` |
| SMIRK モデル | `checkpoints/smirk/smirk_encoder.pt` | georgeretsi/smirk |
| MediaPipe Tasks | `checkpoints/mediapipe/face_landmarker.task` | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task |

---

（後半: **制限事項** と **意図・ライブラリ選定** は続編で記述します）
