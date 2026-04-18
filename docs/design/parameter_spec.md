# 特徴抽出器 / レンダラー パラメータ仕様

更新日: 2026-04-18  
対象モデル (今版): DECA, SMIRK, FlashAvatar  
未対応 (次版追記): HeadGaS, PIRender, Deep3DFaceRecon, 3DDFA V2

本ドキュメントは Converter を実装する際の一次情報源として使用すること。  
コードを読まずにこのファイルだけで変換ロジックを書けることを目標とする。

`⚠️` 印は Phase 2 で MTamon フォーク実装を直接確認して更新が必要な箇所。

---

## 目次

1. [共通規約](#1-共通規約)
2. [DECA Extractor](#2-deca-extractor)
3. [SMIRK Extractor](#3-smirk-extractor)
4. [DECA / SMIRK 差分まとめ](#4-deca--smirk-差分まとめ)
5. [補助情報源: MediaPipe Face Landmarker](#5-補助情報源-mediapipe-face-landmarker)
6. [FlashAvatar Renderer](#6-flashavatar-renderer)

---

## 1. 共通規約

### テンソル記法

```
B  : バッチサイズ (リアルタイム時は常に 1)
T  : 時系列フレーム数 (バッチ保存時)
dtype: float32 (特記なき限り)
```

### 画像テンソル

```
形状 : (B, 3, H, W)
値域 : [0.0, 1.0]
チャネル順: RGB
```

入力画像はすべて `flare/utils/face_detect.py` の `FaceDetector.crop_and_align()` で
クロップ・正方形リサイズ済みの顔領域であること。

### 回転表現の種類

本プロジェクトで使われる回転表現を列挙する。Converter 実装時に混同しないこと。

| 名称 | 次元 | 説明 |
|---|---|---|
| axis-angle (AA) | 3D | 回転軸と回転角を 1 ベクトルにエンコード。ベクトルの大きさが回転角 (rad)。FLAME の生パラメータ。 |
| rotation matrix (R) | 3×3 | SO(3) 回転行列。|
| rotation 6D (6d) | 6D | 回転行列の最初の 2 列をフラット化 (Zhou et al. 2019)。FlashAvatar が要求。|
| quaternion (q) | 4D | FLARE では使用しない。|

変換フロー:
```
AA (3D) → rotation matrix (3×3) → rotation 6D (6D)
```

---

## 2. DECA Extractor

- リポジトリ: `MTamon/DECA@cuda128` (third_party/DECA)
- 実装: `flare/extractors/deca.py`
- 入力画像サイズ: **224×224 px**
- 座標系: **右手系、+X 右、+Y 上、+Z 視点方向 (画面手前)**

### 2.1 出力パラメータ一覧

| キー | 形状 | 単位/値域 | 物理的意味 |
|---|---|---|---|
| `shape` | (B, 100) | 無次元、典型 ±3σ 内 | FLAME identity (形状) PCA 係数 第1-100 主成分 |
| `tex` | (B, 50) | 無次元 | FLAME テクスチャ PCA 係数 第1-50 主成分 |
| `exp` | (B, 50) | 無次元、典型 ±3σ 内 | FLAME expression PCA 係数 第1-50 主成分 |
| `pose` | (B, 6) | rad (axis-angle) | 頭部姿勢 + 顎関節 (詳細は §2.2) |
| `cam` | (B, 3) | 無次元 | 弱透視カメラパラメータ (詳細は §2.3) |
| `light` | (B, 27) | 無次元 | SH 照明係数 9×3 (詳細は §2.4) |
| `detail` | (B, 128) | 無次元 | 詳細変形コード E_detail (Detailed DECA のみ使用) |

LHG 前処理では `return_keys: [shape, exp, pose, cam]` のみ取得する
(tex, light, detail は L2L 入力に不要)。

### 2.2 pose パラメータ

```
pose = [global_rot | jaw_pose]
       [  0:3      |   3:6   ]
```

| スライス | 意味 | 表現 | 値域 |
|---|---|---|---|
| `pose[:, 0:3]` | **global rotation**: ワールド座標系における頭部全体の回転 | axis-angle 3D | 典型 ±π/2 rad |
| `pose[:, 3:6]` | **jaw pose**: 顎関節の開閉回転 | axis-angle 3D | 典型 [0, 0.5] rad (X 成分が主) |

- global_rot は **頭部中心を基準とした回転** (平行移動を含まない)
- jaw は FLAME の joint として bone 中心回転。Y/Z 成分は通常ほぼ 0
- 軸の向き: +X 軸周りが pitch (うなずき)、+Y 軸周りが yaw (首振り)、+Z 軸周りが roll (傾け)

### 2.3 cam パラメータ (弱透視カメラ)

```
cam = [s, tx, ty]
```

| インデックス | 意味 | 値域 |
|---|---|---|
| `cam[:, 0]` | **scale** `s`: 正規化画像座標系でのスケール | 典型 5–12 (224px 入力時) |
| `cam[:, 1]` | **tx**: X 方向平行移動 (正規化座標) | 典型 ±0.3 |
| `cam[:, 2]` | **ty**: Y 方向平行移動 (正規化座標) | 典型 ±0.3 |

**投影式 (DECA 内部):**

```
# 3D 頂点 v (FLAME メッシュ) を 2D 点 p に投影する
# 入力: v ∈ R^3 (FLAME canonical space)
# 出力: p ∈ [-1, 1]^2 (正規化画像座標)

p_x = s * (v_x + tx)
p_y = s * (v_y + ty)
```

- 正規化画像座標 [-1, 1]²  ← → ピクセル座標 [0, 224]²  は `p_pixel = (p_norm + 1) * 112`
- `s` が大きいほど顔が画像内で大きく映る (カメラに近い / 顔が大きい)
- `tx > 0` で顔が右に移動、`ty > 0` で下に移動 (Y 軸が画像上方向に反転していることに注意)
- **depth (Z 方向距離) の情報は cam に含まれない** (弱透視のため)

### 2.4 light パラメータ

```
light = [SH 係数].reshape(9, 3)
```

- 形状: (B, 27) → reshape(9, 3) で (9 基底, RGB)
- 球面調和関数 (SH) 第1-9 基底係数を RGB チャネルで表現
- Converter では通常この情報は**使用しない** (FlashAvatar が照明を内部管理するため)

### 2.5 detail パラメータ

- E_detail エンコーダ (ResNet) で得られる 128 次元の詳細変形コード
- Detail DECA (高精度モード) でのみ意味を持つ
- FLARE では現状使用しない

---

## 3. SMIRK Extractor

- リポジトリ: `MTamon/smirk@release/cuda128` (third_party/smirk — Phase 2 で submodule 追加)
- 実装: `flare/extractors/smirk.py`
- 入力画像サイズ: **224×224 px** (⚠️ 要確認: 224 か 256 か)
- 座標系: **右手系、+X 右、+Y 上、+Z 視点方向** (⚠️ DECA と同じか要確認)
- 論文: "SMIRK: Semi-supervised 3D Face Reconstruction with Neural Inverse Rendering" (2024)

### 3.1 出力パラメータ一覧

| キー | 形状 | 単位/値域 | 物理的意味 |
|---|---|---|---|
| `shape` | (B, 300) | 無次元、典型 ±3σ 内 | FLAME identity (形状) PCA 係数 **第1-300 主成分** (DECA の 3 倍) |
| `exp` | (B, 50) | 無次元、典型 ±3σ 内 | FLAME expression PCA 係数 第1-50 主成分 |
| `pose` | (B, 6) | ⚠️ rad または rotation_6d か要確認 | 頭部姿勢 + 顎関節 (詳細は §3.2) |
| `cam` | (B, 3) | ⚠️ 要確認 | カメラパラメータ (詳細は §3.3) |
| `eyelid` | (B, 2) | [0.0, 1.0] | 瞼の閉じ度合い (DECA に存在しない) (詳細は §3.4) |

### 3.2 pose パラメータ ⚠️

SMIRK の pose も DECA と同じ layout の可能性が高いが、**MTamon fork の実装を直接確認すること**。

```
pose = [global_rot | jaw_pose]   ← DECA と同レイアウトの場合
       [  0:3      |   3:6   ]
```

**既知の相違点 (論文・upstream コードから):**

1. **global_rot の基準フレーム**: SMIRK は DECA と同じく顔中心基準だが、
   正規化方法が異なる可能性がある。SMIRK は入力画像クロップ時の変換を
   cam に吸収させる設計のため、global_rot の値域が DECA より小さくなる可能性あり。

2. **jaw の表現**: axis-angle か rotation_6d かを `third_party/smirk/src/smirk_encoder.py`
   の SmirkEncoder 出力層で確認する。

確認コマンド (Phase 2 で実行):
```python
# SmirkEncoder の出力次元確認
import torch
from src.smirk_encoder import SmirkEncoder
enc = SmirkEncoder()
dummy = torch.zeros(1, 3, 224, 224)
out = enc(dummy)
for k, v in out.items():
    print(k, v.shape)
```

### 3.3 cam パラメータ ⚠️

**DECA との主な相違点:**

SMIRK は DECA の弱透視カメラとは異なる投影モデルを採用している可能性がある。
upstream SMIRK の実装では、カメラモデルに以下のパターンが報告されている:

**パターン A (弱透視、DECA と同形式):**
```
cam = [s, tx, ty]
```
DECA と同じ式だが **スケール値の範囲が異なる** 可能性:
- SMIRK は入力を `[-1, 1]` に正規化してから処理するため、
  DECA の `s ≈ 5–12` に対して SMIRK の `s ≈ 0.8–1.2` 程度になることがある。

**パターン B (正規化弱透視):**
```
cam = [s_norm, tx_norm, ty_norm]
# s_norm は [0, 1] スケールに正規化済み
```

**パターン C (完全透視投影):**
```
cam = [focal_length, tx, ty]
# focal_length はピクセル単位の焦点距離
```

⚠️ **Phase 2 確認事項**:
`third_party/smirk/src/smirk_encoder.py` または
`third_party/smirk/src/FLAME/FLAME.py` の `forward()` 内の
投影コード (`weak_perspective_camera` または `full_perspective_camera`) を確認する。

DECA→FlashAvatar Converter では cam を使用しないため実害は少ないが、
SMIRK→LHG バッチ抽出の `centroid` / `face_size` 計算で使用するため確定が必要。

### 3.4 eyelid パラメータ

```
eyelid = [left_eyelid, right_eyelid]
          [     0     ,      1      ]
```

| インデックス | 意味 | 値域 |
|---|---|---|
| `eyelid[:, 0]` | **左目**: 瞼の閉じ度合い | [0.0, 1.0] (0=全開、1=完全閉) |
| `eyelid[:, 1]` | **右目**: 瞼の閉じ度合い | [0.0, 1.0] (0=全開、1=完全閉) |

- FLAME の `eyelids` パラメータに直接対応
- DECA にはこのパラメータが存在しないため、DECA 経路では MediaPipe blendshape で代替
- FlashAvatar の `eyelids (B, 2)` にそのままマップできる

### 3.5 SMIRK の shape と DECA の shape の違い

```
DECA  shape: (B, 100)  ← FLAME generic_model.pkl の shapedirs[:, :, :100]
SMIRK shape: (B, 300)  ← FLAME generic_model.pkl の shapedirs[:, :, :300]
```

- 同一の FLAME `generic_model.pkl` から取る PCA 係数であり、**同一空間に属する**
- SMIRK は第101-300 主成分まで利用することでより細かい個人差を表現できる
- FlashAvatar は shape を入力として使わないため、Converter でこのキーは drop する

---

## 4. DECA / SMIRK 差分まとめ

| 項目 | DECA | SMIRK | 備考 |
|---|---|---|---|
| shape 次元 | 100 | 300 | 同一 FLAME PCA 空間、SMIRK が高次元 |
| exp 次元 | 50 | 50 | 同一 |
| pose layout | [global_rot(3), jaw(3)] AA | ⚠️ 要確認 (同形式とみられる) | |
| cam model | 弱透視 [s, tx, ty] | ⚠️ 要確認 (パターン A-C のいずれか) | スケール値域が異なる可能性大 |
| cam 座標正規化 | 224px 基準 | ⚠️ 要確認 | |
| eyelid | **なし** | **(B, 2)** | SMIRK のみ直接出力 |
| tex | (B, 50) | **なし** | DECA のみ |
| light | (B, 27) | **なし** | DECA のみ |
| detail | (B, 128) | **なし** | DECA のみ |
| 入力サイズ | 224×224 | ⚠️ 要確認 | |
| pytorch3d 依存 | あり (rasterizer) | **なし** (⚠️ install_128.sh で確認済み) | |

### FlashAvatar 向け Converter で必要な情報

| FlashAvatar 入力キー | DECA 経路 | SMIRK 経路 |
|---|---|---|
| `expr` (B, 100) | `exp` ゼロパディング (50→100) | `exp` ゼロパディング (50→100) |
| `jaw_pose` (B, 6) | `pose[:, 3:6]` AA→rotation_6d 変換 | `pose[:, 3:6]` AA→rotation_6d 変換 (⚠️ レイアウト確認後) |
| `eyes_pose` (B, 12) | **MediaPipe blendshape から補完** | **MediaPipe blendshape から補完** (eyelid は別途) |
| `eyelids` (B, 2) | **MediaPipe blendshape から補完** | `eyelid` をそのまま使用 |

---

## 5. 補助情報源: MediaPipe Face Landmarker

- 実装: `flare/utils/face_detect.py` の `FaceDetector.detect_eye_pose()`
- バージョン: MediaPipe 0.10.11 (`mp.solutions` API) または 0.10.14+ (`mp.tasks` API)

### 5.1 使用する blendshape スコア

ARKit 互換の 52 blendshape スコアのうち、以下を使用する:

| blendshape 名 | 意味 | 値域 |
|---|---|---|
| `eyeLookUpLeft` | 左目: 上方向視線 | [0, 1] |
| `eyeLookDownLeft` | 左目: 下方向視線 | [0, 1] |
| `eyeLookInLeft` | 左目: 内側 (鼻方向) 視線 | [0, 1] |
| `eyeLookOutLeft` | 左目: 外側視線 | [0, 1] |
| `eyeLookUpRight` | 右目: 上方向視線 | [0, 1] |
| `eyeLookDownRight` | 右目: 下方向視線 | [0, 1] |
| `eyeLookInRight` | 右目: 内側視線 | [0, 1] |
| `eyeLookOutRight` | 右目: 外側視線 | [0, 1] |
| `eyeBlinkLeft` | 左目: 瞼閉じ | [0, 1] |
| `eyeBlinkRight` | 右目: 瞼閉じ | [0, 1] |

### 5.2 blendshape → FLAME eyes_pose / eyelids 変換

```python
# スケール係数 (face_detect.py:69-73 参照)
EYE_PITCH_SCALE = 0.35  # rad / blendshape score
EYE_YAW_SCALE   = 0.45  # rad / blendshape score

# pitch (上下): 下方向が正
pitch = (lookDown - lookUp) * EYE_PITCH_SCALE

# yaw (左右): 左目は outward が正、右目は inward が正
yaw_left  = (lookOutLeft  - lookInLeft)  * EYE_YAW_SCALE
yaw_right = (lookInRight  - lookOutRight) * EYE_YAW_SCALE

# axis-angle → rotation_6d (Rodrigues + 最初2列取り出し)
left_6d  = aa_to_rotation_6d([pitch_left,  yaw_left,  0])
right_6d = aa_to_rotation_6d([pitch_right, yaw_right, 0])

eyes_pose = cat([left_6d, right_6d], dim=-1)  # (1, 12)
eyelids   = [[eyeBlinkLeft, eyeBlinkRight]]    # (1, 2)
```

### 5.3 DECA 経路での使用

DECA は eyes_pose / eyelids を出力しないため、MediaPipe で補完する:
- `eyes_pose`: blendshape から計算 (上記 5.2)
- `eyelids`: blendshape の `eyeBlink` スコアをそのまま使用

### 5.4 SMIRK 経路での使用

SMIRK は `eyelid (B, 2)` を直接出力するため:
- `eyes_pose`: MediaPipe で補完 (DECA と同じ)
- `eyelids`: **SMIRK の `eyelid` を優先使用**、MediaPipe は SMIRK 失敗時のフォールバック

### 5.5 MediaPipe が利用できない場合のフォールバック

```python
# face_detect.py:76-83 参照
eyes_pose = [[1,0,0,0,1,0, 1,0,0,0,1,0]]  # identity rotation × 2 (1, 12)
eyelids   = [[0.0, 0.0]]                   # 開眼 (1, 2)
```

---

## 6. FlashAvatar Renderer

- リポジトリ: `MTamon/FlashAvatar@release/cuda128-fixed` (third_party/FlashAvatar)
- 実装: `flare/renderers/flashavatar.py`
- 性能: **300 FPS @ 512×512** (RTX 3090 想定)
- 学習: **対象人物ごとに個別学習が必要** (約 30 分 / RTX 3090、VRAM 12GB+)
- 公開 Pretrained 重み: MingSHTM 公式から `.ply` をダウンロード可 (sanity check 用)

### 6.1 入力: condition vector (120D)

FlashAvatar はフレームごとに **1 本の 120 次元ベクトル** を受け取る。
内部では以下の順序で結合 (`torch.cat`) される:

```python
condition = torch.cat([expr, jaw_pose, eyes_pose, eyelids], dim=-1)
# condition.shape == (B, 120)
```

| キー | 形状 | 表現 | 物理的意味 |
|---|---|---|---|
| `expr` | (B, 100) | FLAME expression PCA 係数 | FLAME generic_model 第1-100 主成分。DECA/SMIRK の exp (50D) をゼロパディングして生成。 |
| `jaw_pose` | (B, 6) | rotation_6d | 顎関節の回転。DECA/SMIRK の `pose[:, 3:6]` (axis-angle 3D) を rotation_6d に変換して生成。 |
| `eyes_pose` | (B, 12) | rotation_6d × 2 | 左右眼球回転。`[left_6d(0:6) \| right_6d(6:12)]`。MediaPipe blendshape から生成 (§5 参照)。 |
| `eyelids` | (B, 2) | [0.0, 1.0] スカラー | 瞼の閉じ度合い。`[left, right]`。SMIRK 経路は直接取得、DECA 経路は MediaPipe で補完。 |

**条件ベクトルのオフセット一覧:**

```
condition[...,   0:100]  = expr      (100D)
condition[..., 100:106]  = jaw_pose  (  6D)
condition[..., 106:118]  = eyes_pose ( 12D)
condition[..., 118:120]  = eyelids   (  2D)
```

#### 6.1.1 expr — DECA / SMIRK からの変換

```
DECA:  exp (B, 50) → F.pad(exp, (0, 50)) → expr (B, 100)
SMIRK: exp (B, 50) → F.pad(exp, (0, 50)) → expr (B, 100)
```

第 51-100 主成分はゼロ充填。FlashAvatar の学習時も同様の充填を使用している場合、
精度的損失はない。高精度が必要な場合は第 51-100 成分に対応した Extractor を検討すること。

#### 6.1.2 jaw_pose — rotation 変換

```
pose[:, 3:6]  (axis-angle 3D)
  ↓ Rodrigues
rotation matrix (3×3)
  ↓ 最初 2 列を row-major にフラット化
jaw_pose  (B, 6)  = [R[:,0], R[:,1]].flatten()
```

FlashAvatar は jaw の global_rot (`pose[:, 0:3]`) を condition vector に含まない。
global_rot はレンダリング時に FlashAvatar 内部の視点変換で処理される。

#### 6.1.3 eyes_pose — レイアウト詳細

```
eyes_pose = [left_6d | right_6d]
            [  0:6   |   6:12  ]
```

各 6D は rotation matrix の最初 2 列を row-major にフラット化したもの:

```
rot6d = [r00, r01, r10, r11, r20, r21]
```

#### 6.1.4 eyelids — 値域と向き

```
eyelids = [left_eyelid, right_eyelid]
```

- `0.0` = 完全開眼、`1.0` = 完全閉眼
- SMIRK 出力キー名は `eyelid` (単数形) だが FlashAvatar のキーは `eyelids` (複数形)
- Converter で rename が必要

### 6.2 出力

| 項目 | 値 |
|---|---|
| テンソル形状 | `(B, 3, H, W)` |
| 値域 | `[0.0, 1.0]` (clamp 済み) |
| チャネル順 | RGB |
| デフォルト解像度 | **512×512 px** |
| 背景色 | **黒 (0, 0, 0)** |
| dtype | float32 |

解像度が `output_size` と異なる場合は `F.interpolate(..., mode="bilinear")` でリサイズされる。

### 6.3 setup() — モデルロード

```python
renderer.setup()
# 実質的に以下を実行:
# model = GaussianModel(sh_degree=3)
# model.load_ply("<model_path>/point_cloud/iteration_30000/point_cloud.ply")
# model.to(device)
```

| 項目 | 詳細 |
|---|---|
| ロードファイル | `<model_path>/point_cloud/iteration_30000/point_cloud.ply` |
| イテレーション上書き | `setup(iteration=N)` で `iteration_N` に変更可 |
| source_image | **不要** (FlashAvatar は per-person 学習済みモデルを使用するため) |
| 学習ライブラリ | `scene.gaussian_model.GaussianModel`, `gaussian_renderer.render` |

モデルパスの構造:

```
<model_path>/
└── point_cloud/
    └── iteration_30000/
        └── point_cloud.ply   ← load_ply() がロードするファイル
```

### 6.4 render() — バッチ処理

```python
for i in range(batch_size):
    cond_i = condition[i:i+1]          # (1, 120)
    render_out = gaussian_render(
        model, cond_i,
        bg_color=torch.zeros(3, device=device),
    )
    image = render_out["render"]       # (3, H, W)
```

FlashAvatar の内部 Gaussian Renderer はバッチ非対応のため、B > 1 の場合はループ処理する。
現状の `flare/renderers/flashavatar.py` 実装はこのループ方式を採用している。

### 6.5 Converter — DECA / SMIRK → FlashAvatar 変換フロー

#### DECA 経路 (DECAToFlameAdapter 実装済み)

```
DECA.extract(image)
  → {exp(50), pose(6), cam(3), shape(100), ...}

DECAToFlameAdapter.convert(deca_params)
  → {
      expr     : F.pad(exp, (0,50))          # (1, 100)
      jaw_pose : aa_to_rot6d(pose[:, 3:6])   # (1, 6)
      eyes_pose: zeros → MediaPipe 補完      # (1, 12)
      eyelids  : zeros → MediaPipe 補完      # (1, 2)
    }

FlashAvatarRenderer.render(flame_params)
  → image (1, 3, 512, 512)
```

実装ファイル: `flare/converters/deca_to_flame.py`

#### SMIRK 経路 (SmirkToFlashAvatarAdapter — Phase 2 実装予定)

```
SMIRKExtractor.extract(image)
  → {exp(50), pose(6), cam(3), shape(300), eyelid(2)}

SmirkToFlashAvatarAdapter.convert(smirk_params)
  → {
      expr     : F.pad(exp, (0,50))                # (1, 100)
      jaw_pose : aa_to_rot6d(pose[:, 3:6])         # (1, 6)  ⚠️ pose layout 確認後
      eyes_pose: MediaPipe blendshape から生成      # (1, 12)
      eyelids  : smirk_params["eyelid"]            # (1, 2)  キー rename: eyelid → eyelids
    }

FlashAvatarRenderer.render(flame_params)
  → image (1, 3, 512, 512)
```

実装ファイル (予定): `flare/converters/smirk_to_flash_avatar.py`

### 6.6 注意事項

- **global_rot は condition vector に含まれない**: FlashAvatar は fixed canonical camera を使用する。
  ビデオ合成時に頭部姿勢 (`pose[:, 0:3]`) を使いたい場合は、FlashAvatar 外部での
  画像変換 (アフィン変換等) または FlashAvatar の camera matrix 書き換えが必要。
- **顔以外の領域は背景黒**: 黒背景が不自然な場合は後処理でマスク合成すること。
- **解像度はデフォルト 512×512**: `output_size` パラメータで変更可能だが、
  学習時の解像度と一致させることを推奨 (通常 512×512)。
- **person_id**: FlashAvatar モデルは人物ごとに異なる `.ply` ファイルを使用する。
  `model_path` を `checkpoints/flashavatar/<person_id>/` のように人物 ID で切り替えること。
