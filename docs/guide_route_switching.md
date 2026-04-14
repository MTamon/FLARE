# Guide C: BFM 系 / FLAME 系のルート切り替え

FLARE は 2 つのルート (Route A: BFM, Route B: FLAME) をサポートしています。本ガイドでは、それぞれの特徴と切り替え方法、ルート固有の注意事項を説明します。

## ルート一覧

| | Route A (BFM) | Route B (FLAME) |
|---|---|---|
| **3DMM 基底** | Basel Face Model (BFM) | FLAME |
| **Extractor** | Deep3DFaceRecon / 3DDFA | DECA / SMIRK |
| **Renderer** | PIRender (NeRF) | FlashAvatar (3DGS) |
| **表情次元** | 64 次元 | 50 次元 (DECA) / 100 次元 (FlashAvatar) |
| **形状次元** | id 80 次元 | shape 100 次元 |
| **顎パラメータ** | なし (pose に含む) | あり (jaw_pose 3D) |
| **顔サイズ** | なし | あり (cam[0]) |
| **出力プレフィックス** | `bfm` | `deca` / `smirk` |

## リアルタイムパイプラインでの切り替え

### FLAME ルート

```yaml
# configs/realtime_flame.yaml
extractor:
  type: deca                              # ← ここを変更
  model_path: ./checkpoints/deca/deca_model.tar

renderer:
  type: flash_avatar                      # ← ここを変更
  model_path: ./checkpoints/flashavatar/person01/

pipeline:
  converter_chain:
    - type: deca_to_flame                 # ← DECA→FlashAvatar 変換
```

### BFM ルート

```yaml
# configs/realtime_bfm.yaml
extractor:
  type: deep3d                            # ← ここを変更
  model_path: ./checkpoints/deep3d/deep3d_epoch20.pth

renderer:
  type: pirender                          # ← ここを変更
  model_path: ./checkpoints/pirender/epoch_00190_iteration_000400000_checkpoint.pt
  source_image: ./data/source_images/source_portrait.png  # ← PIRender 必須

pipeline:
  converter_chain: []                     # ← 変換不要
```

**PIRender 固有の注意**: PIRender は `setup()` 時にソース画像 (対象人物の正面顔写真) が必須です。この画像の外見を保持しつつ、3DMM パラメータで表情・姿勢を制御します。`source_image` を設定ファイルで指定するか、Python コードで `renderer.setup(source_image=tensor)` を呼びます。

**FlashAvatar 固有の注意**: FlashAvatar は対象人物ごとの学習済み 3D Gaussian モデルが必要です。`model_path` には学習済みモデルディレクトリ (内部に `point_cloud/iteration_30000/point_cloud.ply` を含む) を指定します。ソース画像は不要です。

## LHG 前処理 (lhg-extract) での切り替え

### FLAME ルート

```bash
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --config configs/lhg_extract_deca.yaml
```

または CLI フラグで直接指定:

```bash
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --extractor deca \
    --model-path ./checkpoints/deca/deca_model.tar
```

### BFM ルート

```bash
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --config configs/lhg_extract_bfm.yaml
```

または:

```bash
python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --extractor deep3d \
    --model-path ./checkpoints/deep3d/deep3d_epoch20.pth
```

### 出力の違い

DECA (FLAME) で抽出した場合:
```
movements/data001/comp/deca_comp_00000_04499.npz
  ├── expression: (T, 50)   ← FLAME 表情 50 次元
  ├── shape:      (100,)    ← FLAME 形状 100 次元
  ├── jaw_pose:   (T, 3)    ← DECA 固有: 顎の回転
  └── face_size:  (T,)      ← DECA 固有: 顔サイズ (cam[0])
```

Deep3D (BFM) で抽出した場合:
```
movements/data001/comp/bfm_comp_00000_04499.npz
  ├── expression: (T, 64)   ← BFM 表情 64 次元
  ├── shape:      (80,)     ← BFM id 80 次元
  ├── (jaw_pose なし)
  └── (face_size なし)
```

## パラメータ変換器 (Converter)

ルート間のパラメータ変換が必要な場合、以下のアダプタを使用できます:

| 変換 | アダプタ | 要点 |
|------|---------|------|
| DECA → FlashAvatar | `DECAToFlameAdapter` | exp 50D → 100D ゼロパディング (同一 PCA 空間) |
| BFM → FLAME | `BFMToFlameAdapter` | 線形変換行列 or ゼロパディング |
| FLAME → PIRender | `FlameToPIRenderAdapter` | expr[:64] スライス + rotation_6d→axis-angle |

### 変換チェーンの使用例

```python
from flare.converters.registry import AdapterRegistry
from flare.converters.deca_to_flame import DECAToFlameAdapter

registry = AdapterRegistry()
registry.register(DECAToFlameAdapter())

adapter = registry.get("deca", "flash_avatar")
flash_params = adapter.convert(deca_params)
```

YAML 設定で変換チェーンを指定:

```yaml
pipeline:
  converter_chain:
    - type: deca_to_flame    # DECA 出力 → FlashAvatar 入力
```

## Extractor の選択指針

| Extractor | 強み | 弱み | 推奨用途 |
|-----------|------|------|---------|
| **DECA** | 安定、FLAME 標準、詳細パラメータ豊富 | やや重い | FLAME ルート標準 |
| **SMIRK** | 非対称表情に強い (2024) | shape 300D と大きい | 表情精度重視 |
| **Deep3DFaceRecon** | BFM 標準、軽量 | 詳細パラメータ少ない | BFM ルート標準 |
| **3DDFA** | 最軽量 (ONNX) | 精度は控えめ | リアルタイム速度重視 |

## ルート混在に関する注意

- 同一データセットに対して DECA と Deep3D の両方で抽出を行うことは可能です。出力プレフィックスが `deca_` / `bfm_` と異なるため、同一ディレクトリに共存します。
- ただし、下流の LHG モデルに入力する際はルートを統一してください。FLAME 表情 50 次元と BFM 表情 64 次元は直接互換性がありません。
- ルート変換 (BFM→FLAME 等) は `BFMToFlameAdapter` で可能ですが、情報の欠損が生じるため、可能な限り同一ルートで統一することを推奨します。
