# LHG 頭部特徴量抽出パイプライン設計

## 1. 目的

旧バージョン（MediaPipe + `extract_angle_cent.py`）の頭部特徴量抽出パイプラインを
FLARE ベースに置換する。DECA（FLAME 系）または Deep3DFaceRecon / 3DDFA（BFM 系）から
per-frame に頭部姿勢・位置・表情係数を抽出し、下流 `databuild_nx8.py` 互換の npz
スキーマで保存する。

## 2. スコープ

| 項目 | 内容 |
|---|---|
| 入力 | `multimodal_dialogue_formed/dataXXX/{comp,host}.{mp4,wav}` + `participant.json` |
| 出力 | `movements/dataXXX/{comp,host}/{deca|bfm}_{role}_{SSSSS}_{EEEEE}.npz` |
| 置換対象 | `extract_angle_cent.py`（`.head` ファイル読込版） |
| 非スコープ | 音声のコピー（下流で別途扱う）、WAV 前処理 |

## 3. 入出力仕様

### 3.1 入力ディレクトリ構造

```
multimodal_dialogue_formed/
└── dataXXX/                   # XXX はゼロ埋め3桁、連番保証なし
    ├── comp.mp4
    ├── comp.wav
    ├── host.mp4
    ├── host.wav
    └── participant.json       # {"host": str, "comp": str, "host_no": int, "comp_no": int}
```

### 3.2 出力ディレクトリ構造

```
movements/
└── dataXXX/
    ├── comp/
    │   ├── deca_comp_00000_17394.npz
    │   ├── deca_comp_17520_23100.npz
    │   └── ...
    ├── host/
    │   └── deca_host_00000_23100.npz
    └── participant.json       # 元ファイルを単純コピー
```

ファイル名テンプレート: `{prefix}_{role}_{start:05d}_{end:05d}.npz`

- `prefix`: `deca`（FLAME 系）または `bfm`（Deep3DFaceRecon / 3DDFA）
- `role`: `comp` または `host`
- `start`, `end`: 元動画内のフレーム番号（`end` は inclusive の末尾インデックス）

### 3.3 npz スキーマ

下流 `databuild_nx8.py` が参照する必須キーと、学習時に利用可能な追加キーを定義する。

| キー | dtype | shape | 必須 | 意味 |
|---|---|---|---|---|
| `section` | int32 | (2,) | ✓ | `[start_frame, end_frame]` |
| `speaker_id` | int64 | () | ✓ | participant.json 由来 (`host_no` / `comp_no`) |
| `speaker_name` | `<U*` | () | ✓ | participant.json 由来（`host` / `comp` 文字列値） |
| `fps` | float32 | () | ✓ | 元動画の FPS |
| `angle` | float32 | (T, 3) | ✓ | 正規化済み軸角回転 |
| `centroid` | float32 | (T, 3) | ✓ | 正規化済み位置（DECA: `cam`, BFM: `trans`） |
| `expression` | float32 | (T, D) | ✓ | 正規化済み表情係数（DECA: 50, BFM: 64） |
| `angle_mean` / `angle_std` | float32 | (3,) | ✓ | 対話単位正規化統計量 |
| `centroid_mean` / `centroid_std` | float32 | (3,) | ✓ | 〃 |
| `expression_mean` / `expression_std` | float32 | (D,) | ✓ | 〃 |
| `jaw_pose` | float32 | (T, 3) | DECA 時 | 非正規化軸角 |
| `face_size` | float32 | (T,) | DECA 時 | `cam[0]`（scale） |
| `shape` | float32 | (S,) | ✓ | シーケンス代表 shape（中央値, DECA:100 / BFM:80） |
| `extractor_type` | `<U*` | () | ✓ | `"deca"` / `"deep3d"` / `"smirk"` / `"tdddfa"` |
| `param_version` | `<U*` | () | ✓ | `"flare-v2.2"` |

**設計判断:**

- `shoulder_centroid` は FLARE が提供しないため廃止
- 旧 `rel_` プレフィックス（相対座標）は廃止
- `shape` はシーケンス内中央値 1本のみ（話者不変の仮定）
- `jaw_pose` は非正規化で保存（表情と相関が強く、正規化するとモデルが壊れる可能性）

## 4. 処理フロー

```
main(args):
    configs = load_yaml(args.config)
    dataXXX_dirs = enumerate_dataXXX(args.path)

    with multiprocessing.Pool(args.num_workers) as pool:
        pool.map(process_dataXXX, dataXXX_dirs)

process_dataXXX(dataXXX_dir):
    participant = load_json(dataXXX_dir / "participant.json")
    for role in ("comp", "host"):
        if output_exists(role) and not redo:
            continue
        raw_frames = per_frame_extract(video=dataXXX_dir / f"{role}.mp4")
        # raw_frames: list[dict | None] of length T
        filled = fill_short_gaps(raw_frames, fps, max_gap_sec)
        sequences = split_on_long_gaps(filled, min_seq_len)
        for seq in sequences:
            stats = compute_stats(seq)
            normed = normalize(seq, stats)
            save_npz(normed, stats, role, participant, start, end)
    copy(dataXXX_dir / "participant.json", out_dir / "participant.json")
```

### 4.1 `per_frame_extract`

```
for frame_idx, frame in enumerate(cv2.VideoCapture(path)):
    bbox = face_detector.detect(frame)
    if bbox is None:
        yield None
        continue
    cropped = face_detector.crop_and_align(frame, bbox, size=extractor.input_size)
    image = to_tensor(cropped).to(device)         # (1, 3, H, W) float[0,1]
    params = extractor.extract(image)             # dict[str, Tensor]
    yield to_cpu_numpy(params)
```

### 4.2 `fill_short_gaps`

詳細は `interpolation.md` および `rotation_interpolation.md` 参照。

- 線形空間特徴量（expression, centroid, jaw_pose, face_size, shape）:
  `interp_linear(values, mask, order=cfg.linear_order)`
- 回転特徴量（angle = `pose[:, :3]`）:
  `interp_rotation(rotations, mask, order=cfg.rotation_order)`
- 欠損長 ≥ `max_gap_sec * fps` のギャップは補間せず、後段で分割境界となる

### 4.3 `split_on_long_gaps`

`filled` の中で「補間されなかった None 連続領域」を検出し、
その前後で独立シーケンスに分割する。各シーケンス長が `min_seq_len` 未満なら破棄。

### 4.4 `compute_stats`

```
mean = seq.mean(axis=0)
std = seq.std(axis=0).clip(min=1e-6)
```

対話単位（シーケンスごとではなく 1 動画ファイル全体）で算出することで、
同一シーケンス内の相対的な動きのみが学習される。

### 4.5 `save_npz`

`np.savez` を用いて §3.3 のスキーマで保存。`speaker_name` は `participant.json` の
`host` または `comp` キーの**文字列値**（話者名）、`speaker_id` は `host_no` または
`comp_no` の**整数値**を使う。

## 5. 並列化戦略

- 最外ループ（dataXXX 単位）を `multiprocessing.Pool(num_workers)` で並列化
- 各プロセスが独立に Extractor インスタンスを遅延構築
- GPU は round-robin: `cuda:{gpu_ids[worker_idx % len(gpu_ids)]}`
- `torch.cuda.set_device()` をプロセス起動時に呼び、CUDA コンテキストを分離
- `num_workers` のデフォルト: `min(os.cpu_count(), len(gpu_ids) * 2)`
  - 1 GPU あたり最大 2 プロセスで十分（これ以上は GPU メモリ競合が増加）

## 6. エラー処理

| 事象 | 方針 |
|---|---|
| 動画オープン失敗 | WARNING ログ、該当 dataXXX スキップ |
| 顔検出失敗 | 該当フレームを `None` に、ギャップ処理へ |
| Extractor 推論失敗 | 該当フレームを `None` に、ギャップ処理へ |
| `participant.json` 欠損 | ERROR ログ、該当 dataXXX スキップ |
| 全フレーム `None` | WARNING、ファイル出力なし |
| 既存出力あり | `--redo` 無ければスキップ、あれば上書き |

## 7. CLI

```
uv run python tool.py lhg-extract \
    --path ./data/multimodal_dialogue_formed \
    --output ./data/movements \
    --extractor {deca,deep3d,smirk,3ddfa} \
    [--config configs/lhg_extract_deca.yaml] \
    [--num-workers N] \
    [--gpus 0,1] \
    [--redo] \
    [--dry-run]
```

YAML（`configs/lhg_extract_deca.yaml` 等）で以下を制御:

```yaml
pipeline:
  name: lhg_extract_deca
  fps: 30
extractor:
  type: deca
  model_path: ./checkpoints/deca/deca_model.tar
  input_size: 224
lhg_extract:
  interpolation:
    linear_order: linear        # linear | pchip
    rotation_order: slerp       # slerp | linear
    max_gap_sec: 0.4
  sequence:
    min_length: 100
  normalization:
    scope: sequence             # 対話単位
  output:
    prefix: deca                # deca | bfm
    shape_aggregation: median
```

### 7.1 `--dry-run`

実際の動画処理を行わず、設定のロード・dataXXX 列挙・participant.json 読取までを
検証して終了する。モデルファイルが無くても動作確認が可能。

## 8. 下流互換性の根拠

`databuild_nx8.py` のソースを grep した結果、実際に参照されるキーは以下のみ:

```
L606-608:  sections_self = [m["section"] for m in self_npz]
L742-745:  host_speaker_id = host_motion_data[0]["speaker_id"]
L765-776:  motion["section"][0], motion["section"][-1]
L491-494:  _file_path.startswith(start_phrase["host"])  # "host" / "comp" プレフィックス
```

したがって以下を保証すれば下流互換:
1. ファイル名が `{prefix}_{role}_...npz` で `role ∈ {"host", "comp"}`
2. npz 内に `section` と `speaker_id` キーが存在
3. `speaker_id` は `int` にキャスト可能な値

それ以外の特徴量キー（`angle`, `centroid`, `expression` 等）は
学習コード側で選択的に読まれるため、スキーマ拡張は自由。

## 9. テスト方針

### 9.1 単体テスト

- `test_interp.py`: `interp_linear`, `fill_gaps`, `split_on_long_gaps` の各関数
- `test_rotation_interp.py`: `axis_angle_to_quaternion`, `slerp`, `interp_rotation`
- `test_lhg_batch.py`: `compute_stats`, `normalize`, npz スキーマ検証

### 9.2 統合テスト

- 合成動画 2本（各 5 秒、30fps、固定顔画像）で end-to-end 実行
- 出力 npz の全必須キー存在確認
- `section` 値が元フレーム番号と一致することを確認

### 9.3 リグレッション

既存 Phase 1〜5 の 254 テストが全て通過し続けることを確認する。

## 10. 実装ファイル一覧

| ファイル | 役割 |
|---|---|
| `flare/pipeline/lhg_batch.py` | `LHGBatchPipeline` クラス本体 |
| `flare/utils/interp.py` | 線形空間補間ユーティリティ |
| `flare/utils/rotation_interp.py` | SO(3) 補間ユーティリティ |
| `flare/config.py` (更新) | `LHGExtractConfig` Pydantic モデル追加 |
| `flare/cli.py` (更新) | `lhg-extract` サブコマンド追加 |
| `configs/lhg_extract_deca.yaml` | DECA ルート設定例 |
| `configs/lhg_extract_bfm.yaml` | BFM ルート設定例 |
| `tests/test_interp.py` | 線形補間の単体テスト |
| `tests/test_rotation_interp.py` | 回転補間の単体テスト |
| `tests/test_lhg_batch.py` | パイプライン統合テスト |
