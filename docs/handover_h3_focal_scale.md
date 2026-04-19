# H3 申し送り: FlashAvatar focal_scale ヒューリスティクス校正

本ドキュメントは Phase B 統合監査 (H1/H2/M1/M2 修正完了済み) で **保留** とした
**H3 = `flame_converter.py:weak_perspective_to_full()` の `focal_scale=5.0`
ヒューリスティクス** の取り扱いについて、別セッションでの議論用にコンテキストを
集約したもの。

## 1. 問題の所在

FlashAvatar への DECA/SMIRK パラメータ流し込みで使う camera 内部パラメータ
(focal length) が、フォークで新規追加された heuristic 一行に依存している。

### 該当コード

ファイル: `third_party/FlashAvatar/utils/flame_converter.py:109` 付近
(MTamon/FlashAvatar@release/cuda128-fixed のフォーク独自追加)

```python
def weak_perspective_to_full(scale, tx, ty, img_w, img_h, focal_scale=5.0):
    ...
    focal = focal_scale * img_w * scale   # ← H3 の対象行
    ...
```

docstring には `focal_scale: heuristic multiplier for focal length estimation`
としか書かれておらず、`5.0` の出典・校正根拠は **無い**。

### 公式 FlashAvatar との差分

`MTamon/FlashAvatar` の `main` (公式 fork 直後) と `release/cuda128-fixed`
の比較結果:

| 項目 | 公式 main | release/cuda128-fixed |
| --- | --- | --- |
| `utils/flame_converter.py` | 存在しない | **新規追加** (フォーク独自) |
| `requirements_128.txt` / `install_128.sh` | なし | 追加 |
| `docs/conversion_rationale.md` 等 | なし | 追加 |
| `scene/__init__.py`, `train.py`, `gaussian_renderer/__init__.py` | — | **camera 関連の改変なし** |

公式 FlashAvatar は metrical-tracker で得た `.frame` ファイルから直接 camera
内部パラメータを取るため、`focal_scale` のような heuristic 乗数自体が存在しない。
本フォークでは DECA/SMIRK の弱透視 `cam = [s, tx, ty]` から full perspective
camera を「式で導出」するため、heuristic に頼っている。

## 2. 影響

- `focal_scale=5.0` のまま学習すると、3DGS が学習した暗黙の焦点距離と
  inference 時に推定する `focal` がズレ、レンダリング結果に
  - 顔が遠近で潰れる/広がる
  - jaw_pose の動きが過大/過小に見える
  といった系統的ズレが乗る可能性がある。
- ただし、学習・推論を **同じ `focal_scale=5.0`** で揃えていれば 3DGS が
  そのバイアスを吸収するため、最終的な absolute 値の正確性より
  「学習・推論の一貫性」のほうが重要。
- H1/H2 の固定クロップ + 1.25× 再クロップで `cam = [s, tx, ty]` の per-frame
  ジッタは大幅に低減できているはずなので、`focal` 自体の安定性は別問題。

## 3. 短期 (本セッションで完了済み) の対応

- **H1**: `crop_and_align(margin_scale=1.25)` を全 DECA/SMIRK 呼び出し箇所に適用
- **H2**: `extract_{deca,smirk}_frames.py` を 2-pass 化
  (Phase 0: median bbox + `fixed_margin=2.0` で `crop_region.json` 生成、
   Phase 1: imgs/ は固定領域、DECA/SMIRK 入力は per-frame 1.25×)
- **M1**: `train_face_decoder.py` で `extractor_type` dispatch 実装
  (DECA/SMIRK→56D, BFM→67D)
- **M2**: `generate_masks_mediapipe.py` の mouth mask を `hull_inner` のみに

→ これにより `cam` パラメータのフレーム間ジッタは抑えられる。
   focal 校正は **据え置き** (focal_scale=5.0)。

## 4. 中期 (次セッションで議論したい H3 修正方針)

### 選択肢

#### 案 1: 公式 FlashAvatar に倣う

- metrical-tracker で `.frame` から直接 intrinsics を読むパスに切り替える
- `flame_converter.py` の heuristic 自体を廃止
- **メリット**: 論文準拠 / 校正不要
- **デメリット**: metrical-tracker のセットアップ・出力フォーマット変換が
  必要。LHG パイプライン (`lhg-extract`) のリファクタが大きい。

#### 案 2: focal_scale を実測校正

- 学習動画から `focal_scale` を grid-search で fit
- 評価指標: 学習後 3DGS の再構成 PSNR、もしくは保留中の photo loss 値
- 候補値: 1.0, 2.0, 3.0, 5.0, 7.5, 10.0 程度
- **メリット**: 既存パイプラインへの影響が小さい (定数を 1 個変えるだけ)
- **デメリット**: 学習を複数回走らせる必要があり時間コストが大きい
  (FlashAvatar 30k iter @ RTX3090 で ~30 分 × N 件)

#### 案 3: DECA cam → camera intrinsics の正しい式

- DECA の弱透視 `[s, tx, ty]` から full perspective camera への
  正規化変換は別の式があり、論文一般の慣習では概ね
  `focal = scale × img_h / (2 × s)` 程度
- ただし FlashAvatar の `Scene_mica` (`scene/__init__.py`) が想定する
  入力範囲を別途確認しないと数値が合わない可能性あり
- **メリット**: heuristic からの脱却
- **デメリット**: `Scene_mica.cameras` 周りの読解が必要

### 推奨

短期は据え置き、中期は **案 2 (実測校正)** を別タスクとして起こすのが現実的。
案 1 は metrical-tracker への置き換えが大規模リファクタになるため、
LHG ルートとの統合方針を再設計する別議題と合わせて進めたい。

## 5. 議論のための事前確認事項

別セッションで H3 修正に着手する前に、以下を確認しておくと議論がスムーズ:

1. **現状の学習結果が既に satisfactory か?**
   - 既存の FlashAvatar 学習済みチェックポイント (どれか 1 体) について
     レンダリング結果が「許容範囲」なら案 2 のコストを払う価値が薄い
2. **校正対象の動画が複数個体か単一個体か?**
   - 単一なら定数 1 個の手動 fit で十分
   - 複数なら焦点距離が **動画ごとに変わる** ので heuristic 自体の妥当性が再議論
3. **3DGS 出力の使用先**
   - 単に学習済み avatar の inference に使うだけなら一貫性のみで十分
   - 別カメラへの projection / 多視点合成を想定するなら絶対値の正しさが必要

## 6. 関連コミット / ブランチ

- 修正完了 (H1/H2/M1/M2): ブランチ `claude/flare-smirk-integration-phase2-bfEaO`
- フォーク本体: `third_party/FlashAvatar` =
  `MTamon/FlashAvatar@release/cuda128-fixed`

## 7. 参考: H3 と同時に保留した M3 の結論

監査で M3 として挙がっていた `flare/converters/deca_to_flame.py` の
`identity_6d.repeat(B, 1).repeat(1, 2)` は **コードとして正しく**、
修正不要と確定済み。`(B, 6) → (B, 12)` で各行が `[identity, identity]`
となり、左右両眼が identity rotation になる意図通りの挙動。

---

最終更新: 2026-04-19
作成セッション: `claude/flare-smirk-integration-phase2-bfEaO` ブランチ
