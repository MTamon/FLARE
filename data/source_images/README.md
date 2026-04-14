# ソース画像の準備

## 概要

PIRender (BFM ルート) でのレンダリングには、対象人物の **正面顔写真** が 1 枚必要です。
FlashAvatar (FLAME ルート) ではソース画像は不要です。

## 要件

| 条件 | 詳細 |
|------|------|
| 顔の向き | 正面 (yaw / pitch が小さい) |
| 照明 | 均一 (強い影がない) |
| 表情 | 自然 (ニュートラル) |
| 解像度 | 256x256 以上 |
| 形式 | PNG または JPEG |

## 準備方法

### 方法 1: 動画から手動選定

対話動画の中から条件を満たすフレームを選び、保存します。

```python
import cv2

cap = cv2.VideoCapture("data/multimodal_dialogue_formed/data001/comp.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 動画中央付近のフレームを候補にする
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ret, frame = cap.read()
cap.release()

cv2.imwrite("data/source_images/data001_comp.png", frame)
```

### 方法 2: 写真撮影

正面から自然な照明で撮影し、顔部分をクロッピングして 256x256 以上にリサイズします。

## 配置例

```bash
data/source_images/
├── data001_comp.png    # data001 の comp 話者用
├── data001_host.png    # data001 の host 話者用
└── source_portrait.png # 汎用テスト用
```

## 設定での参照

```yaml
# configs/realtime_bfm.yaml
renderer:
  source_image: ./data/source_images/source_portrait.png
```

## 関連ガイド

- [Guide E: BFM 可視化 (PIRender)](../../docs/guide_bfm_visualization.md) — ソース画像の詳細な選定基準
- [Guide C: ルート切り替え](../../docs/guide_route_switching.md) — PIRender と FlashAvatar の違い
