# Face Decoder (ニューラルデコーダ) の準備

## 概要

Face Decoder は **対象人物ごとに個別学習が必要** な AdaIN ベースのニューラルデコーダです。
`scripts/train_face_decoder.py` で学習します。

## 学習方法

対象人物の動画から自動的にデータ準備・学習を行います:

```bash
python scripts/train_face_decoder.py \
    --video ./data/multimodal_dialogue_formed/data001/comp.mp4 \
    --deca-path ./checkpoints/deca/deca_model.tar \
    --output-dir ./checkpoints/face_decoder/data001_comp/ \
    --device cuda:0
```

## 学習後の構成

```bash
checkpoints/face_decoder/
├── data001_comp/
│   ├── face_decoder.pth         # 推論に使用するモデル
│   ├── source_image.png         # ソース画像 (推論時に必要)
│   ├── train_config.json        # 学習設定の記録
│   ├── checkpoint_epoch0100.pth # エポックごとのチェックポイント
│   └── samples/                 # 学習進捗サンプル画像
│       ├── epoch_0001.png
│       └── ...
├── data001_host/
│   └── ...
└── ...
```

## 必要な前提

- DECA チェックポイント (`checkpoints/deca/deca_model.tar`) — 学習データ準備時に使用
- 対象人物の動画ファイル
- CUDA 対応 GPU

## 関連ガイド

- [Guide D: FLAME デコーダ学習と可視化](../../docs/guide_flame_decoder.md) — 学習・可視化の詳細手順
