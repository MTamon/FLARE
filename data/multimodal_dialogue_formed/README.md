# 対面対話データセットの配置

## 概要

LHG モデル学習用の対面対話動画データセットをこのディレクトリに配置します。
`lhg-extract` コマンドの入力データとして使用されます。

## ディレクトリ構成

```bash
data/multimodal_dialogue_formed/
├── data001/
│   ├── comp.mp4          # 対話参加者 (comparator) の動画
│   ├── host.mp4          # ホスト側の動画
│   └── participant.json  # 話者メタデータ (必須)
├── data002/
│   ├── comp.mp4
│   ├── host.mp4
│   └── participant.json
└── ...
```

## participant.json の形式

各対話ディレクトリに `participant.json` を配置してください:

```json
{
  "comp": "田中太郎",
  "comp_no": 12,
  "host": "鈴木花子",
  "host_no": 7
}
```

| キー | 型 | 説明 |
|------|-----|------|
| `comp` | string | 対話参加者の名前 → npz の `speaker_name` |
| `comp_no` | int | 対話参加者の ID → npz の `speaker_id` |
| `host` | string | ホストの名前 |
| `host_no` | int | ホストの ID |

## 動画の要件

- 1 人の顔が映っている動画（1 動画に 1 人）
- 顔が概ね正面を向いている
- 解像度の制約は特にないが、224x224 以上を推奨

## データの入手

このデータセットは研究用に収集された対面対話データです。
ご自身の研究データを上記の構成に合わせて配置してください。

## 関連ガイド

- [Guide B: LHG 前処理](../../docs/guide_lhg_preprocess.md) — バッチ特徴抽出の全手順
