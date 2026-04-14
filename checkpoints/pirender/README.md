# PIRender チェックポイントの準備

## 必要ファイル

| ファイル | 説明 |
|----------|------|
| `epoch_00190_iteration_000400000_checkpoint.pt` | PIRender 学習済みモデル |

## 入手手順

### Step 1: リポジトリのクローン

```bash
git clone https://github.com/RenYurui/PIRender.git
cd PIRender
pip install -r requirements.txt
```

### Step 2: チェックポイントのダウンロード

**Google Drive からダウンロード (推奨)**:

```bash
pip install gdown
gdown 1-0xOf6g58OmtKtEWJlU3VlnfRqPN9Uq7 -O pirender_pretrained.zip
unzip pirender_pretrained.zip

# チェックポイントを FLARE にコピー
cp result/face/epoch_00190_iteration_000400000_checkpoint.pt \
   ../FLARE_by_Claude/checkpoints/pirender/
```

**Baidu Netdisk からダウンロード (中国からのアクセス)**:
- URL: https://pan.baidu.com/s/18B3xfKMXnm4tOqlFSB8ntg
- 抽出コード: `4sy1`

**PIRender リポジトリのスクリプトを使用**:
```bash
cd PIRender
bash scripts/download_weights.sh
```

## 配置後の構成

```
checkpoints/pirender/
└── epoch_00190_iteration_000400000_checkpoint.pt
```

## PIRender リポジトリのセットアップ

PIRender はソースコードのインポートが必要なため、リポジトリのクローンが必須です:

```bash
# プロジェクトルートにクローン (推奨配置)
cd FLARE_by_Claude/
git clone https://github.com/RenYurui/PIRender.git
cd PIRender
pip install -r requirements.txt
cd ..
```

実行時に `pirender_dir` パラメータでリポジトリパスを指定してください:

```yaml
renderer:
  type: pirender
  model_path: ./checkpoints/pirender/epoch_00190_iteration_000400000_checkpoint.pt
  source_image: ./data/source_images/source_portrait.png
  # pirender_dir: ./PIRender
```

## 注意事項

- PIRender は BFM ルート (Route A) のレンダラです
- **ソース画像 (対象人物の正面顔写真) が別途必要です** → [`data/source_images/README.md`](../../data/source_images/README.md)
- NeRF ベースのフローフィールド推定で顔画像を生成します
- ~100 FPS @ 256x256 (GPU 使用時)

## 関連ガイド

- [環境構築ガイド](../../docs/guide_setup.md#25-pirender) — セットアップの詳細
- [Guide E: BFM 可視化 (PIRender)](../../docs/guide_bfm_visualization.md) — PIRender での可視化手順
- [Guide C: ルート切り替え](../../docs/guide_route_switching.md) — BFM/FLAME の選択
