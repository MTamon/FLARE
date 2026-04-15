# セッション引継ぎメモ (FLARE 整備作業)

本ドキュメントは、長時間セッションで実施した FLARE プロジェクト整備作業の
引継ぎメモ。次のセッションで作業を継続する際の文脈を保つ目的で作成。

## プロジェクト概要

**FLARE** (Facial Landmark Analysis & Rendering Engine): LHG (Listening Head Generation)
研究向けの 3DMM 特徴量抽出 + レンダリング統合ツール。

**ルート構成**:
- **Route A (BFM)**: Deep3DFaceRecon / 3DDFA → PIRender
- **Route B (FLAME)**: DECA / SMIRK → FlashAvatar

## これまでに完了した大きな作業

### Phase A-E (過去のセッション)
- LHG 特徴抽出パイプラインの実装 (`lhg-extract` サブコマンド + YAML configs)
- 回帰テスト整備 (現在 298 passed, 2 skipped)

### 本セッションの主な成果

1. **FLAME デコーダの実装** (`flare/decoders/`)
   - `flame_mesh_renderer.py`: 学習不要のメッシュ描画 (OpenCV + 純 PyTorch 実装)
   - `face_decoder_net.py`: AdaIN ベースのニューラルデコーダ
   - `scripts/train_face_decoder.py`: 学習スクリプト
   - `scripts/demo_visualize.py`: 可視化デモ (mesh / neural 両モード)

2. **タスク別ガイド作成** (`docs/guide_*.md`)
   - Guide A: リアルタイム特徴抽出
   - Guide B: LHG バッチ前処理
   - Guide C: BFM/FLAME ルート切り替え
   - Guide D: FLAME デコーダ学習・可視化
   - Guide E: BFM 可視化 (PIRender)

3. **ディレクトリ構造とパス統一**
   - `checkpoints/{deca,deep3d,smirk,3ddfa,flame,pirender,flashavatar,l2l,face_decoder,batch}/`
   - `data/{multimodal_dialogue_formed,movements,source_images}/`
   - 全ファイルのデフォルトパスを `checkpoints/<モデル>/ファイル名` 形式に統一
   - 旧パス (`checkpoints/deca_model.tar` 等) は**全て**新パス (`checkpoints/deca/deca_model.tar`) に移行済み

4. **YAML 設定ファイル**
   - `configs/realtime_flame.yaml` (DECA + FlashAvatar)
   - `configs/realtime_bfm.yaml` (Deep3D + PIRender, source_image 必須)
   - `configs/lhg_extract_deca.yaml` / `configs/lhg_extract_bfm.yaml`
   - `configs/train_face_decoder.yaml`

5. **チェックポイント取得の整備**
   - `scripts/download_checkpoints.py`: 自動ダウンロード可能なものは `--all`, 手動必要なものは `--list-manual`
     - 自動対応: DECA (`gdown`), 3DDFA (リポジトリから copy)
     - 手動必要: Deep3D, SMIRK, FLAME, PIRender, FlashAvatar, L2L

6. **環境構築ガイド** (`docs/guide_setup.md`) ← **直近の作業**
   - FLARE 本体 + 7 つの外部リポジトリ個別セットアップ
   - 各リポジトリの正確な git URL, Python/CUDA バージョン, 依存パッケージ,
     チェックポイントのダウンロード URL/Google Drive ID を調査済み
   - 情報源: 各リポジトリの README/requirements を実際に確認済み

7. **各 checkpoint ディレクトリの README** (詳細版)
   - 具体的な `git clone`, `gdown`, Google Drive ID, ステップ手順を記載済み

## 重要な情報 (誤りやすいポイント)

### パス体系
- デフォルトパスは全て `./checkpoints/<モデル名>/<ファイル名>` の形式
- `./data/source_images/source_portrait.png` (PIRender 用のソース画像)
- `./data/multimodal_dialogue_formed/data<NNN>/{comp,host}.mp4` (対話データセット)
- `./data/movements/data<NNN>/{comp,host}/{deca,bfm}_<role>_<start>_<end>.npz` (LHG 出力)

### 外部リポジトリの調査済み情報 (正確)
- **DECA**: https://github.com/yfeng95/DECA (YadiraF はリダイレクト), Python 3.7, `gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje`
- **Deep3DFaceRecon**: https://github.com/sicxu/Deep3DFaceRecon_pytorch, Python 3.6 + PyTorch 1.6,
  submodule 無し, nvdiffrast 0.3.0 + insightface が必要, `BFM_model_front.mat` は runtime 自動生成
- **SMIRK**: https://github.com/georgeretsi/smirk, Python 3.9 + PyTorch 2.0.1 + CUDA 11.7,
  pytorch3d の wheel URL あり, `quick_install.sh` が便利, モデル ID `1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE`
- **3DDFA_V2**: https://github.com/cleardusk/3DDFA_V2, Python 3, ONNX は別途ダウンロード必要
  (`1YpO1KfXvJHRmCBkErNa62dHm-CUjsoIk`), BFM データは `configs/bfm_noneck_v3.pkl` に同梱,
  Cython ビルド (`build.sh`) が必要
- **PIRender**: https://github.com/RenYurui/PIRender, チェックポイント Google Drive ID `1-0xOf6g58OmtKtEWJlU3VlnfRqPN9Uq7`,
  Baidu Netdisk 抽出コード `4sy1`
- **FlashAvatar**: https://github.com/MingZhongCodes/FlashAvatar, **対象人物ごとに個別学習必要**,
  `diff-gaussian-rasterization` パッケージが必要
- **L2L**: https://github.com/evonneng/learning2listen, Python 3.6 + CUDA 9.0 (古い),
  4 話者のモデル ID (Conan/Fallon/Stephen/Trevor) あり

### ルート依存の違い
- FlashAvatar: ソース画像**不要** (事前学習済み 3DGS を使用)
- PIRender: ソース画像**必須** (`renderer.setup(source_image=...)`)
- `realtime_flame.yaml` の `source_image: null` / `realtime_bfm.yaml` の `source_image: ./data/source_images/...`

## 環境・テスト実行

### Python 環境
- Windows 11 + Git Bash
- `.venv` は numpy しか入っていない
- `python` コマンドは Windows の Microsoft Store トリガー → `.venv/Scripts/python.exe` または `uv run` を使うこと
- テスト実行コマンド:
  ```bash
  uv run --with pytest --with click --with loguru --with pydantic --with pyyaml \
         --with torch --with numpy --with opencv-python-headless \
         pytest tests/ -q --ignore=tests/test_phase4.py
  ```
  (`test_phase4.py` は headless 環境で OpenCV の GUI 呼び出しが失敗するためスキップ)

### 現在のテスト状態
- 298 passed, 2 skipped (全パス)
- `test_phase4.py::test_run_sets_buffers` は headless 環境での既知の失敗 (本質的な問題ではない)

## 未完了・フォローアップ候補

### 特になし (現時点で完結している)
- 今セッションで user が明示的に依頼した作業は全て完了
- コミットはユーザーの指示があるまで行わない方針

### 将来のタスク候補 (user との相談次第)
- 外部リポジトリの実際の動作確認 (現状は情報として整備済みだが、実際に全ルートを動かしての検証は未実施)
- `scripts/download_checkpoints.py` の実動作テスト (ネット接続が必要)
- ガイド文書のスクリーンショット・図の追加
- 英語版ドキュメントの作成 (現在は日本語のみ)
- CI 設定 (GitHub Actions 等)

## プロジェクト規約・スタイル

- **言語**: ドキュメント・コメント共に日本語ベース (user は日本語話者)
- **コメント**: 最小限。WHY のみ記載
- **絵文字**: 使わない (user の明示的指示がない限り)
- **コミット**: user の明示的指示があるまで作らない
- **破壊的操作**: 事前確認必須
- **README 作成**: user の明示的要求に応じてのみ (無駄な .md を増やさない)

## 主要ファイルの場所一覧

```
FLARE_by_Claude/
├── README.md                      # ルート README (ガイドへのリンクハブ)
├── tool.py                        # CLI エントリポイント
├── config.yaml                    # デフォルト設定 (リアルタイム用)
├── configs/
│   ├── realtime_flame.yaml        # FLAME ルート設定
│   ├── realtime_bfm.yaml          # BFM ルート設定
│   ├── lhg_extract_deca.yaml      # LHG 前処理 (DECA)
│   ├── lhg_extract_bfm.yaml       # LHG 前処理 (BFM)
│   └── train_face_decoder.yaml    # デコーダ学習
├── docs/
│   ├── guide_setup.md             # 環境構築 ← 直近追加
│   ├── guide_realtime.md          # Guide A
│   ├── guide_lhg_preprocess.md    # Guide B
│   ├── guide_route_switching.md   # Guide C
│   ├── guide_flame_decoder.md     # Guide D
│   ├── guide_bfm_visualization.md # Guide E
│   ├── session_handover.md        # ← 本ファイル
│   └── design/                    # 設計ドキュメント
├── flare/
│   ├── cli.py                     # Click CLI (lhg-extract 含む)
│   ├── config.py                  # pydantic v2 設定
│   ├── extractors/                # deca, deep3d, smirk, tdddfa
│   ├── renderers/                 # flashavatar, pirender, headgas
│   ├── decoders/                  # flame_mesh_renderer, face_decoder_net
│   ├── converters/                # deca_to_flame, bfm_to_flame, flame_to_pirender
│   ├── model_interface/           # l2l, vico
│   ├── pipeline/                  # realtime, batch, lhg_batch
│   └── utils/                     # face_detect, interp, rotation_interp 等
├── scripts/
│   ├── train_face_decoder.py
│   ├── demo_visualize.py
│   └── download_checkpoints.py    # 自動ダウンロードスクリプト
├── examples/
│   ├── realtime_extract.py
│   └── visualize_bfm_pirender.py
├── checkpoints/
│   ├── deca/       (+ README.md)  # 各ディレクトリに詳細な入手手順
│   ├── deep3d/     (+ README.md)
│   ├── smirk/      (+ README.md)
│   ├── 3ddfa/      (+ README.md)
│   ├── flame/      (+ README.md)
│   ├── pirender/   (+ README.md)
│   ├── flashavatar/(+ README.md)
│   ├── l2l/        (+ README.md)
│   ├── face_decoder/(+ README.md)
│   └── batch/      (+ README.md)
├── data/
│   ├── multimodal_dialogue_formed/ (+ README.md)
│   ├── movements/                  (出力先)
│   └── source_images/              (+ README.md)
└── tests/                         # 298 passed, 2 skipped
```

## 対話で確立した合意事項・設計判断

設計仕様書には明記されていないが、user との対話を通じて合意・決定した事項。
これらは次セッションで再検討しないこと (user が明示的に覆すまで維持)。

### ディレクトリ構造の設計判断

- **checkpoints の組織化**: 最初は `checkpoints/` 直下にフラットに配置されていたが、
  user の要請により `checkpoints/<モデル名>/<ファイル>` の階層構造へ移行
  - 理由: 各モデルに付随するファイル (BFM 基底, 設定ファイル等) が混在しないように
  - **この構造は維持すること**。旧パスへのフォールバックは実装しない (backwards-compat なし)

- **scripts/ と examples/ の役割分担**:
  - `scripts/`: 学習・ユーティリティ系 (train_face_decoder.py, demo_visualize.py, download_checkpoints.py)
  - `examples/`: すぐ動くサンプル (realtime_extract.py, visualize_bfm_pirender.py)
  - 新しいスクリプトを追加する際もこの区別に従う

- **空ディレクトリの扱い**: user の明示的な要請で
  「ディレクトリ自体は空で OK」という方針。`.gitkeep` + `README.md` (手順書) の
  セットで各ディレクトリを配置している。README.md は手順書扱いで、チェックポイント
  配置後もそのまま残しておく。

### パス規約 (絶対維持)

- 全てのデフォルトパスは `./checkpoints/<モデル>/<ファイル>` 形式
- ソース画像: `./data/source_images/source_portrait.png`
  (旧 `./data/source_portrait.png` は完全廃止)
- LHG 抽出出力: `./data/movements/data<NNN>/{comp,host}/{deca,bfm}_<role>_<start>_<end>.npz`
- 対話データセット: `./data/multimodal_dialogue_formed/data<NNN>/{comp.mp4,host.mp4,participant.json}`

### PIRender vs FlashAvatar の非対称性 (重要)

調査を通じて判明した非自明な差異。**頻繁に混乱を招くため、次セッションでも意識すること**:

- **PIRender**: `renderer.setup(source_image=tensor)` が**必須**
  - 対象人物の正面顔写真を 1 枚要求
  - `configs/realtime_bfm.yaml` では `source_image: ./data/source_images/source_portrait.png`
  - 画像品質が出力品質に直結する (正面性・照明・解像度 256x256 以上)

- **FlashAvatar**: source_image **不要**
  - 対象人物ごとに 3DGS を事前学習済み (`point_cloud/iteration_30000/point_cloud.ply`)
  - `configs/realtime_flame.yaml` では `source_image: null` (明示的に null)
  - 各人物ごとに約 30 分 (RTX 3090) の個別学習が必要

これは `guide_route_switching.md` と `guide_bfm_visualization.md` で明記済み。

### FLAME モデル (generic_model.pkl) の要否

調査を通じて判明:

- **DECA Extractor として使う分には `generic_model.pkl` は不要**
  - `deca_model.tar` が FLAME の基底データを内包しているため
- **メッシュ可視化 (`demo_visualize.py --mode mesh`) を使う場合のみ必要**
- これは user との対話で確認した重要な区別。`checkpoints/flame/README.md` と
  `checkpoints/deca/README.md` の両方で明示している

### FaceDecoderNet の設計判断

user の「研究して正しく実装してほしい」という要請に対する設計:

- **アーキテクチャ決定**: ResNet-18 encoder + AdaIN decoder
  - 代替案 (pytorch3d 依存, Gaussian Splatting) は重いため却下
  - 選択理由: 外部依存ゼロで完結する軽量デコーダ
- **損失関数**: L1 + Perceptual Loss (VGG-16)
- **位置づけ**: プロトタイプ・検証用。本番品質は FlashAvatar 使用を推奨
  (guide_flame_decoder.md の比較表で明記)
- **per-person 学習**: 対象人物の動画から自動的にデータ準備 + 学習

### 自動ダウンロードの方針

- **自動ダウンロード可能**: DECA (gdown), 3DDFA (リポジトリから copy)
- **手動必要**: Deep3D, SMIRK, FLAME, PIRender, FlashAvatar, L2L
  - 理由: 学術ライセンス, 複雑なセットアップ, 対象人物ごとの学習要件
- `scripts/download_checkpoints.py --list-manual` で手動項目を一覧表示
- この分類は外部要因 (ライセンス等) で変更される可能性があるため、
  次セッションで user から「これも自動化して」と言われたら再検討可

### 外部リポジトリの Python バージョン衝突問題

調査で判明した事実と、それに対する合意:

- DECA (3.7), Deep3D (3.6), SMIRK (3.9), L2L (3.6), FLARE (3.11) と全て異なる
- **合意事項**: FLARE は Python 3.11 で動作。外部リポジトリの**コードを直接 import する
  わけではなく、チェックポイントファイルのみを利用する**方針
  - 例外: Deep3D, SMIRK, 3DDFA, L2L は `<model>_dir` パラメータ経由で `sys.path` に追加し、
    内部モジュールをインポートする場合がある (各 Extractor 実装を参照)
- この方針のおかげで、ユーザーは外部リポジトリの conda 環境を毎回切り替える必要はない

### ドキュメント方針

- **言語**: 日本語統一 (user が日本語話者のため)
- **逆引き設計**: `README.md` の「やりたいこと → ガイド」表は機能別ではなく、
  ユーザーの**意図**から逆引きする形式にした (user の明示的要請)
- **ガイドの独立性**: Guide A-E は各々独立した手順書として読めるようにした
  (ある程度の内容重複を許容)
- **絵文字**: 一切使用しない (user の durable preference)

### Windows 環境の落とし穴 (開発時の注意)

次セッションでコードを動かす際に役立つ実務的な情報:

- `python` コマンドは Windows Store トリガーを起動するため使用不可
  - 代わりに `.venv/Scripts/python.exe` または `uv run` を使う
- `.venv` には numpy しか入っていない。テスト実行時は以下を使用:
  ```bash
  uv run --with pytest --with click --with loguru --with pydantic --with pyyaml \
         --with torch --with numpy --with opencv-python-headless \
         pytest tests/ -q --ignore=tests/test_phase4.py
  ```
- `test_phase4.py::test_run_sets_buffers` は headless 環境で OpenCV GUI 呼び出しが
  失敗するため**スキップでよい**。本質的な問題ではない。
- パスはフォワードスラッシュを使用 (bash 環境)

### 作業中の変更を迷わないための原則

対話で暗黙的に合意した作業スタイル:

- **コミットは user の明示指示待ち**。勝手に git commit しない
- **破壊的操作**: 事前確認必須 (ファイル削除, force push 等)
- **後方互換**: 今回のリネームでは backwards-compat 層は作らない方針で合意
  (旧パス対応コード・警告メッセージ等は書かない)
- **README.md の乱造禁止**: 手順書としての README のみ許可。進捗ログや
  分析結果のような .md は user の明示要請なしには作らない
- **コメント最小主義**: コードにコメントを書くのは「非自明な WHY」のみ

## 次のセッションで最初にやるべきこと

1. 本ファイル (`docs/session_handover.md`) を読む
2. `README.md` で現在のガイド構成を把握
3. user から新しい指示を受けたら、上記ディレクトリ構造を前提に作業開始
4. パスを追加・変更する場合は既存の `./checkpoints/<モデル>/` 体系を維持
5. 外部リポジトリの情報が必要になったら `docs/guide_setup.md` を参照
   (既に調査済みの内容は再調査不要)
