"""FlashAvatar 学習パイプライン設定モジュール。

configs/train_flashavatar.yaml の内容を pydantic v2 で検証・ロードする。
scripts/train_flashavatar.py が使用し、テストからも直接インポートできる。

Example:
    YAML ファイルから設定を読み込む::

        from flare.training import TrainFlashAvatarConfig

        cfg = TrainFlashAvatarConfig.from_yaml("configs/train_flashavatar.yaml")
        print(cfg.pipeline.extractor)   # "deca"
        print(cfg.flashavatar.iterations)  # 30000
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class PipelineTrainSettings(BaseModel):
    """学習パイプライン基本設定。"""

    extractor: Literal["deca", "smirk"] = Field(
        default="deca",
        description="特徴量抽出器。'deca' または 'smirk'。",
    )
    device: str = Field(default="cuda:0", description="CUDA デバイス")


class VideoSettings(BaseModel):
    """入力動画の正規化設定。"""

    target_fps: Optional[int] = Field(
        default=25,
        ge=1,
        le=120,
        description="FPS 正規化目標。null なら元動画の FPS を維持。",
    )
    img_size: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="フレーム保存サイズ (px)。FlashAvatar の image_res にも使う。",
    )
    center_crop: bool = Field(
        default=True,
        description="短辺基準中央クロップ後リサイズするか。False なら単純リサイズ。",
    )
    max_frames: Optional[int] = Field(
        default=None,
        ge=1,
        description="最大処理フレーム数。null なら全フレーム。",
    )


class ExtractorSubSettings(BaseModel):
    """DECA / SMIRK 個別設定の共通形。"""

    model_path: str
    repo_dir: str
    input_size: int = Field(default=224, ge=1)


class ExtractorSettings(BaseModel):
    """deca / smirk 個別設定をまとめる。"""

    deca: ExtractorSubSettings = Field(
        default=ExtractorSubSettings(
            model_path="./checkpoints/deca/deca_model.tar",
            repo_dir="./third_party/DECA",
            input_size=224,
        )
    )
    smirk: ExtractorSubSettings = Field(
        default=ExtractorSubSettings(
            model_path="./checkpoints/smirk/SMIRK_em1.pt",
            repo_dir="./third_party/smirk",
            input_size=224,
        )
    )


class FlashAvatarSettings(BaseModel):
    """FlashAvatar リポジトリ・学習ハイパーパラメータ設定。"""

    repo_dir: str = Field(default="./third_party/FlashAvatar")
    iterations: int = Field(
        default=30000,
        ge=1,
        description="学習イテレーション数 (RTX 3090 + img_size=512 で約 30 分)。",
    )
    data_root: str = Field(
        default="./data/flashavatar_training",
        description="学習データ出力ルートディレクトリ。",
    )
    resume_if_exists: bool = Field(
        default=True,
        description="既存チェックポイントを検出した場合、--start_checkpoint で再開する。",
    )
    head_pose_aware: bool = Field(
        default=False,
        description=(
            "頭部位置考慮モード。DECA/SMIRK のクロップ bbox 中心を元画像空間から "
            "クロップ画像空間に写像し、Step 3 (.frame 変換) で K を、Step 4 "
            "(FlashAvatar 学習) で world-view translation を補正して、"
            "検出された頭部位置に Gaussian を配置する。"
        ),
    )


class StageSettings(BaseModel):
    """各ステージの実行可否フラグ。"""

    extract: bool = Field(default=True, description="Step 1: フレーム抽出 + 特徴抽出")
    masks: bool = Field(default=True, description="Step 2: MediaPipe マスク生成")
    convert: bool = Field(default=True, description="Step 3: .pt → .frame 変換")
    train: bool = Field(default=True, description="Step 4: FlashAvatar 学習")
    test: bool = Field(default=True, description="Step 5: 検証動画生成")


class LoggingSettings(BaseModel):
    """ログ設定。"""

    level: str = Field(default="INFO")
    file: str = Field(default="./logs/train_flashavatar.log")
    rotation: str = Field(default="10 MB")


class TrainFlashAvatarConfig(BaseModel):
    """FlashAvatar 学習パイプラインの全設定。"""

    pipeline: PipelineTrainSettings = Field(default_factory=PipelineTrainSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    extractor: ExtractorSettings = Field(default_factory=ExtractorSettings)
    flashavatar: FlashAvatarSettings = Field(default_factory=FlashAvatarSettings)
    stages: StageSettings = Field(default_factory=StageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "TrainFlashAvatarConfig":
        if self.pipeline.extractor not in ("deca", "smirk"):
            raise ValueError(
                f"extractor must be 'deca' or 'smirk', got: {self.pipeline.extractor}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainFlashAvatarConfig":
        """YAML ファイルから設定を読み込む。

        Args:
            path: YAML ファイルのパス。存在しない場合は FileNotFoundError。

        Returns:
            バリデーション済みの設定オブジェクト。

        Raises:
            FileNotFoundError: ファイルが存在しない場合。
            ValueError: YAML の内容がスキーマに合わない場合。
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"設定ファイルが見つかりません: {p}\n"
                f"デフォルト設定を使用する場合は --config を省略してください。"
            )
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def active_extractor_settings(self) -> ExtractorSubSettings:
        """現在選択中の Extractor 設定を返す。"""
        return getattr(self.extractor, self.pipeline.extractor)
