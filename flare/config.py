"""FLARE パイプライン設定管理モジュール。

YAML形式の設定ファイルを読み込み、pydantic v2によるバリデーションを行う。
仕様書§8.7で定義された全設定項目をサポートする。

Example:
    YAMLファイルから設定を読み込む::

        from flare.config import PipelineConfig

        config = PipelineConfig.from_yaml("config.yaml")
        print(config.extractor.type)  # "deca"
        print(config.buffer.overflow_policy)  # "drop_oldest"

    Pythonオブジェクトから直接生成::

        config = PipelineConfig(
            pipeline=PipelineSettings(name="test", fps=30),
            extractor=ExtractorConfig(type="deca", model_path="./model.tar"),
            ...
        )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, Field


class PipelineSettings(BaseModel):
    """パイプライン全体の基本設定。

    Attributes:
        name: パイプライン名。実行ログやチェックポイントの識別に使用。
        fps: 目標フレームレート。リアルタイムモードの制御に使用。
        device: デフォルトの計算デバイス。device_mapで個別指定がない場合に使用。
        converter_chain: パラメータ変換チェーンの定義。各要素は{"type": "<adapter名>"}。
    """

    name: str = Field(default="lhg_realtime_v1", description="パイプライン名")
    fps: int = Field(default=30, ge=1, le=300, description="目標フレームレート")
    device: str = Field(default="cuda:0", description="デフォルト計算デバイス")
    converter_chain: list[dict[str, str]] = Field(
        default_factory=list,
        description="パラメータ変換チェーン定義",
    )


class ExtractorConfig(BaseModel):
    """特徴量抽出モジュールの設定。

    Attributes:
        type: Extractorの種別。"deca", "deep3d", "smirk", "tdddfa"のいずれか。
        model_path: 事前学習済みモデルファイルのパス。
        input_size: 入力画像のサイズ（正方形の一辺）。
        return_keys: extract()が返すDictのキーリスト。
    """

    type: str = Field(default="deca", description="Extractor種別")
    model_path: str = Field(default="./checkpoints/deca_model.tar", description="モデルパス")
    input_size: int = Field(default=224, ge=1, description="入力画像サイズ")
    return_keys: list[str] = Field(
        default_factory=lambda: ["shape", "exp", "pose", "detail"],
        description="出力Dictのキーリスト",
    )


class RendererConfig(BaseModel):
    """レンダリングモジュールの設定。

    Attributes:
        type: Rendererの種別。"flash_avatar", "pirender", "headgas"のいずれか。
        model_path: 事前学習済みモデルのディレクトリまたはファイルパス。
        source_image: setup()時に登録するソース肖像画像のパス。
        output_size: 出力画像の[幅, 高さ]。
    """

    type: str = Field(default="flash_avatar", description="Renderer種別")
    model_path: str = Field(
        default="./checkpoints/flashavatar/", description="モデルパス"
    )
    source_image: Optional[str] = Field(
        default=None, description="ソース肖像画像パス（setup()時に使用）"
    )
    output_size: list[int] = Field(
        default_factory=lambda: [512, 512], description="出力画像サイズ [幅, 高さ]"
    )


class LHGModelConfig(BaseModel):
    """LHGモデルインターフェースの設定。

    Attributes:
        type: LHGモデルの種別。"learning2listen"等。
        model_path: 事前学習済みモデルファイルのパス。
        window_size: ウィンドウレベル入力のフレーム数。L2Lでは64。
        codebook_size: VQ-VAEのコードブックサイズ。
    """

    type: str = Field(default="learning2listen", description="LHGモデル種別")
    model_path: str = Field(
        default="./checkpoints/l2l_vqvae.pth", description="モデルパス"
    )
    window_size: int = Field(default=64, ge=1, description="ウィンドウフレーム数")
    codebook_size: int = Field(default=256, ge=1, description="コードブックサイズ")


class AudioConfig(BaseModel):
    """音声入力・特徴量抽出の設定。

    Attributes:
        sample_rate: 音声のサンプリングレート（Hz）。
        feature_type: 音声特徴量の種別。"mel", "hubert", "wav2vec2"のいずれか。
        n_mels: メル周波数ビンの数。feature_type="mel"の場合に使用。
    """

    sample_rate: int = Field(default=16000, ge=1, description="サンプリングレート (Hz)")
    feature_type: str = Field(default="mel", description="音声特徴量種別")
    n_mels: int = Field(default=128, ge=1, description="メル周波数ビン数")


class BufferConfig(BaseModel):
    """PipelineBufferの設定。

    Attributes:
        max_size: バッファの最大フレーム数。
        timeout_sec: get()のタイムアウト秒数。
        overflow_policy: オーバーフロー時の方針。
            "drop_oldest", "block", "interpolate"のいずれか。
    """

    max_size: int = Field(default=256, ge=1, description="最大フレーム数")
    timeout_sec: float = Field(default=0.5, ge=0.0, description="get()タイムアウト秒数")
    overflow_policy: str = Field(
        default="drop_oldest",
        pattern=r"^(drop_oldest|block|interpolate)$",
        description="オーバーフローポリシー",
    )


class DeviceMapConfig(BaseModel):
    """マルチGPU配置戦略の設定。

    各コンポーネントの計算デバイスを個別に指定する。
    単一GPU構成では全て同一デバイスを指定する。

    Attributes:
        extractor: Extractorの計算デバイス。
        lhg_model: LHGモデルの計算デバイス。
        renderer: Rendererの計算デバイス。
    """

    extractor: str = Field(default="cuda:0", description="Extractorデバイス")
    lhg_model: str = Field(default="cuda:0", description="LHGモデルデバイス")
    renderer: str = Field(default="cuda:0", description="Rendererデバイス")


class LoggingConfig(BaseModel):
    """Loguruベースのロギング設定。

    Attributes:
        level: ログレベル。"DEBUG", "INFO", "WARNING", "ERROR"のいずれか。
        file: ログファイルの出力パス。
        rotation: ログファイルのローテーション閾値。
    """

    level: str = Field(default="INFO", description="ログレベル")
    file: str = Field(default="./logs/pipeline.log", description="ログファイルパス")
    rotation: str = Field(default="10 MB", description="ローテーション閾値")


class CheckpointConfig(BaseModel):
    """バッチ処理のチェックポイント設定。

    Attributes:
        enabled: チェックポイント機能の有効/無効。
        save_dir: チェックポイントファイルの保存ディレクトリ。
        format: チェックポイントのファイル形式。
    """

    enabled: bool = Field(default=True, description="チェックポイント有効/無効")
    save_dir: str = Field(
        default="./checkpoints/batch/", description="保存ディレクトリ"
    )
    format: str = Field(default="json", description="ファイル形式")


class PipelineConfig(BaseModel):
    """FLAREパイプラインのルート設定モデル。

    仕様書§8.7で定義されたYAML設定ファイルの全構造をモデル化する。
    pydantic v2のBaseModelを継承し、型バリデーションとデフォルト値を提供する。

    Attributes:
        pipeline: パイプライン全体の基本設定。
        extractor: 特徴量抽出モジュールの設定。
        lhg_model: LHGモデルインターフェースの設定。
        renderer: レンダリングモジュールの設定。
        audio: 音声入力・特徴量抽出の設定。
        buffer: PipelineBufferの設定。
        device_map: マルチGPU配置戦略の設定。
        logging: ロギングの設定。
        checkpoint: チェックポイントの設定。

    Example:
        YAMLファイルからの読み込み::

            config = PipelineConfig.from_yaml("config.yaml")
            print(config.pipeline.name)

        デフォルト設定での生成::

            config = PipelineConfig()
    """

    pipeline: PipelineSettings = Field(
        default_factory=PipelineSettings, description="パイプライン基本設定"
    )
    extractor: ExtractorConfig = Field(
        default_factory=ExtractorConfig, description="Extractor設定"
    )
    lhg_model: LHGModelConfig = Field(
        default_factory=LHGModelConfig, description="LHGモデル設定"
    )
    renderer: RendererConfig = Field(
        default_factory=RendererConfig, description="Renderer設定"
    )
    audio: AudioConfig = Field(
        default_factory=AudioConfig, description="音声設定"
    )
    buffer: BufferConfig = Field(
        default_factory=BufferConfig, description="バッファ設定"
    )
    device_map: DeviceMapConfig = Field(
        default_factory=DeviceMapConfig, description="デバイスマップ設定"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="ロギング設定"
    )
    checkpoint: CheckpointConfig = Field(
        default_factory=CheckpointConfig, description="チェックポイント設定"
    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> PipelineConfig:
        """YAMLファイルからPipelineConfigを生成する。

        Args:
            path: YAML設定ファイルのパス。文字列またはPathオブジェクト。

        Returns:
            バリデーション済みのPipelineConfigインスタンス。

        Raises:
            FileNotFoundError: 指定パスにファイルが存在しない場合。
            yaml.YAMLError: YAMLの構文が不正な場合。
            pydantic.ValidationError: 設定値がスキーマに適合しない場合。
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls.model_validate(data)
