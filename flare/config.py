"""FLARE configuration management using YAML and pydantic v2.

仕様書8.7節のYAML設定スキーマをpydantic v2 BaseModelで完全にモデル化する。
YAML形式の設定ファイルの読み込み・バリデーション・書き出しを提供する。

Example:
    >>> from flare.config import load_config, save_config
    >>> config = load_config("config.yaml")
    >>> print(config.pipeline.name)
    'lhg_realtime_v1'
    >>> save_config(config, "config_backup.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field


class ConverterChainItemConfig(BaseModel):
    """コンバータチェーン内の個別コンバータ設定。

    パイプラインで適用されるパラメータ変換の1ステップを表す。

    Attributes:
        type: コンバータの種別名。AdapterRegistryでの解決に使用される。
            例: ``"deca_to_flame"``, ``"identity"``
    """

    type: str


class PipelineConfig(BaseModel):
    """パイプライン全体の実行設定。

    リアルタイムモード・バッチモード共通のパイプライン動作パラメータを定義する。

    Attributes:
        name: パイプラインの識別名。ログやチェックポイントの識別に使用。
        fps: 目標フレームレート。リアルタイムモードでの処理速度目標。
        device: デフォルトの計算デバイス。device_mapで個別指定がない場合に使用。
        converter_chain: 適用するコンバータの順序付きリスト。
    """

    name: str
    fps: int = Field(default=30, gt=0, description="目標フレームレート")
    device: str = Field(default="cuda:0", description="デフォルト計算デバイス")
    converter_chain: List[ConverterChainItemConfig] = Field(
        default_factory=list,
        description="パラメータ変換チェーンの設定リスト",
    )


class ExtractorConfig(BaseModel):
    """特徴量抽出器（Extractor）の設定。

    3DMMパラメータ抽出に使用するモデルとその動作パラメータを定義する。
    ルートA（Deep3DFaceRecon / BFM）またはルートB（DECA / SMIRK / FLAME）を選択。

    Attributes:
        type: Extractorの種別名。例: ``"deca"``, ``"deep3d"``, ``"smirk"``
        model_path: 事前学習済みモデルのチェックポイントパス。
        input_size: 入力画像のリサイズ先（正方形の辺の長さ、ピクセル単位）。
        return_keys: extract()の戻りDictに含めるキーのリスト。
    """

    type: str
    model_path: str
    input_size: int = Field(default=224, gt=0, description="入力画像サイズ（px）")
    return_keys: List[str] = Field(
        default_factory=list,
        description="抽出パラメータのキーリスト",
    )


class RendererConfig(BaseModel):
    """レンダラーの設定。

    3DMMパラメータからフォトリアルな顔画像を生成するレンダラーを定義する。
    ルートA（PIRender / BFM）またはルートB（FlashAvatar / HeadGaS / FLAME）を選択。

    Attributes:
        type: Rendererの種別名。例: ``"flash_avatar"``, ``"pirender"``, ``"headgas"``
        model_path: 事前学習済みモデルのパスまたはディレクトリ。
        source_image: setup()時に登録するソース肖像画像のパス。
            FlashAvatarでは不要（学習済みNeRFをロード）、PIRenderでは必須。
        output_size: 出力画像の解像度 ``(height, width)``。
    """

    type: str
    model_path: str
    source_image: Optional[str] = Field(
        default=None,
        description="ソース肖像画像パス（PIRender用）",
    )
    output_size: Tuple[int, int] = Field(
        default=(512, 512),
        description="出力画像解像度 (H, W)",
    )


class LHGModelConfig(BaseModel):
    """LHGモデルインターフェースの設定。

    音声特徴量と話者動作からリスナー動作を予測するLHGモデルの設定を定義する。

    Attributes:
        type: LHGモデルの種別名。例: ``"learning2listen"``
        model_path: 事前学習済みモデルのチェックポイントパス。
        window_size: ウィンドウレベル入力時のフレーム数。L2Lでは64。
        codebook_size: VQ-VAE等で使用するコードブックサイズ。
    """

    type: str
    model_path: str
    window_size: int = Field(
        default=64,
        gt=0,
        description="入力ウィンドウのフレーム数",
    )
    codebook_size: int = Field(
        default=256,
        gt=0,
        description="VQ-VAEコードブックサイズ",
    )


class AudioConfig(BaseModel):
    """音声入力・特徴量抽出の設定。

    音声ストリームのサンプリングレートおよび特徴量抽出方式を定義する。

    Attributes:
        sample_rate: 音声のサンプリングレート（Hz）。
        feature_type: 音声特徴量の種別。``"mel"``, ``"hubert"``, ``"wav2vec2"`` 等。
        n_mels: メルスペクトログラムのメルフィルタバンク数。
            feature_typeが ``"mel"`` の場合に使用。
    """

    sample_rate: int = Field(
        default=16000,
        gt=0,
        description="サンプリングレート（Hz）",
    )
    feature_type: str = Field(
        default="mel",
        description="音声特徴量種別",
    )
    n_mels: int = Field(
        default=128,
        gt=0,
        description="メルフィルタバンク数",
    )


class BufferConfig(BaseModel):
    """PipelineBufferの設定。

    パイプラインのステージ間でデータを受け渡すキューバッファの動作設定。

    Attributes:
        max_size: バッファの最大フレーム数。
        timeout_sec: get()操作のタイムアウト秒数。
        overflow_policy: オーバーフロー時の方針。
            ``"drop_oldest"``（リアルタイム用）または ``"block"``（バッチ用）。
    """

    max_size: int = Field(
        default=256,
        gt=0,
        description="バッファ最大フレーム数",
    )
    timeout_sec: float = Field(
        default=0.5,
        gt=0.0,
        description="get()タイムアウト（秒）",
    )
    overflow_policy: str = Field(
        default="drop_oldest",
        description="オーバーフローポリシー: drop_oldest / block",
    )


class DeviceMapConfig(BaseModel):
    """マルチGPU配置戦略の設定。

    各コンポーネントを配置するCUDAデバイスを指定する。
    単一GPU構成では全て同一デバイスを指定し、マルチGPU構成では
    コンポーネントごとに異なるデバイスを割り当てる。

    Attributes:
        extractor: Extractorを配置するデバイス。
        lhg_model: LHGモデルを配置するデバイス。
        renderer: Rendererを配置するデバイス。
    """

    extractor: str = Field(default="cuda:0", description="Extractorデバイス")
    lhg_model: str = Field(default="cuda:0", description="LHGモデルデバイス")
    renderer: str = Field(default="cuda:0", description="Rendererデバイス")


class LoggingConfig(BaseModel):
    """Loguruロギングフレームワークの設定。

    ログレベル、出力先ファイル、ローテーション設定を定義する。

    Attributes:
        level: ログレベル。``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"`` 等。
        file: ログファイルの出力パス。
        rotation: ログファイルのローテーション条件。例: ``"10 MB"``。
    """

    level: str = Field(default="INFO", description="ログレベル")
    file: str = Field(
        default="./logs/pipeline.log",
        description="ログファイルパス",
    )
    rotation: str = Field(
        default="10 MB",
        description="ログローテーション条件",
    )


class CheckpointConfig(BaseModel):
    """バッチ処理チェックポイントの設定。

    バッチモードにおける中断・再開機能のためのチェックポイント設定を定義する。

    Attributes:
        enabled: チェックポイント機能の有効/無効。
        save_dir: チェックポイントファイルの保存ディレクトリ。
        format: チェックポイントのシリアライズ形式。現在は ``"json"`` のみ対応。
    """

    enabled: bool = Field(default=True, description="チェックポイント有効化")
    save_dir: str = Field(
        default="./checkpoints/batch/",
        description="チェックポイント保存先",
    )
    format: str = Field(default="json", description="チェックポイント形式")


class FLAREConfig(BaseModel):
    """FLARE統合設定のルートモデル。

    仕様書8.7節で定義されるYAML設定スキーマの全セクションを統合する。
    ``load_config()`` でYAMLファイルから読み込み、pydantic v2による
    バリデーションを自動的に適用する。

    Attributes:
        pipeline: パイプライン全体の実行設定。
        extractor: 特徴量抽出器の設定。
        renderer: レンダラーの設定。
        lhg_model: LHGモデルの設定。
        audio: 音声入力・特徴量抽出の設定。
        buffer: PipelineBufferの設定。
        device_map: マルチGPU配置戦略の設定。
        logging: ロギングフレームワークの設定。
        checkpoint: バッチ処理チェックポイントの設定。

    Example:
        >>> config = FLAREConfig(
        ...     pipeline=PipelineConfig(name="test"),
        ...     extractor=ExtractorConfig(type="deca", model_path="./model.tar"),
        ...     renderer=RendererConfig(type="flash_avatar", model_path="./fa/"),
        ...     lhg_model=LHGModelConfig(type="learning2listen", model_path="./l2l.pth"),
        ... )
        >>> config.pipeline.fps
        30
    """

    pipeline: PipelineConfig
    extractor: ExtractorConfig
    renderer: RendererConfig
    lhg_model: LHGModelConfig
    audio: AudioConfig = Field(default_factory=AudioConfig)
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    device_map: DeviceMapConfig = Field(default_factory=DeviceMapConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


def load_config(path: Union[str, Path]) -> FLAREConfig:
    """YAMLファイルからFLARE設定を読み込みバリデーションする。

    指定されたパスのYAMLファイルを読み込み、pydantic v2による
    型チェック・制約バリデーションを適用して ``FLAREConfig`` を返す。

    Args:
        path: YAML設定ファイルのパス。文字列またはPathオブジェクト。

    Returns:
        バリデーション済みのFLARE統合設定。

    Raises:
        ConfigError: ファイルが存在しない場合、YAMLパースに失敗した場合、
            またはバリデーションに失敗した場合。

    Example:
        >>> config = load_config("config.yaml")
        >>> print(config.extractor.type)
        'deca'
    """
    from flare.utils.errors import ConfigError

    filepath: Path = Path(path)

    if not filepath.exists():
        raise ConfigError(f"設定ファイルが見つかりません: {filepath}")

    try:
        raw_text: str = filepath.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"設定ファイルの読み込みに失敗しました: {filepath}") from exc

    try:
        data: dict = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAMLパースに失敗しました: {filepath}") from exc

    if data is None or not isinstance(data, dict):
        raise ConfigError(f"設定ファイルの内容が不正です（dictが期待されます）: {filepath}")

    try:
        config: FLAREConfig = FLAREConfig.model_validate(data)
    except Exception as exc:
        raise ConfigError(
            f"設定バリデーションに失敗しました: {exc}"
        ) from exc

    return config


def save_config(config: FLAREConfig, path: Union[str, Path]) -> None:
    """FLAREConfigをYAMLファイルに書き出す。

    pydantic モデルをシリアライズし、YAML形式で指定パスに保存する。
    親ディレクトリが存在しない場合は自動的に作成する。

    Args:
        config: 書き出すFLARE統合設定。
        path: 出力先YAMLファイルのパス。文字列またはPathオブジェクト。

    Raises:
        OSError: ファイル書き出しに失敗した場合。

    Example:
        >>> save_config(config, "config_backup.yaml")
    """
    filepath: Path = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data: dict = config.model_dump(mode="python")

    # Tuple → List 変換（PyYAMLはtupleをタグ付き出力するため）
    _convert_tuples(data)

    yaml_str: str = yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    filepath.write_text(yaml_str, encoding="utf-8")


def _convert_tuples(obj: object) -> object:
    """ネストされた辞書・リスト内のtupleを再帰的にlistに変換する。

    PyYAMLはtupleを ``!!python/tuple`` タグ付きで出力するため、
    YAML書き出し前にlistへ変換してクリーンな出力を得る。

    Args:
        obj: 変換対象のオブジェクト。dictまたはlistの場合は再帰処理する。

    Returns:
        tuple→list変換済みのオブジェクト。入力がdictの場合はin-placeで変更される。
    """
    if isinstance(obj, dict):
        for key in obj:
            if isinstance(obj[key], tuple):
                obj[key] = list(obj[key])
            else:
                _convert_tuples(obj[key])
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, tuple):
                obj[i] = list(item)
            else:
                _convert_tuples(item)
    return obj
