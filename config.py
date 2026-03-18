"""設定管理 (YAML + pydantic v2 バリデーション)

Section 8.7 の config.yaml スキーマを pydantic v2 モデルで表現する。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class ConverterStepConfig(BaseModel):
    """converter_chain の各ステップ"""
    type: str  # e.g. "deca_to_flame", "identity"


class PipelineConfig(BaseModel):
    name: str = "lhg_realtime_v1"
    fps: int = Field(default=30, ge=1, le=120)
    device: str = "cuda:0"
    converter_chain: List[ConverterStepConfig] = Field(default_factory=list)


class ExtractorConfig(BaseModel):
    type: str  # e.g. "deca", "deep3d", "smirk", "tdddfa"
    model_path: str
    input_size: int = 224
    return_keys: List[str] = Field(default_factory=list)


class LHGModelConfig(BaseModel):
    type: str  # e.g. "learning2listen"
    model_path: str
    window_size: int = Field(default=64, ge=1)
    codebook_size: Optional[int] = None


class RendererConfig(BaseModel):
    type: str  # e.g. "flash_avatar", "pirender", "headgas"
    model_path: str
    source_image: Optional[str] = None  # setup() 時に使用
    output_size: Tuple[int, int] = (512, 512)

    @field_validator("output_size", mode="before")
    @classmethod
    def _coerce_output_size(cls, v):  # noqa: N805
        if isinstance(v, list):
            return tuple(v)
        return v


class AudioConfig(BaseModel):
    sample_rate: int = Field(default=16000, ge=8000)
    feature_type: Literal["mel", "hubert", "wav2vec2"] = "mel"
    n_mels: int = Field(default=128, ge=1)


class BufferConfig(BaseModel):
    max_size: int = Field(default=256, ge=1)
    timeout_sec: float = Field(default=0.5, gt=0.0)
    overflow_policy: Literal["drop_oldest", "block", "interpolate"] = "drop_oldest"


class DeviceMapConfig(BaseModel):
    extractor: str = "cuda:0"
    lhg_model: str = "cuda:0"
    renderer: str = "cuda:0"


class LoggingConfig(BaseModel):
    level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file: str = "./logs/pipeline.log"
    rotation: str = "10 MB"


class CheckpointConfig(BaseModel):
    enabled: bool = True
    save_dir: str = "./checkpoints/batch/"
    format: Literal["json"] = "json"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class ToolConfig(BaseModel):
    """LHG Tool のルート設定。Section 8.7 の YAML スキーマに 1:1 対応。"""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    extractor: ExtractorConfig
    lhg_model: LHGModelConfig
    renderer: RendererConfig
    audio: AudioConfig = Field(default_factory=AudioConfig)
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    device_map: DeviceMapConfig = Field(default_factory=DeviceMapConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @model_validator(mode="after")
    def _validate_route_consistency(self) -> "ToolConfig":
        """ルート A (BFM) とルート B (FLAME) の混在を検出して警告する。

        Section 5.3: 「ルート A とルート B を混在させず、どちらかに統一するのが
        最も実用的」
        """
        bfm_extractors = {"deep3d", "tdddfa"}
        flame_extractors = {"deca", "smirk", "spark", "emoca"}
        bfm_renderers = {"pirender"}
        flame_renderers = {"flash_avatar", "flashavatar", "headgas"}

        ext = self.extractor.type.lower()
        rnd = self.renderer.type.lower()

        ext_is_bfm = ext in bfm_extractors
        ext_is_flame = ext in flame_extractors
        rnd_is_bfm = rnd in bfm_renderers
        rnd_is_flame = rnd in flame_renderers

        if (ext_is_bfm and rnd_is_flame) or (ext_is_flame and rnd_is_bfm):
            import warnings
            warnings.warn(
                f"Route mismatch: extractor='{ext}' and renderer='{rnd}' "
                f"belong to different routes (BFM / FLAME). "
                f"Section 5.3 recommends using a single route.",
                UserWarning,
                stacklevel=2,
            )
        return self


def load_config(path: Union[str, Path]) -> ToolConfig:
    """YAML ファイルから ToolConfig を読み込む。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ToolConfig(**raw)