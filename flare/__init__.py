"""FLARE: Facial Landmark Analysis & Rendering Engine.

LHG研究における特徴量抽出（エンコード）とフォトリアルレンダリング（デコード）の
統合ツール。BFMベース（ルートA）とFLAMEベース（ルートB）の2ルートをサポートし、
リアルタイムモードとバッチ前処理モードの両方に対応する。

Example:
    >>> import flare
    >>> print(flare.__version__)
    '0.1.0'
    >>> config = flare.load_config("config.yaml")
"""

from __future__ import annotations

__version__: str = "0.1.0"

# --- Re-exports from flare.config ---
from flare.config import FLAREConfig as FLAREConfig
from flare.config import PipelineConfig as PipelineConfig
from flare.config import load_config as load_config

# --- Re-exports from flare.utils.errors ---
from flare.utils.errors import ConfigError as ConfigError
from flare.utils.errors import FaceNotDetectedError as FaceNotDetectedError
from flare.utils.errors import ModelLoadError as ModelLoadError
from flare.utils.errors import PipelineError as PipelineError
from flare.utils.errors import ErrorPolicy as ErrorPolicy

# --- Re-exports from flare.utils.logging ---
from flare.utils.logging import setup_logger as setup_logger

__all__: list[str] = [
    "__version__",
    # config
    "FLAREConfig",
    "PipelineConfig",
    "load_config",
    # errors
    "ConfigError",
    "FaceNotDetectedError",
    "ModelLoadError",
    "PipelineError",
    "ErrorPolicy",
    # logging
    "setup_logger",
]
