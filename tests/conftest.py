"""pytest 共通フィクスチャ (Phase 1)"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch

from flare.config import ToolConfig
from flare.converters.base import BaseAdapter
from flare.extractors.base import BaseExtractor
from flare.model_interface.base import BaseLHGModel
from flare.renderers.base import BaseRenderer


# ---------------------------------------------------------------------------
# ダミー (モック) 実装 — ABC の具象化テスト用
# ---------------------------------------------------------------------------


class DummyExtractor(BaseExtractor):
    """DECA 風のダミー Extractor。exp 50D + pose 6D を返す。"""

    def extract(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = image.shape[0]
        return {
            "shape": torch.zeros(B, 100),
            "exp": torch.randn(B, 50),
            "pose": torch.randn(B, 6),
            "detail": torch.zeros(B, 128),
        }

    @property
    def param_dim(self) -> int:
        return 100 + 50 + 6 + 128  # 284

    @property
    def param_keys(self) -> List[str]:
        return ["shape", "exp", "pose", "detail"]


class DummyRenderer(BaseRenderer):
    """FlashAvatar 風のダミー Renderer。"""

    def __init__(self) -> None:
        self._initialized = False

    def setup(self, source_image: Optional[torch.Tensor] = None, **kwargs) -> None:
        self._initialized = True

    def render(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self._initialized:
            raise RuntimeError("setup() not called")
        return torch.zeros(1, 3, 512, 512)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class DummyLHGModel(BaseLHGModel):
    """L2L 風のダミー LHG モデル。ウィンドウサイズ 64。"""

    def predict(
        self,
        audio_features: torch.Tensor,
        speaker_motion: torch.Tensor,
    ) -> torch.Tensor:
        B = audio_features.shape[0]
        return torch.randn(B, 64, 56)  # (B, T_out, D_out)

    @property
    def requires_window(self) -> bool:
        return True

    @property
    def window_size(self) -> Optional[int]:
        return 64


class DummyAdapter(BaseAdapter):
    """IdentityAdapter 風のダミー。"""

    def convert(self, source_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return source_params

    @property
    def source_format(self) -> str:
        return "dummy_src"

    @property
    def target_format(self) -> str:
        return "dummy_tgt"


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_extractor() -> DummyExtractor:
    return DummyExtractor()


@pytest.fixture
def dummy_renderer() -> DummyRenderer:
    return DummyRenderer()


@pytest.fixture
def dummy_lhg_model() -> DummyLHGModel:
    return DummyLHGModel()


@pytest.fixture
def dummy_adapter() -> DummyAdapter:
    return DummyAdapter()


@pytest.fixture
def sample_config_dict() -> dict:
    """Section 8.7 の config.yaml に対応する最小限の dict。"""
    return {
        "pipeline": {
            "name": "test_pipeline",
            "fps": 30,
            "device": "cuda:0",
            "converter_chain": [{"type": "identity"}],
        },
        "extractor": {
            "type": "deca",
            "model_path": "./checkpoints/deca_model.tar",
            "input_size": 224,
            "return_keys": ["shape", "exp", "pose", "detail"],
        },
        "lhg_model": {
            "type": "learning2listen",
            "model_path": "./checkpoints/l2l_vqvae.pth",
            "window_size": 64,
            "codebook_size": 256,
        },
        "renderer": {
            "type": "flash_avatar",
            "model_path": "./checkpoints/flashavatar/",
            "source_image": "./data/source_portrait.png",
            "output_size": [512, 512],
        },
        "audio": {
            "sample_rate": 16000,
            "feature_type": "mel",
            "n_mels": 128,
        },
        "buffer": {
            "max_size": 256,
            "timeout_sec": 0.5,
            "overflow_policy": "drop_oldest",
        },
        "device_map": {
            "extractor": "cuda:0",
            "lhg_model": "cuda:0",
            "renderer": "cuda:0",
        },
        "logging": {
            "level": "INFO",
            "file": "./logs/test.log",
            "rotation": "10 MB",
        },
        "checkpoint": {
            "enabled": True,
            "save_dir": "./checkpoints/batch/",
            "format": "json",
        },
    }


@pytest.fixture
def sample_config(sample_config_dict) -> ToolConfig:
    return ToolConfig(**sample_config_dict)


@pytest.fixture
def sample_config_yaml(tmp_path, sample_config_dict) -> Path:
    """一時ディレクトリに config.yaml を書き出して返す。"""
    import yaml

    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(sample_config_dict, f, default_flow_style=False)
    return path