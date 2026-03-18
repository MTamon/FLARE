"""config.py のテスト (Section 8.7)"""

from __future__ import annotations

import warnings

import pytest

from flare.config import ToolConfig, load_config


class TestToolConfig:
    """ToolConfig の pydantic バリデーションテスト。"""

    def test_load_from_dict(self, sample_config_dict):
        cfg = ToolConfig(**sample_config_dict)
        assert cfg.pipeline.name == "test_pipeline"
        assert cfg.extractor.type == "deca"
        assert cfg.renderer.type == "flash_avatar"
        assert cfg.renderer.output_size == (512, 512)
        assert cfg.lhg_model.window_size == 64
        assert cfg.buffer.overflow_policy == "drop_oldest"

    def test_load_from_yaml(self, sample_config_yaml):
        cfg = load_config(sample_config_yaml)
        assert cfg.extractor.type == "deca"
        assert cfg.renderer.output_size == (512, 512)

    def test_output_size_coercion_from_list(self, sample_config_dict):
        """output_size が list で渡されても tuple に変換される。"""
        sample_config_dict["renderer"]["output_size"] = [256, 256]
        cfg = ToolConfig(**sample_config_dict)
        assert cfg.renderer.output_size == (256, 256)

    def test_route_mismatch_warning(self, sample_config_dict):
        """Section 5.3: BFM extractor + FLAME renderer で警告が出る。"""
        sample_config_dict["extractor"]["type"] = "deep3d"
        sample_config_dict["renderer"]["type"] = "flash_avatar"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ToolConfig(**sample_config_dict)
            route_warnings = [x for x in w if "Route mismatch" in str(x.message)]
            assert len(route_warnings) == 1

    def test_no_warning_for_consistent_route(self, sample_config_dict):
        """同一ルートなら警告なし。"""
        sample_config_dict["extractor"]["type"] = "deca"
        sample_config_dict["renderer"]["type"] = "flash_avatar"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ToolConfig(**sample_config_dict)
            route_warnings = [x for x in w if "Route mismatch" in str(x.message)]
            assert len(route_warnings) == 0

    def test_invalid_fps_rejected(self, sample_config_dict):
        sample_config_dict["pipeline"]["fps"] = 0
        with pytest.raises(Exception):  # pydantic ValidationError
            ToolConfig(**sample_config_dict)

    def test_invalid_feature_type_rejected(self, sample_config_dict):
        sample_config_dict["audio"]["feature_type"] = "invalid"
        with pytest.raises(Exception):
            ToolConfig(**sample_config_dict)

    def test_converter_chain_parsed(self, sample_config_dict):
        sample_config_dict["pipeline"]["converter_chain"] = [
            {"type": "deca_to_flame"},
            {"type": "identity"},
        ]
        cfg = ToolConfig(**sample_config_dict)
        assert len(cfg.pipeline.converter_chain) == 2
        assert cfg.pipeline.converter_chain[0].type == "deca_to_flame"