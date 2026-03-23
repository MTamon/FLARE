"""flare.config モジュールのテスト。

load_config / save_config / 各ConfigモデルのデフォルトB値・バリデーションを検証する。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flare.config import (
    BufferConfig,
    FLAREConfig,
    PipelineConfig,
    load_config,
    save_config,
)
from flare.utils.errors import ConfigError


class TestLoadConfig:
    """load_config() のテストスイート。"""

    def test_load_config_valid(self, dummy_config_yaml: Path) -> None:
        """有効なconfig.yamlを読み込みFLAREConfigが返ることを確認する。

        Args:
            dummy_config_yaml: 有効なYAMLファイルのパス。
        """
        config: FLAREConfig = load_config(dummy_config_yaml)

        assert isinstance(config, FLAREConfig)
        assert config.pipeline.name == "test_pipeline"
        assert config.extractor.type == "deca"
        assert config.renderer.type == "flash_avatar"
        assert config.lhg_model.type == "learning2listen"
        assert config.audio.sample_rate == 16000
        assert config.buffer.max_size == 256
        assert config.device_map.extractor == "cuda:0"
        assert config.logging.level == "INFO"
        assert config.checkpoint.enabled is True

    def test_load_config_missing_file(self) -> None:
        """存在しないファイルパスでConfigErrorが送出されることを確認する。"""
        with pytest.raises(ConfigError, match="見つかりません"):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_invalid_yaml(self, tmp_path: Path) -> None:
        """不正なYAML内容でConfigErrorが送出されることを確認する。

        Args:
            tmp_path: pytestが提供する一時ディレクトリ。
        """
        bad_yaml: Path = tmp_path / "bad.yaml"
        bad_yaml.write_text("pipeline:\n  name: test\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="バリデーション"):
            load_config(bad_yaml)

    def test_load_config_malformed_yaml(self, tmp_path: Path) -> None:
        """構文的に不正なYAMLでConfigErrorが送出されることを確認する。

        Args:
            tmp_path: pytestが提供する一時ディレクトリ。
        """
        malformed: Path = tmp_path / "malformed.yaml"
        malformed.write_text("{invalid: [}", encoding="utf-8")

        with pytest.raises(ConfigError, match="YAML"):
            load_config(malformed)


class TestConfigDefaults:
    """各Configモデルのデフォルト値テスト。"""

    def test_pipeline_config_defaults(self) -> None:
        """PipelineConfigのデフォルト値を確認する。"""
        config: PipelineConfig = PipelineConfig(name="test")

        assert config.fps == 30
        assert config.device == "cuda:0"
        assert config.converter_chain == []

    def test_buffer_config_defaults(self) -> None:
        """BufferConfigのデフォルト値を確認する。"""
        config: BufferConfig = BufferConfig()

        assert config.max_size == 256
        assert config.timeout_sec == 0.5
        assert config.overflow_policy == "drop_oldest"


class TestSaveAndReloadConfig:
    """save_config / load_config のラウンドトリップテスト。"""

    def test_save_and_reload_config(
        self, dummy_flare_config: FLAREConfig, tmp_path: Path
    ) -> None:
        """save_config()で書き出した後load_config()で再読み込みして等価であることを確認する。

        Args:
            dummy_flare_config: テスト用FLAREConfig。
            tmp_path: pytestが提供する一時ディレクトリ。
        """
        yaml_path: Path = tmp_path / "roundtrip.yaml"

        save_config(dummy_flare_config, yaml_path)
        reloaded: FLAREConfig = load_config(yaml_path)

        assert reloaded.pipeline.name == dummy_flare_config.pipeline.name
        assert reloaded.pipeline.fps == dummy_flare_config.pipeline.fps
        assert reloaded.pipeline.device == dummy_flare_config.pipeline.device
        assert reloaded.extractor.type == dummy_flare_config.extractor.type
        assert reloaded.extractor.model_path == dummy_flare_config.extractor.model_path
        assert reloaded.extractor.input_size == dummy_flare_config.extractor.input_size
        assert reloaded.renderer.type == dummy_flare_config.renderer.type
        assert reloaded.renderer.output_size == dummy_flare_config.renderer.output_size
        assert reloaded.lhg_model.type == dummy_flare_config.lhg_model.type
        assert reloaded.lhg_model.window_size == dummy_flare_config.lhg_model.window_size
        assert reloaded.audio.sample_rate == dummy_flare_config.audio.sample_rate
        assert reloaded.audio.feature_type == dummy_flare_config.audio.feature_type
        assert reloaded.buffer.max_size == dummy_flare_config.buffer.max_size
        assert reloaded.buffer.overflow_policy == dummy_flare_config.buffer.overflow_policy
        assert reloaded.device_map.extractor == dummy_flare_config.device_map.extractor
        assert reloaded.logging.level == dummy_flare_config.logging.level
        assert reloaded.checkpoint.enabled == dummy_flare_config.checkpoint.enabled
