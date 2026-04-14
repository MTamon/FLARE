"""PipelineConfig の pydantic バリデーションテスト。

正常系（デフォルト値、YAML読み込み）と異常系（不正値、ファイル不在）を検証する。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from flare.config import (
    BufferConfig,
    ExtractorConfig,
    PipelineConfig,
    PipelineSettings,
    RendererConfig,
)


class TestPipelineConfigDefaults:
    """デフォルト値によるインスタンス化テスト。"""

    def test_default_instantiation(self, minimal_config: PipelineConfig) -> None:
        """全フィールドがデフォルト値で正しくインスタンス化されること。"""
        assert minimal_config.pipeline.name == "lhg_realtime_v1"
        assert minimal_config.pipeline.fps == 30
        assert minimal_config.pipeline.device == "cuda:0"
        assert minimal_config.extractor.type == "deca"
        assert minimal_config.renderer.type == "flash_avatar"
        assert minimal_config.buffer.overflow_policy == "drop_oldest"
        assert minimal_config.lhg_model.window_size == 64
        assert minimal_config.audio.sample_rate == 16000
        assert minimal_config.logging.level == "INFO"
        assert minimal_config.checkpoint.enabled is True

    def test_converter_chain_default_empty(
        self, minimal_config: PipelineConfig
    ) -> None:
        """converter_chainのデフォルトが空リストであること。"""
        assert minimal_config.pipeline.converter_chain == []


class TestPipelineConfigValidation:
    """pydantic バリデーションテスト。"""

    def test_invalid_overflow_policy(self) -> None:
        """不正なoverflow_policyでValidationErrorが発生すること。"""
        with pytest.raises(ValidationError):
            BufferConfig(overflow_policy="invalid_policy")

    def test_valid_overflow_policies(self) -> None:
        """3つの正規overflow_policyが受け入れられること。"""
        for policy in ("drop_oldest", "block", "interpolate"):
            cfg = BufferConfig(overflow_policy=policy)
            assert cfg.overflow_policy == policy

    def test_negative_fps_rejected(self) -> None:
        """負のFPS値が拒否されること。"""
        with pytest.raises(ValidationError):
            PipelineSettings(fps=0)

    def test_negative_max_size_rejected(self) -> None:
        """負のmax_sizeが拒否されること。"""
        with pytest.raises(ValidationError):
            BufferConfig(max_size=0)

    def test_custom_values(self) -> None:
        """カスタム値でのインスタンス化が正しく動作すること。"""
        config = PipelineConfig(
            pipeline=PipelineSettings(name="test_pipe", fps=60),
            extractor=ExtractorConfig(type="smirk", input_size=256),
            renderer=RendererConfig(
                type="pirender", output_size=[1024, 1024]
            ),
        )
        assert config.pipeline.name == "test_pipe"
        assert config.pipeline.fps == 60
        assert config.extractor.type == "smirk"
        assert config.extractor.input_size == 256
        assert config.renderer.output_size == [1024, 1024]


class TestPipelineConfigFromYaml:
    """from_yaml() のテスト。"""

    def test_from_yaml_full(self, tmp_path: Path) -> None:
        """仕様書§8.7準拠のYAMLが正しく読み込まれること。"""
        yaml_data = {
            "pipeline": {
                "name": "lhg_realtime_v1",
                "fps": 30,
                "device": "cuda:0",
                "converter_chain": [
                    {"type": "deca_to_flame"},
                    {"type": "identity"},
                ],
            },
            "extractor": {
                "type": "deca",
                "model_path": "./checkpoints/deca/deca_model.tar",
                "input_size": 224,
                "return_keys": ["shape", "exp", "pose", "detail"],
            },
            "lhg_model": {
                "type": "learning2listen",
                "model_path": "./checkpoints/l2l/l2l_vqvae.pth",
                "window_size": 64,
                "codebook_size": 256,
            },
            "renderer": {
                "type": "flash_avatar",
                "model_path": "./checkpoints/flashavatar/",
                "source_image": "./data/source_images/source_portrait.png",
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
                "file": "./logs/pipeline.log",
                "rotation": "10 MB",
            },
            "checkpoint": {
                "enabled": True,
                "save_dir": "./checkpoints/batch/",
                "format": "json",
            },
        }
        yaml_path = tmp_path / "config.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f)

        config = PipelineConfig.from_yaml(yaml_path)
        assert config.pipeline.name == "lhg_realtime_v1"
        assert len(config.pipeline.converter_chain) == 2
        assert config.pipeline.converter_chain[0] == {"type": "deca_to_flame"}
        assert config.extractor.type == "deca"
        assert config.renderer.source_image == "./data/source_images/source_portrait.png"
        assert config.lhg_model.codebook_size == 256
        assert config.audio.n_mels == 128
        assert config.buffer.overflow_policy == "drop_oldest"
        assert config.device_map.renderer == "cuda:0"
        assert config.logging.rotation == "10 MB"
        assert config.checkpoint.format == "json"

    def test_from_yaml_partial(self, tmp_path: Path) -> None:
        """部分的なYAMLでデフォルト値が補完されること。"""
        yaml_path = tmp_path / "partial.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump({"pipeline": {"name": "test"}}, f)

        config = PipelineConfig.from_yaml(yaml_path)
        assert config.pipeline.name == "test"
        assert config.pipeline.fps == 30
        assert config.extractor.type == "deca"

    def test_from_yaml_empty(self, tmp_path: Path) -> None:
        """空YAMLでも全デフォルト値でインスタンス化されること。"""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("", encoding="utf-8")

        config = PipelineConfig.from_yaml(yaml_path)
        assert config.pipeline.name == "lhg_realtime_v1"

    def test_from_yaml_file_not_found(self) -> None:
        """存在しないファイルでFileNotFoundErrorが発生すること。"""
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml("/nonexistent/config.yaml")

    def test_from_yaml_invalid_content(self, tmp_path: Path) -> None:
        """不正な値を含むYAMLでValidationErrorが発生すること。"""
        yaml_path = tmp_path / "invalid.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                {"buffer": {"overflow_policy": "nonexistent_policy"}}, f
            )

        with pytest.raises(ValidationError):
            PipelineConfig.from_yaml(yaml_path)
