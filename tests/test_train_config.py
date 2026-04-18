"""TrainFlashAvatarConfig の pydantic バリデーションテストおよび
scripts/train_flashavatar.py の CLI サーフェステスト。

- configs/train_flashavatar.yaml のロードを検証
- 各フィールドのデフォルト値・バリデーションを確認
- CLI 引数の上書きロジックを検証
- エラーメッセージの内容を確認
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from flare.training import (
    ExtractorSettings,
    FlashAvatarSettings,
    PipelineTrainSettings,
    StageSettings,
    TrainFlashAvatarConfig,
    VideoSettings,
)


# ---------------------------------------------------------------------------
# デフォルト値テスト
# ---------------------------------------------------------------------------

class TestTrainFlashAvatarConfigDefaults:
    """デフォルト値による正常インスタンス化テスト。"""

    def test_default_extractor(self) -> None:
        cfg = TrainFlashAvatarConfig()
        assert cfg.pipeline.extractor == "deca"

    def test_default_device(self) -> None:
        cfg = TrainFlashAvatarConfig()
        assert cfg.pipeline.device == "cuda:0"

    def test_default_img_size(self) -> None:
        cfg = TrainFlashAvatarConfig()
        assert cfg.video.img_size == 512

    def test_default_target_fps(self) -> None:
        cfg = TrainFlashAvatarConfig()
        assert cfg.video.target_fps == 25

    def test_default_center_crop(self) -> None:
        cfg = TrainFlashAvatarConfig()
        assert cfg.video.center_crop is True

    def test_default_iterations(self) -> None:
        cfg = TrainFlashAvatarConfig()
        assert cfg.flashavatar.iterations == 30000

    def test_default_resume_if_exists(self) -> None:
        cfg = TrainFlashAvatarConfig()
        assert cfg.flashavatar.resume_if_exists is True

    def test_default_stages_all_true(self) -> None:
        stages = StageSettings()
        assert stages.extract is True
        assert stages.masks is True
        assert stages.convert is True
        assert stages.train is True
        assert stages.test is True

    def test_default_deca_model_path(self) -> None:
        cfg = TrainFlashAvatarConfig()
        assert "deca_model.tar" in cfg.extractor.deca.model_path

    def test_default_smirk_model_path(self) -> None:
        cfg = TrainFlashAvatarConfig()
        assert "SMIRK_em1.pt" in cfg.extractor.smirk.model_path


# ---------------------------------------------------------------------------
# バリデーションテスト
# ---------------------------------------------------------------------------

class TestTrainFlashAvatarConfigValidation:
    """不正値でのバリデーションエラーテスト。"""

    def test_invalid_extractor_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrainFlashAvatarConfig.model_validate(
                {"pipeline": {"extractor": "unknown_extractor"}}
            )

    def test_img_size_too_small_raises(self) -> None:
        with pytest.raises(ValidationError):
            VideoSettings(img_size=0)

    def test_img_size_too_large_raises(self) -> None:
        with pytest.raises(ValidationError):
            VideoSettings(img_size=5000)

    def test_iterations_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            FlashAvatarSettings(iterations=0)

    def test_target_fps_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            VideoSettings(target_fps=0)

    def test_target_fps_too_large_raises(self) -> None:
        with pytest.raises(ValidationError):
            VideoSettings(target_fps=200)

    def test_smirk_extractor_is_valid(self) -> None:
        cfg = TrainFlashAvatarConfig.model_validate(
            {"pipeline": {"extractor": "smirk"}}
        )
        assert cfg.pipeline.extractor == "smirk"

    def test_target_fps_none_is_valid(self) -> None:
        cfg = VideoSettings(target_fps=None)
        assert cfg.target_fps is None

    def test_max_frames_none_is_valid(self) -> None:
        cfg = VideoSettings(max_frames=None)
        assert cfg.max_frames is None


# ---------------------------------------------------------------------------
# YAML ロードテスト
# ---------------------------------------------------------------------------

class TestTrainFlashAvatarConfigFromYaml:
    """YAML ファイルからの設定読み込みテスト。"""

    def test_load_shipped_yaml(self) -> None:
        """同梱の configs/train_flashavatar.yaml が正常にロードできること。"""
        yaml_path = Path(__file__).parent.parent / "configs" / "train_flashavatar.yaml"
        if not yaml_path.exists():
            pytest.skip("configs/train_flashavatar.yaml が存在しません")
        cfg = TrainFlashAvatarConfig.from_yaml(yaml_path)
        assert cfg.pipeline.extractor in ("deca", "smirk")
        assert cfg.flashavatar.iterations > 0
        assert cfg.video.img_size > 0

    def test_yaml_extractor_smirk(self) -> None:
        """extractor: smirk が YAML から正しくロードされること。"""
        data = {"pipeline": {"extractor": "smirk"}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(data, f)
            tmp_path = f.name
        cfg = TrainFlashAvatarConfig.from_yaml(tmp_path)
        assert cfg.pipeline.extractor == "smirk"

    def test_yaml_partial_override(self) -> None:
        """YAML に一部のフィールドのみ指定した場合もデフォルト値で補完されること。"""
        data = {"flashavatar": {"iterations": 10000}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(data, f)
            tmp_path = f.name
        cfg = TrainFlashAvatarConfig.from_yaml(tmp_path)
        assert cfg.flashavatar.iterations == 10000
        assert cfg.pipeline.extractor == "deca"

    def test_yaml_not_found_raises(self) -> None:
        """存在しない YAML パスで FileNotFoundError が発生すること。"""
        with pytest.raises(FileNotFoundError, match="設定ファイルが見つかりません"):
            TrainFlashAvatarConfig.from_yaml("/no/such/config.yaml")

    def test_yaml_empty_file_uses_defaults(self) -> None:
        """空の YAML ファイルがデフォルト設定として読み込まれること。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            tmp_path = f.name
        cfg = TrainFlashAvatarConfig.from_yaml(tmp_path)
        assert cfg.pipeline.extractor == "deca"
        assert cfg.flashavatar.iterations == 30000


# ---------------------------------------------------------------------------
# active_extractor_settings テスト
# ---------------------------------------------------------------------------

class TestActiveExtractorSettings:
    """active_extractor_settings() の動作テスト。"""

    def test_deca_extractor_returns_deca_settings(self) -> None:
        cfg = TrainFlashAvatarConfig.model_validate({"pipeline": {"extractor": "deca"}})
        s = cfg.active_extractor_settings()
        assert "deca" in s.model_path.lower() or "deca" in s.repo_dir.lower()

    def test_smirk_extractor_returns_smirk_settings(self) -> None:
        cfg = TrainFlashAvatarConfig.model_validate({"pipeline": {"extractor": "smirk"}})
        s = cfg.active_extractor_settings()
        assert "smirk" in s.model_path.lower() or "smirk" in s.repo_dir.lower()


# ---------------------------------------------------------------------------
# video normalization ロジックテスト
# ---------------------------------------------------------------------------

class TestVideoNormalizationLogic:
    """normalize_video の _needs_normalization ロジックをテスト。"""

    def _make_cfg(self, target_fps=25, img_size=512, center_crop=True):
        return TrainFlashAvatarConfig.model_validate({
            "video": {
                "target_fps": target_fps,
                "img_size": img_size,
                "center_crop": center_crop,
            }
        })

    def test_no_normalization_needed_square_correct_fps(self) -> None:
        from scripts.train_flashavatar import _needs_normalization
        cfg = self._make_cfg(target_fps=25, img_size=512, center_crop=True)
        assert _needs_normalization(25.0, 512, 512, cfg) is False

    def test_fps_mismatch_needs_normalization(self) -> None:
        from scripts.train_flashavatar import _needs_normalization
        cfg = self._make_cfg(target_fps=25, img_size=512)
        assert _needs_normalization(30.0, 512, 512, cfg) is True

    def test_non_square_with_center_crop_needs_normalization(self) -> None:
        from scripts.train_flashavatar import _needs_normalization
        cfg = self._make_cfg(target_fps=None, img_size=512, center_crop=True)
        assert _needs_normalization(25.0, 1920, 1080, cfg) is True

    def test_non_square_without_center_crop_still_needs_resize(self) -> None:
        from scripts.train_flashavatar import _needs_normalization
        cfg = self._make_cfg(target_fps=None, img_size=512, center_crop=False)
        assert _needs_normalization(25.0, 1920, 1080, cfg) is True

    def test_target_fps_none_no_fps_normalization(self) -> None:
        from scripts.train_flashavatar import _needs_normalization
        cfg = self._make_cfg(target_fps=None, img_size=512, center_crop=False)
        assert _needs_normalization(60.0, 512, 512, cfg) is False


# ---------------------------------------------------------------------------
# エラーメッセージテスト
# ---------------------------------------------------------------------------

class TestErrorMessages:
    """エラーメッセージ関数の内容テスト。"""

    def test_video_not_found_message(self) -> None:
        from scripts.train_flashavatar import _err_video_not_found
        msg = _err_video_not_found("/no/such/video.mp4")
        assert "/no/such/video.mp4" in msg
        assert "ERROR" in msg

    def test_checkpoint_not_found_message(self) -> None:
        from scripts.train_flashavatar import _err_checkpoint_not_found
        msg = _err_checkpoint_not_found("DECA", "/path/to/model.tar", "bash install/setup_deca.sh")
        assert "DECA" in msg
        assert "setup_deca.sh" in msg

    def test_submodule_not_init_message(self) -> None:
        from scripts.train_flashavatar import _err_submodule_not_init
        msg = _err_submodule_not_init("FlashAvatar", "bash install/setup_flashavatar.sh")
        assert "FlashAvatar" in msg
        assert "setup_flashavatar.sh" in msg
