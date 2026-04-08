"""PipelineErrorHandler の各エラー種別→ErrorPolicy マッピングテスト。

仕様書§8.5の分岐ロジックを検証する。
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from flare.utils.errors import (
    ConfigError,
    ErrorPolicy,
    FaceNotDetectedError,
    ModelLoadError,
    PipelineErrorHandler,
    RendererNotInitializedError,
)


class TestErrorPolicyMapping:
    """エラー種別→ErrorPolicy のマッピングテスト。"""

    def test_face_not_detected_returns_skip(self) -> None:
        """FaceNotDetectedError → SKIP であること。"""
        handler = PipelineErrorHandler()
        policy = handler.handle(
            FaceNotDetectedError("no face"), {"frame": 0}
        )
        assert policy == ErrorPolicy.SKIP

    def test_model_load_error_returns_abort(self) -> None:
        """ModelLoadError → ABORT であること。"""
        handler = PipelineErrorHandler()
        policy = handler.handle(
            ModelLoadError("bad weights"), {"module": "extractor"}
        )
        assert policy == ErrorPolicy.ABORT

    def test_config_error_returns_abort(self) -> None:
        """ConfigError → ABORT であること。"""
        handler = PipelineErrorHandler()
        policy = handler.handle(
            ConfigError("invalid yaml"), {"file": "config.yaml"}
        )
        assert policy == ErrorPolicy.ABORT

    def test_unknown_error_returns_skip(self) -> None:
        """未知のエラー → SKIP であること（リアルタイムでは継続優先）。"""
        handler = PipelineErrorHandler()
        policy = handler.handle(
            RuntimeError("unexpected"), {"frame": 42}
        )
        assert policy == ErrorPolicy.SKIP

    def test_value_error_returns_skip(self) -> None:
        """ValueError → SKIP であること。"""
        handler = PipelineErrorHandler()
        policy = handler.handle(ValueError("bad value"), {})
        assert policy == ErrorPolicy.SKIP

    def test_oom_returns_retry(self) -> None:
        """torch.cuda.OutOfMemoryError → RETRY であること。"""
        handler = PipelineErrorHandler()
        oom = torch.cuda.OutOfMemoryError("CUDA OOM")
        policy = handler.handle(oom, {"frame": 10})
        assert policy == ErrorPolicy.RETRY


class TestOOMHandling:
    """GPU OOM 時の torch.cuda.empty_cache() 呼び出しテスト。"""

    def test_oom_calls_empty_cache(self) -> None:
        """OOM時にtorch.cuda.empty_cache()が呼ばれること。"""
        handler = PipelineErrorHandler()
        oom = torch.cuda.OutOfMemoryError("CUDA OOM")

        with patch("flare.utils.errors.torch.cuda.empty_cache") as mock_clear:
            policy = handler.handle(oom, {"frame": 0})

        mock_clear.assert_called_once()
        assert policy == ErrorPolicy.RETRY


class TestErrorPolicyEnum:
    """ErrorPolicy Enum のテスト。"""

    def test_skip_value(self) -> None:
        """SKIPの値が'skip'であること。"""
        assert ErrorPolicy.SKIP.value == "skip"

    def test_retry_value(self) -> None:
        """RETRYの値が'retry'であること。"""
        assert ErrorPolicy.RETRY.value == "retry"

    def test_abort_value(self) -> None:
        """ABORTの値が'abort'であること。"""
        assert ErrorPolicy.ABORT.value == "abort"

    def test_enum_members_count(self) -> None:
        """ErrorPolicyが3つのメンバーを持つこと。"""
        assert len(ErrorPolicy) == 3


class TestCustomExceptions:
    """カスタム例外クラスのテスト。"""

    def test_face_not_detected_is_exception(self) -> None:
        """FaceNotDetectedErrorがExceptionのサブクラスであること。"""
        assert issubclass(FaceNotDetectedError, Exception)

    def test_model_load_error_is_exception(self) -> None:
        """ModelLoadErrorがExceptionのサブクラスであること。"""
        assert issubclass(ModelLoadError, Exception)

    def test_config_error_is_exception(self) -> None:
        """ConfigErrorがExceptionのサブクラスであること。"""
        assert issubclass(ConfigError, Exception)

    def test_renderer_not_initialized_is_exception(self) -> None:
        """RendererNotInitializedErrorがExceptionのサブクラスであること。"""
        assert issubclass(RendererNotInitializedError, Exception)

    def test_exception_message_preserved(self) -> None:
        """例外メッセージが保持されること。"""
        msg = "test error message"
        for exc_cls in [
            FaceNotDetectedError,
            ModelLoadError,
            ConfigError,
            RendererNotInitializedError,
        ]:
            exc = exc_cls(msg)
            assert str(exc) == msg
