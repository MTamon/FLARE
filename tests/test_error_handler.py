"""PipelineErrorHandler のテスト (Section 8.5)"""

from __future__ import annotations

import pytest
import torch

from flare.utils.errors import (
    ConfigError,
    ErrorPolicy,
    FaceNotDetectedError,
    FeatureExtractionError,
    ModelLoadError,
    PipelineErrorHandler,
    RendererNotInitializedError,
)


@pytest.fixture
def handler() -> PipelineErrorHandler:
    return PipelineErrorHandler()


class TestClassify:
    """ErrorPolicy 分類のテスト。"""

    def test_face_not_detected_is_skip(self, handler):
        assert handler.classify(FaceNotDetectedError()) is ErrorPolicy.SKIP

    def test_feature_extraction_is_skip(self, handler):
        assert handler.classify(FeatureExtractionError()) is ErrorPolicy.SKIP

    def test_cuda_oom_is_retry(self, handler):
        # torch.cuda.OutOfMemoryError は CUDA 環境でしか発生しないが
        # 分類ロジック自体はテスト可能
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        exc = torch.cuda.OutOfMemoryError("test")
        assert handler.classify(exc) is ErrorPolicy.RETRY

    def test_io_error_is_retry(self, handler):
        assert handler.classify(IOError("disk")) is ErrorPolicy.RETRY

    def test_model_load_error_is_abort(self, handler):
        assert handler.classify(ModelLoadError()) is ErrorPolicy.ABORT

    def test_config_error_is_abort(self, handler):
        assert handler.classify(ConfigError()) is ErrorPolicy.ABORT

    def test_unknown_error_is_skip(self, handler):
        """未知の例外はリアルタイム継続優先で SKIP。"""
        assert handler.classify(ValueError("unknown")) is ErrorPolicy.SKIP


class TestExecuteWithPolicy:
    """execute_with_policy のテスト。"""

    def test_success(self, handler):
        result = handler.execute_with_policy(lambda: 42)
        assert result == 42

    def test_skip_returns_none(self, handler):
        def fail():
            raise FaceNotDetectedError("no face")

        result = handler.execute_with_policy(fail)
        assert result is None

    def test_abort_reraises(self, handler):
        def fail():
            raise ModelLoadError("bad model")

        with pytest.raises(ModelLoadError):
            handler.execute_with_policy(fail)

    def test_retry_then_succeed(self, handler):
        """1 回失敗 (IOError=RETRY) → 2 回目で成功。"""
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise IOError("temporary")
            return "ok"

        result = handler.execute_with_policy(
            flaky, backoff_base=0.01  # テスト高速化
        )
        assert result == "ok"
        assert call_count == 2

    def test_retry_exhaustion_falls_back_to_skip(self, handler):
        """リトライ上限超過 → SKIP (None)。"""
        def always_fail():
            raise IOError("persistent")

        result = handler.execute_with_policy(
            always_fail, max_retries=2, backoff_base=0.01
        )
        assert result is None