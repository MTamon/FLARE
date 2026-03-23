"""flare.utils.errors のテスト。

PipelineErrorHandlerの各エラー種別→ErrorPolicyマッピング、
retry_with_backoffデコレータの動作、ErrorPolicy enumの値を検証する。

仕様書8.5節「エラーハンドリングポリシー」に基づく3段階ポリシー:
    SKIP: 顔未検出、特徴抽出失敗 → フレームスキップ
    RETRY: GPU OOM、一時的I/Oエラー → 指数バックオフリトライ
    ABORT: モデルロード失敗、設定不正 → パイプライン停止
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from flare.utils.errors import (
    ConfigError,
    ErrorPolicy,
    FaceNotDetectedError,
    ModelLoadError,
    PipelineErrorHandler,
    retry_with_backoff,
)


class TestPipelineErrorHandler:
    """PipelineErrorHandlerのテストスイート。"""

    @pytest.fixture()
    def handler(self) -> PipelineErrorHandler:
        """PipelineErrorHandlerインスタンスを返すフィクスチャ。

        Returns:
            PipelineErrorHandler。
        """
        return PipelineErrorHandler()

    def test_face_not_detected_returns_skip(
        self, handler: PipelineErrorHandler
    ) -> None:
        """FaceNotDetectedErrorでErrorPolicy.SKIPが返ることを確認する。

        Args:
            handler: テスト対象のエラーハンドラ。
        """
        error: FaceNotDetectedError = FaceNotDetectedError("顔が見つかりません")
        policy: ErrorPolicy = handler.handle(error, {"frame_index": 42})

        assert policy == ErrorPolicy.SKIP

    def test_oom_returns_retry(self, handler: PipelineErrorHandler) -> None:
        """torch.cuda.OutOfMemoryErrorでErrorPolicy.RETRYが返ることを確認する。

        torchのインポートが不可能な環境でも動作するよう、
        _is_cuda_oomをモックする。

        Args:
            handler: テスト対象のエラーハンドラ。
        """
        # OOMエラーを模擬するため _is_cuda_oom をモック
        oom_error: RuntimeError = RuntimeError("CUDA out of memory")

        with patch("flare.utils.errors._is_cuda_oom", return_value=True), \
             patch("flare.utils.errors._safe_cuda_empty_cache"):
            policy: ErrorPolicy = handler.handle(oom_error, {"gpu": 0})

        assert policy == ErrorPolicy.RETRY

    def test_model_load_error_returns_abort(
        self, handler: PipelineErrorHandler
    ) -> None:
        """ModelLoadErrorでErrorPolicy.ABORTが返ることを確認する。

        Args:
            handler: テスト対象のエラーハンドラ。
        """
        error: ModelLoadError = ModelLoadError("モデルファイルが見つかりません")
        policy: ErrorPolicy = handler.handle(error, {"path": "model.pth"})

        assert policy == ErrorPolicy.ABORT

    def test_config_error_returns_abort(
        self, handler: PipelineErrorHandler
    ) -> None:
        """ConfigErrorでErrorPolicy.ABORTが返ることを確認する。

        Args:
            handler: テスト対象のエラーハンドラ。
        """
        error: ConfigError = ConfigError("設定ファイルが不正です")
        policy: ErrorPolicy = handler.handle(error, {"file": "config.yaml"})

        assert policy == ErrorPolicy.ABORT

    def test_unknown_error_returns_skip(
        self, handler: PipelineErrorHandler
    ) -> None:
        """未分類のExceptionでErrorPolicy.SKIPが返ることを確認する。

        リアルタイムモードでは処理継続を最優先とするため、
        未知の例外もSKIPポリシーで処理される。

        Args:
            handler: テスト対象のエラーハンドラ。
        """
        error: ValueError = ValueError("予期しないエラー")
        policy: ErrorPolicy = handler.handle(error, {})

        assert policy == ErrorPolicy.SKIP


class TestRetryWithBackoff:
    """retry_with_backoffデコレータのテストスイート。"""

    def test_retry_with_backoff_success(self) -> None:
        """2回失敗後3回目に成功する関数が正常に値を返すことを確認する。"""
        call_count: int = 0

        @retry_with_backoff(
            max_retries=3, base_delay=0.01, exceptions=(IOError,)
        )
        def flaky_function() -> str:
            """2回失敗後に成功するテスト関数。

            Returns:
                成功時の文字列。
            """
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise IOError("一時的な失敗")
            return "success"

        result: str = flaky_function()

        assert result == "success"
        assert call_count == 3

    def test_retry_with_backoff_exhausted(self) -> None:
        """max_retries回すべて失敗した場合に例外が送出されることを確認する。"""
        call_count: int = 0

        @retry_with_backoff(
            max_retries=2, base_delay=0.01, exceptions=(IOError,)
        )
        def always_fails() -> None:
            """常に失敗するテスト関数。

            Raises:
                IOError: 毎回送出。
            """
            nonlocal call_count
            call_count += 1
            raise IOError("永続的な失敗")

        with pytest.raises(IOError, match="永続的な失敗"):
            always_fails()

        # 初回 + 2リトライ = 3回呼び出し
        assert call_count == 3

    def test_retry_with_backoff_non_matching_exception(self) -> None:
        """リトライ対象外の例外は即座に伝播することを確認する。"""
        call_count: int = 0

        @retry_with_backoff(
            max_retries=3, base_delay=0.01, exceptions=(IOError,)
        )
        def wrong_exception() -> None:
            """リトライ対象外の例外を送出するテスト関数。

            Raises:
                ValueError: IOError以外の例外。
            """
            nonlocal call_count
            call_count += 1
            raise ValueError("リトライ対象外")

        with pytest.raises(ValueError, match="リトライ対象外"):
            wrong_exception()

        # リトライなしで即座に伝播
        assert call_count == 1


class TestErrorPolicyEnum:
    """ErrorPolicy enumのテストスイート。"""

    def test_error_policy_enum_values(self) -> None:
        """SKIP/RETRY/ABORTの3値が正しいことを確認する。"""
        assert ErrorPolicy.SKIP.value == "skip"
        assert ErrorPolicy.RETRY.value == "retry"
        assert ErrorPolicy.ABORT.value == "abort"
        assert len(ErrorPolicy) == 3
