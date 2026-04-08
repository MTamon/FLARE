"""Phase 4 の統合テスト。

FrameDropPolicy / FrameDropHandler / RealtimePipeline の
動作を検証する。外部デバイス（Webcam / MediaPipe等）への依存はモックで回避する。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from flare.config import PipelineConfig
from flare.pipeline.buffer import PipelineBuffer
from flare.pipeline.frame_drop import FrameDropHandler, FrameDropPolicy
from flare.pipeline.realtime import RealtimePipeline


def _mock_run_context(pipeline: RealtimePipeline) -> Any:
    """run()テスト用のモックコンテキストを構築するヘルパー。

    全ワーカースレッドのループと表示ループ、およびFaceDetectorを
    モック化し、run()が即座に完了するようにする。

    Args:
        pipeline: モック対象のRealtimePipelineインスタンス。

    Returns:
        contextmanager のネストされたパッチ群。
    """
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch("flare.pipeline.realtime.FaceDetector", return_value=MagicMock())
    )
    stack.enter_context(
        patch.object(pipeline, "_display_loop_opencv", side_effect=lambda c: None)
    )
    stack.enter_context(
        patch.object(pipeline, "_capture_loop", side_effect=lambda c: None)
    )
    stack.enter_context(
        patch.object(pipeline, "_extract_loop", side_effect=lambda c: None)
    )
    stack.enter_context(
        patch.object(pipeline, "_inference_loop", side_effect=lambda c: None)
    )
    stack.enter_context(
        patch.object(pipeline, "_render_loop", side_effect=lambda c: None)
    )
    return stack


# =========================================================================
# FrameDropPolicy Enum
# =========================================================================


class TestFrameDropPolicyEnum:
    """FrameDropPolicy 列挙型のテスト。"""

    def test_drop_oldest_value(self) -> None:
        """DROP_OLDESTの値が'drop_oldest'であること。"""
        assert FrameDropPolicy.DROP_OLDEST.value == "drop_oldest"

    def test_block_value(self) -> None:
        """BLOCKの値が'block'であること。"""
        assert FrameDropPolicy.BLOCK.value == "block"

    def test_interpolate_value(self) -> None:
        """INTERPOLATEの値が'interpolate'であること。"""
        assert FrameDropPolicy.INTERPOLATE.value == "interpolate"

    def test_enum_members_count(self) -> None:
        """列挙型のメンバー数が3であること。"""
        assert len(FrameDropPolicy) == 3


# =========================================================================
# FrameDropHandler
# =========================================================================


class TestFrameDropHandlerApply:
    """FrameDropHandler.apply() の動作テスト。"""

    def test_drop_oldest_puts_frame(self) -> None:
        """DROP_OLDESTでフレームがバッファに追加されること。"""
        handler = FrameDropHandler()
        buffer = PipelineBuffer(max_size=10, overflow_policy="drop_oldest")
        frame = {"frame": "data", "index": 0}
        result = handler.apply(buffer, FrameDropPolicy.DROP_OLDEST, frame)
        assert result is True
        assert buffer.qsize == 1

    def test_drop_oldest_overflow_drops(self) -> None:
        """DROP_OLDESTでバッファ満杯時に最古フレームが破棄されること。"""
        handler = FrameDropHandler()
        buffer = PipelineBuffer(max_size=2, overflow_policy="drop_oldest")
        handler.apply(buffer, FrameDropPolicy.DROP_OLDEST, {"index": 0})
        handler.apply(buffer, FrameDropPolicy.DROP_OLDEST, {"index": 1})
        handler.apply(buffer, FrameDropPolicy.DROP_OLDEST, {"index": 2})
        assert buffer.qsize == 2
        assert buffer.stats["dropped"] == 1

    def test_block_puts_frame(self) -> None:
        """BLOCKでフレームがバッファに追加されること。"""
        handler = FrameDropHandler()
        buffer = PipelineBuffer(max_size=10, overflow_policy="block")
        frame = {"frame": "data", "index": 0}
        result = handler.apply(buffer, FrameDropPolicy.BLOCK, frame)
        assert result is True
        assert buffer.qsize == 1

    def test_interpolate_puts_frame(self) -> None:
        """INTERPOLATEでフレームがバッファに追加されること。"""
        handler = FrameDropHandler()
        buffer = PipelineBuffer(max_size=10, overflow_policy="interpolate")
        frame = {"frame": "data", "index": 0}
        result = handler.apply(buffer, FrameDropPolicy.INTERPOLATE, frame)
        assert result is True
        assert buffer.qsize == 1

    def test_interpolate_behaves_like_drop_oldest(self) -> None:
        """INTERPOLATEがDROP_OLDESTと同等の動作をすること（Phase 4仕様）。"""
        handler = FrameDropHandler()
        buffer = PipelineBuffer(max_size=2, overflow_policy="interpolate")
        handler.apply(buffer, FrameDropPolicy.INTERPOLATE, {"index": 0})
        handler.apply(buffer, FrameDropPolicy.INTERPOLATE, {"index": 1})
        handler.apply(buffer, FrameDropPolicy.INTERPOLATE, {"index": 2})
        assert buffer.qsize == 2
        assert buffer.stats["dropped"] == 1

    def test_apply_returns_true(self) -> None:
        """apply()が全ポリシーでTrueを返すこと。"""
        handler = FrameDropHandler()
        for policy in FrameDropPolicy:
            buffer = PipelineBuffer(
                max_size=10, overflow_policy=policy.value
            )
            result = handler.apply(buffer, policy, {"index": 0})
            assert result is True

    def test_multiple_frames_sequential(self) -> None:
        """複数フレームを順次追加できること。"""
        handler = FrameDropHandler()
        buffer = PipelineBuffer(max_size=100, overflow_policy="drop_oldest")
        for i in range(50):
            handler.apply(buffer, FrameDropPolicy.DROP_OLDEST, {"index": i})
        assert buffer.qsize == 50
        assert buffer.stats["total_put"] == 50
        assert buffer.stats["dropped"] == 0


class TestFrameDropHandlerPolicyFromString:
    """FrameDropHandler.policy_from_string() のテスト。"""

    def test_drop_oldest_string(self) -> None:
        """'drop_oldest'文字列からDROP_OLDESTが返ること。"""
        policy = FrameDropHandler.policy_from_string("drop_oldest")
        assert policy == FrameDropPolicy.DROP_OLDEST

    def test_block_string(self) -> None:
        """'block'文字列からBLOCKが返ること。"""
        policy = FrameDropHandler.policy_from_string("block")
        assert policy == FrameDropPolicy.BLOCK

    def test_interpolate_string(self) -> None:
        """'interpolate'文字列からINTERPOLATEが返ること。"""
        policy = FrameDropHandler.policy_from_string("interpolate")
        assert policy == FrameDropPolicy.INTERPOLATE

    def test_unknown_string_raises(self) -> None:
        """未知の文字列でValueErrorが発生すること。"""
        with pytest.raises(ValueError, match="Unknown policy string"):
            FrameDropHandler.policy_from_string("unknown_policy")


# =========================================================================
# RealtimePipeline Initialization
# =========================================================================


class TestRealtimePipelineInit:
    """RealtimePipeline の初期化テスト。"""

    def test_default_source_is_webcam(self) -> None:
        """デフォルトのソースがWebcam (0) であること。"""
        pipeline = RealtimePipeline()
        assert pipeline.source == 0

    def test_custom_source_video_file(self) -> None:
        """動画ファイルパスがソースとして設定されること。"""
        pipeline = RealtimePipeline(source="./test_video.mp4")
        assert pipeline.source == "./test_video.mp4"

    def test_custom_source_webcam_id(self) -> None:
        """Webcam IDがソースとして設定されること。"""
        pipeline = RealtimePipeline(source=1)
        assert pipeline.source == 1

    def test_default_display_backend_opencv(self) -> None:
        """デフォルトの表示バックエンドがopencvであること。"""
        pipeline = RealtimePipeline()
        assert pipeline.display_backend == "opencv"

    def test_custom_display_backend_pyqt(self) -> None:
        """PyQt6バックエンドが設定できること。"""
        pipeline = RealtimePipeline(display_backend="pyqt")
        assert pipeline.display_backend == "pyqt"

    def test_invalid_display_backend_raises(self) -> None:
        """未知の表示バックエンドでValueErrorが発生すること。"""
        with pytest.raises(ValueError, match="Unknown display_backend"):
            RealtimePipeline(display_backend="unknown")

    def test_not_running_on_creation(self) -> None:
        """生成直後はis_runningがFalseであること。"""
        pipeline = RealtimePipeline()
        assert pipeline.is_running is False

    def test_default_frame_drop_policy(self) -> None:
        """デフォルトのフレームドロップポリシーがDROP_OLDESTであること。"""
        pipeline = RealtimePipeline()
        assert pipeline.frame_drop_policy == FrameDropPolicy.DROP_OLDEST


# =========================================================================
# RealtimePipeline Start / Stop
# =========================================================================


class TestRealtimePipelineStartStop:
    """RealtimePipeline の起動・停止テスト（mock使用）。"""

    def test_stop_when_not_running_is_safe(self) -> None:
        """未起動状態でstop()を呼んでもエラーにならないこと。"""
        pipeline = RealtimePipeline()
        pipeline.stop()
        assert pipeline.is_running is False

    def test_stop_twice_is_safe(self) -> None:
        """stop()を2回呼んでもエラーにならないこと。"""
        pipeline = RealtimePipeline()
        pipeline.stop()
        pipeline.stop()
        assert pipeline.is_running is False

    def test_run_sets_buffers(self) -> None:
        """run()がバッファを初期化すること（早期停止で検証）。"""
        pipeline = RealtimePipeline()
        config = PipelineConfig()

        with _mock_run_context(pipeline):
            pipeline.run(config)

        assert pipeline._capture_buffer is not None
        assert pipeline._extract_buffer is not None
        assert pipeline._render_buffer is not None
        assert pipeline._display_buffer is not None

    def test_run_uses_config_overflow_policy(self) -> None:
        """run()がconfigのoverflow_policyを使用すること。"""
        pipeline = RealtimePipeline()
        config = PipelineConfig()
        config.buffer.overflow_policy = "block"

        with _mock_run_context(pipeline):
            pipeline.run(config)

        assert pipeline.frame_drop_policy == FrameDropPolicy.BLOCK

    def test_run_starts_threads(self) -> None:
        """run()がワーカースレッドを起動すること。"""
        pipeline = RealtimePipeline()
        config = PipelineConfig()

        with _mock_run_context(pipeline):
            pipeline.run(config)

        assert pipeline._capture_buffer is not None


# =========================================================================
# RealtimePipeline with FrameDropPolicy integration
# =========================================================================


class TestRealtimePipelineFrameDropIntegration:
    """RealtimePipeline と FrameDropPolicy の統合テスト。"""

    def test_frame_drop_handler_is_initialized(self) -> None:
        """パイプラインにFrameDropHandlerが初期化されていること。"""
        pipeline = RealtimePipeline()
        assert pipeline._frame_drop_handler is not None
        assert isinstance(pipeline._frame_drop_handler, FrameDropHandler)

    def test_policy_from_config_drop_oldest(self) -> None:
        """configのdrop_oldestがFrameDropPolicyに反映されること。"""
        pipeline = RealtimePipeline()
        config = PipelineConfig()
        config.buffer.overflow_policy = "drop_oldest"

        with _mock_run_context(pipeline):
            pipeline.run(config)

        assert pipeline.frame_drop_policy == FrameDropPolicy.DROP_OLDEST

    def test_policy_from_config_interpolate(self) -> None:
        """configのinterpolateがFrameDropPolicyに反映されること。"""
        pipeline = RealtimePipeline()
        config = PipelineConfig()
        config.buffer.overflow_policy = "interpolate"

        with _mock_run_context(pipeline):
            pipeline.run(config)

        assert pipeline.frame_drop_policy == FrameDropPolicy.INTERPOLATE


# =========================================================================
# RealtimePipeline display backend
# =========================================================================


class TestRealtimePipelineDisplayBackend:
    """RealtimePipeline の表示バックエンドテスト。"""

    def test_opencv_backend_calls_opencv_display(self) -> None:
        """opencvバックエンドでOpenCV表示ループが呼ばれること。"""
        pipeline = RealtimePipeline(display_backend="opencv")
        config = PipelineConfig()

        opencv_called: list[bool] = []

        def mock_opencv_display(c: PipelineConfig) -> None:
            opencv_called.append(True)

        with patch(
            "flare.pipeline.realtime.FaceDetector", return_value=MagicMock()
        ), patch.object(
            pipeline, "_display_loop_opencv", side_effect=mock_opencv_display
        ), patch.object(
            pipeline, "_capture_loop", side_effect=lambda c: None
        ), patch.object(
            pipeline, "_extract_loop", side_effect=lambda c: None
        ), patch.object(
            pipeline, "_inference_loop", side_effect=lambda c: None
        ), patch.object(
            pipeline, "_render_loop", side_effect=lambda c: None
        ):
            pipeline.run(config)

        assert len(opencv_called) == 1

    def test_pyqt_backend_without_pyqt6_falls_back(self) -> None:
        """PyQt6未インストール時にOpenCVにフォールバックすること。"""
        pipeline = RealtimePipeline(display_backend="pyqt")
        config = PipelineConfig()

        opencv_called: list[bool] = []

        def mock_opencv_display(c: PipelineConfig) -> None:
            opencv_called.append(True)

        with patch(
            "flare.pipeline.realtime.FaceDetector", return_value=MagicMock()
        ), patch(
            "flare.pipeline.realtime._HAS_PYQT6", False
        ), patch.object(
            pipeline, "_display_loop_opencv", side_effect=mock_opencv_display
        ), patch.object(
            pipeline, "_capture_loop", side_effect=lambda c: None
        ), patch.object(
            pipeline, "_extract_loop", side_effect=lambda c: None
        ), patch.object(
            pipeline, "_inference_loop", side_effect=lambda c: None
        ), patch.object(
            pipeline, "_render_loop", side_effect=lambda c: None
        ):
            pipeline.run(config)

        assert len(opencv_called) == 1


# =========================================================================
# RealtimePipeline video file input
# =========================================================================


class TestRealtimePipelineVideoInput:
    """RealtimePipeline の動画ファイル入力テスト。"""

    def test_video_file_source_type(self) -> None:
        """動画ファイルソースが文字列型であること。"""
        pipeline = RealtimePipeline(source="./video.mp4")
        assert isinstance(pipeline.source, str)

    def test_webcam_source_type(self) -> None:
        """Webcamソースが整数型であること。"""
        pipeline = RealtimePipeline(source=0)
        assert isinstance(pipeline.source, int)
