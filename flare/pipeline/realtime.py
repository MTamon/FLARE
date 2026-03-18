"""リアルタイムパイプライン骨格 (Section 6)

Section 6 の設計:
  - Webcam → Face Detection → 3DMM Extraction → LHG Model → Renderer → Display
  - マルチスレッド (Section 6.2): キャプチャ / 抽出 / 推論 / レンダリング / 表示
  - マルチ GPU 配置 (Section 6.3): device_map で 1-3 GPU 対応
  - フレームドロップ戦略 (Section 6.4): drop_oldest デフォルト
  - 速度目標: Route B で 80-100+ FPS、最低 10 FPS (Section 1.2)

Phase 1 では骨格のみ定義。Phase 4 でマルチスレッド実装を行う。
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from loguru import logger

from flare.config import ToolConfig
from flare.pipeline.buffer import PipelineBuffer
from flare.utils.errors import PipelineErrorHandler
from flare.utils.metrics import FPSCounter, LatencyTracker


class RealtimePipeline:
    """リアルタイムパイプライン。

    Section 6.2 のマルチスレッド設計:
      - キャプチャスレッド:   Webcam フレーム取得 (CPU)
      - 抽出スレッド:         顔検出 + 3DMM パラメータ抽出 (GPU)
      - 推論スレッド:         LHG モデル推論 (GPU)
      - レンダリングスレッド: 画像生成 (GPU)
      - 表示スレッド:         GUI 表示 (CPU)

    Phase 1 ではインターフェースのみ。Phase 4 で実装する。
    """

    def __init__(self, config: ToolConfig) -> None:
        self._config = config
        self._error_handler = PipelineErrorHandler()
        self._tracker = LatencyTracker()
        self._fps_counter = FPSCounter()
        self._running = False
        self._stop_event = threading.Event()

        # Section 8.4: ステージ間バッファ
        buffer_cfg = config.buffer
        self._capture_to_extract = PipelineBuffer(
            max_size=buffer_cfg.max_size,
            timeout=buffer_cfg.timeout_sec,
            overflow_policy=buffer_cfg.overflow_policy,  # type: ignore[arg-type]
        )
        self._extract_to_infer = PipelineBuffer(
            max_size=buffer_cfg.max_size,
            timeout=buffer_cfg.timeout_sec,
            overflow_policy=buffer_cfg.overflow_policy,  # type: ignore[arg-type]
        )
        self._infer_to_render = PipelineBuffer(
            max_size=buffer_cfg.max_size,
            timeout=buffer_cfg.timeout_sec,
            overflow_policy=buffer_cfg.overflow_policy,  # type: ignore[arg-type]
        )
        self._render_to_display = PipelineBuffer(
            max_size=buffer_cfg.max_size,
            timeout=buffer_cfg.timeout_sec,
            overflow_policy=buffer_cfg.overflow_policy,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # ライフサイクル
    # ------------------------------------------------------------------

    def start(self, camera_id: int = 0) -> None:
        """パイプラインを開始する。

        Args:
            camera_id: Webcam デバイス番号。
        """
        if self._running:
            logger.warning("RealtimePipeline is already running.")
            return

        logger.info(
            f"Starting RealtimePipeline: camera={camera_id}, "
            f"device_map={self._config.device_map}"
        )
        self._running = True
        self._stop_event.clear()

        # TODO (Phase 4): 各スレッドを起動
        #   - _capture_thread(camera_id)
        #   - _extract_thread()
        #   - _infer_thread()
        #   - _render_thread()
        #   - _display_thread()
        raise NotImplementedError(
            "RealtimePipeline.start is a Phase 1 skeleton. "
            "Implement in Phase 4."
        )

    def stop(self) -> None:
        """パイプラインを安全に停止する。"""
        if not self._running:
            return
        logger.info("Stopping RealtimePipeline...")
        self._stop_event.set()
        self._running = False

        # 統計ログ
        self._tracker.log_summary()
        for name, buf in [
            ("capture→extract", self._capture_to_extract),
            ("extract→infer", self._extract_to_infer),
            ("infer→render", self._infer_to_render),
            ("render→display", self._render_to_display),
        ]:
            s = buf.stats
            logger.info(f"Buffer [{name}]: {s}")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # スレッドエントリポイント (Phase 4 で実装)
    # ------------------------------------------------------------------

    def _capture_loop(self, camera_id: int) -> None:
        """キャプチャスレッド: Webcam フレーム取得 → capture_to_extract バッファ。

        Section 6.1: cv2.VideoCapture, ~0.5ms。
        """
        # TODO (Phase 4)
        pass

    def _extract_loop(self) -> None:
        """抽出スレッド: 顔検出 + 3DMM パラメータ抽出。

        Section 6.1: face_detect.py (~3-15ms) + Extractor (~8-20ms)。
        デバイス: device_map.extractor。
        """
        # TODO (Phase 4)
        pass

    def _infer_loop(self) -> None:
        """推論スレッド: LHG モデル推論。

        デバイス: device_map.lhg_model。
        requires_window に基づきバッファリング戦略を切り替える。
        """
        # TODO (Phase 4)
        pass

    def _render_loop(self) -> None:
        """レンダリングスレッド: 画像生成。

        Section 6.1: FlashAvatar ~3.3ms (300 FPS) / PIRender ~33-50ms。
        デバイス: device_map.renderer。
        """
        # TODO (Phase 4)
        pass

    def _display_loop(self) -> None:
        """表示スレッド: GUI 表示 (cv2.imshow / PyQt)。CPU のみ。"""
        # TODO (Phase 4)
        pass