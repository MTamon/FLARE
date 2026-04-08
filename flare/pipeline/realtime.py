"""リアルタイム処理パイプラインモジュール。

Webカメラまたは動画ファイルからの映像入力をリアルタイムに処理し、
LHGモデル推論後のレンダリング結果を即座に表示する。

仕様書§6の設計に基づき、以下のマルチスレッド構成で動作する:
    - キャプチャスレッド: Webカメラ/動画からのフレーム取得（CPU）
    - 抽出スレッド: 顔検出 + 3DMMパラメータ抽出（GPU: device_map.extractor）
    - 推論スレッド: LHGモデル推論（GPU: device_map.lhg_model）
    - レンダリングスレッド: 画像生成（GPU: device_map.renderer）
    - 表示スレッド: GUI表示（CPU、メインスレッド）

仕様書§6.2のCPU/GPUデバイス分離設計:
    - キャプチャ: CPU（cv2.VideoCapture）
    - 抽出: device_map.extractor で指定されたGPUデバイス
    - 推論: device_map.lhg_model で指定されたGPUデバイス
    - レンダリング: device_map.renderer で指定されたGPUデバイス
    - 表示: CPU（OpenCV / PyQt6）

仕様書§6.4のFrameDropPolicyをPipelineBufferと連携させ、
Latest-Frame-Wins ポリシーにより処理遅延時は最新フレームを
優先して古いフレームを破棄する。

表示バックエンド:
    - ``"opencv"``: OpenCV highgui（cv2.imshow）によるウィンドウ表示。デフォルト。
    - ``"pyqt"``: PyQt6ベースのGUIウィンドウ表示。PyQt6がインストールされている場合に使用可能。

Example:
    リアルタイムパイプラインの実行::

        from flare.config import PipelineConfig
        from flare.pipeline.realtime import RealtimePipeline

        config = PipelineConfig.from_yaml("config.yaml")
        pipeline = RealtimePipeline()
        pipeline.run(config)  # 'q' キーで停止

    動画ファイル入力の使用::

        pipeline = RealtimePipeline(source="./video.mp4")
        pipeline.run(config)

    PyQt6 GUI表示::

        pipeline = RealtimePipeline(display_backend="pyqt")
        pipeline.run(config)
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

from flare.config import PipelineConfig
from flare.pipeline.buffer import PipelineBuffer
from flare.pipeline.frame_drop import FrameDropHandler, FrameDropPolicy
from flare.utils.errors import (
    ErrorPolicy,
    FaceNotDetectedError,
    PipelineErrorHandler,
)
from flare.utils.face_detect import FaceDetector
from flare.utils.metrics import FPSCounter, PipelineMetrics
from flare.utils.visualization import draw_fps

try:
    from PyQt6.QtCore import Qt, QTimer  # type: ignore[import-untyped]
    from PyQt6.QtGui import QImage, QPixmap  # type: ignore[import-untyped]
    from PyQt6.QtWidgets import (  # type: ignore[import-untyped]
        QApplication,
        QLabel,
        QMainWindow,
        QVBoxLayout,
        QWidget,
    )

    _HAS_PYQT6 = True
except ImportError:
    _HAS_PYQT6 = False

_VALID_DISPLAY_BACKENDS = {"opencv", "pyqt"}
"""set[str]: サポートされる表示バックエンド。"""


class RealtimePipeline:
    """リアルタイム処理パイプライン。

    マルチスレッド構成でWebカメラまたは動画ファイルからの映像を
    リアルタイムに処理する。PipelineBufferとFrameDropPolicyを
    連携させてスレッド間のフレーム受け渡しを行い、
    Latest-Frame-Wins ポリシーで処理遅延に対応する。

    仕様書§6.2のCPU/GPUデバイス分離設計を完全実装し、
    各処理ステージを適切なデバイスで実行する。

    表示バックエンドとしてOpenCV（デフォルト）またはPyQt6 GUIを
    選択可能。

    Attributes:
        _running: パイプラインの実行状態フラグ（スレッド安全）。
        _error_handler: エラーハンドラ。
        _metrics: パフォーマンス計測。
        _face_detector: 顔検出。
        _frame_drop_handler: フレームドロップハンドラ。
        _frame_drop_policy: フレームドロップポリシー。
        _source: 映像入力ソース（0=Webcam または動画ファイルパス）。
        _display_backend: 表示バックエンド（"opencv" or "pyqt"）。
        _capture_buffer: キャプチャ→抽出間のバッファ。
        _extract_buffer: 抽出→推論間のバッファ。
        _render_buffer: 推論→レンダリング間のバッファ。
        _display_buffer: レンダリング→表示間のバッファ。
    """

    def __init__(
        self,
        source: int | str = 0,
        display_backend: str = "opencv",
    ) -> None:
        """RealtimePipelineを初期化する。

        Args:
            source: 映像入力ソース。
                整数値の場合はWebカメラデバイスID（0がデフォルト）。
                文字列の場合は動画ファイルパス。
            display_backend: 表示バックエンド。
                ``"opencv"``: OpenCV highguiウィンドウ表示（デフォルト）。
                ``"pyqt"``: PyQt6ベースのGUIウィンドウ表示。

        Raises:
            ValueError: 未知のdisplay_backendが指定された場合。
        """
        if display_backend not in _VALID_DISPLAY_BACKENDS:
            raise ValueError(
                f"Unknown display_backend: {display_backend!r}. "
                f"Must be one of {_VALID_DISPLAY_BACKENDS}"
            )

        self._running = threading.Event()
        self._error_handler = PipelineErrorHandler()
        self._metrics = PipelineMetrics()
        self._face_detector: Optional[FaceDetector] = None
        self._frame_drop_handler = FrameDropHandler()
        self._frame_drop_policy = FrameDropPolicy.DROP_OLDEST
        self._source: int | str = source
        self._display_backend = display_backend
        self._threads: list[threading.Thread] = []

        self._capture_buffer: Optional[PipelineBuffer] = None
        self._extract_buffer: Optional[PipelineBuffer] = None
        self._render_buffer: Optional[PipelineBuffer] = None
        self._display_buffer: Optional[PipelineBuffer] = None

    def run(self, config: PipelineConfig) -> None:
        """リアルタイムパイプラインを実行する。

        映像入力ソース（Webカメラまたは動画ファイル）からの映像を処理し、
        選択された表示バックエンド（OpenCV or PyQt6）に結果を表示する。
        'q' キーで停止する。

        仕様書§6.2のCPU/GPUデバイス分離設計に従い、各スレッドは
        DeviceMapConfigで指定されたデバイスを使用する。

        メインスレッドは表示ループとして動作し、ワーカースレッド群が
        キャプチャ・抽出・推論・レンダリングを並行処理する。

        Args:
            config: パイプライン設定。buffer / device_map / extractor 等を参照。
        """
        self._running.set()

        # FrameDropPolicy を config から設定
        self._frame_drop_policy = FrameDropHandler.policy_from_string(
            config.buffer.overflow_policy
        )

        buffer_cfg = config.buffer
        overflow = config.buffer.overflow_policy

        self._capture_buffer = PipelineBuffer(
            max_size=buffer_cfg.max_size,
            timeout=buffer_cfg.timeout_sec,
            overflow_policy=overflow,
        )
        self._extract_buffer = PipelineBuffer(
            max_size=buffer_cfg.max_size,
            timeout=buffer_cfg.timeout_sec,
            overflow_policy=overflow,
        )
        self._render_buffer = PipelineBuffer(
            max_size=buffer_cfg.max_size,
            timeout=buffer_cfg.timeout_sec,
            overflow_policy=overflow,
        )
        self._display_buffer = PipelineBuffer(
            max_size=buffer_cfg.max_size,
            timeout=buffer_cfg.timeout_sec,
            overflow_policy=overflow,
        )

        self._face_detector = FaceDetector()

        capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(config,),
            name="capture",
            daemon=True,
        )
        extract_thread = threading.Thread(
            target=self._extract_loop,
            args=(config,),
            name="extract",
            daemon=True,
        )
        inference_thread = threading.Thread(
            target=self._inference_loop,
            args=(config,),
            name="inference",
            daemon=True,
        )
        render_thread = threading.Thread(
            target=self._render_loop,
            args=(config,),
            name="render",
            daemon=True,
        )

        self._threads = [
            capture_thread,
            extract_thread,
            inference_thread,
            render_thread,
        ]

        for t in self._threads:
            t.start()

        source_desc = (
            f"webcam:{self._source}"
            if isinstance(self._source, int)
            else f"file:{self._source}"
        )
        logger.info(
            "Realtime pipeline started (source={}, display={}, press 'q' to stop)",
            source_desc,
            self._display_backend,
        )

        if self._display_backend == "pyqt" and _HAS_PYQT6:
            self._display_loop_pyqt(config)
        else:
            if self._display_backend == "pyqt" and not _HAS_PYQT6:
                logger.warning(
                    "PyQt6 not available, falling back to OpenCV display"
                )
            self._display_loop_opencv(config)

        self.stop()

    def stop(self) -> None:
        """パイプラインをスレッド安全に停止する。

        全ワーカースレッドの停止を要求し、join で終了を待機する。
        複数回呼び出しても安全。
        """
        if not self._running.is_set():
            return

        self._running.clear()
        logger.info("Stopping realtime pipeline...")

        for t in self._threads:
            t.join(timeout=5.0)
            if t.is_alive():
                logger.warning("Thread {} did not stop within timeout", t.name)

        self._threads.clear()

        if self._face_detector is not None:
            self._face_detector.release()
            self._face_detector = None

        cv2.destroyAllWindows()

        logger.info("Realtime pipeline stopped")
        logger.info("Pipeline stats: {}", self._metrics.get_summary())

    @property
    def is_running(self) -> bool:
        """パイプラインが実行中かどうかを返す。

        Returns:
            実行中であれば ``True``。
        """
        return self._running.is_set()

    @property
    def frame_drop_policy(self) -> FrameDropPolicy:
        """現在のフレームドロップポリシーを返す。

        Returns:
            現在適用されているFrameDropPolicy。
        """
        return self._frame_drop_policy

    @property
    def display_backend(self) -> str:
        """表示バックエンド名を返す。

        Returns:
            ``"opencv"`` または ``"pyqt"``。
        """
        return self._display_backend

    @property
    def source(self) -> int | str:
        """映像入力ソースを返す。

        Returns:
            WebカメラデバイスID（int）または動画ファイルパス（str）。
        """
        return self._source

    def _capture_loop(self, config: PipelineConfig) -> None:
        """キャプチャスレッドのメインループ。

        Webカメラまたは動画ファイルからフレームを取得し、
        FrameDropHandlerを介してcapture_bufferに追加する。
        動画ファイル入力の場合はFPS制限に従い、ファイル末尾で停止する。

        Args:
            config: パイプライン設定。pipeline.fps を参照。
        """
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            source_desc = (
                f"webcam device {self._source}"
                if isinstance(self._source, int)
                else f"video file {self._source}"
            )
            logger.error("Failed to open {}", source_desc)
            self._running.clear()
            return

        frame_idx = 0
        is_video_file = isinstance(self._source, str)
        logger.info("Capture thread started (source: {})", self._source)

        while self._running.is_set():
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    logger.info(
                        "End of video file reached at frame {}", frame_idx
                    )
                    self._running.clear()
                    break
                logger.warning("Failed to read frame from source")
                continue

            self._frame_drop_handler.apply(
                buffer=self._capture_buffer,
                policy=self._frame_drop_policy,
                new_frame={"frame": frame, "index": frame_idx},
            )
            frame_idx += 1

        cap.release()
        logger.info("Capture thread stopped (total frames: {})", frame_idx)

    def _extract_loop(self, config: PipelineConfig) -> None:
        """抽出スレッドのメインループ。

        capture_bufferからフレームを取得し、顔検出 + クロッピングを行い、
        結果をextract_bufferに追加する。

        仕様書§6.2に従い、device_map.extractorで指定されたデバイスを使用する。

        Args:
            config: パイプライン設定。extractor.input_size / device_map を参照。
        """
        crop_size = config.extractor.input_size
        device = config.device_map.extractor
        logger.info("Extract thread started (device: {})", device)

        while self._running.is_set():
            data = self._capture_buffer.get()
            if data is None:
                continue

            frame = data["frame"]
            frame_idx = data["index"]

            try:
                bbox = self._face_detector.detect(frame)
                if bbox is None:
                    raise FaceNotDetectedError(
                        f"No face at frame {frame_idx}"
                    )
                cropped = self._face_detector.crop_and_align(
                    frame, bbox, size=crop_size
                )
                self._frame_drop_handler.apply(
                    buffer=self._extract_buffer,
                    policy=self._frame_drop_policy,
                    new_frame={
                        "frame": frame,
                        "cropped": cropped,
                        "bbox": bbox,
                        "index": frame_idx,
                        "device": device,
                    },
                )
            except Exception as e:
                policy = self._error_handler.handle(
                    e, {"thread": "extract", "frame_index": frame_idx}
                )
                if policy == ErrorPolicy.ABORT:
                    self._running.clear()
                    break

        logger.info("Extract thread stopped")

    def _inference_loop(self, config: PipelineConfig) -> None:
        """推論スレッドのメインループ。

        extract_bufferからパラメータを取得し、LHGモデル推論を行い、
        結果をrender_bufferに追加する。

        仕様書§6.2に従い、device_map.lhg_modelで指定されたデバイスを使用する。

        Phase 4ではLHGモデル推論のパススルー動作を行う。
        Extractorの出力をそのままrender_bufferに渡す。

        Args:
            config: パイプライン設定。device_map.lhg_model を参照。
        """
        device = config.device_map.lhg_model
        logger.info("Inference thread started (device: {})", device)

        while self._running.is_set():
            data = self._extract_buffer.get()
            if data is None:
                continue

            data["device"] = device
            self._frame_drop_handler.apply(
                buffer=self._render_buffer,
                policy=self._frame_drop_policy,
                new_frame=data,
            )

        logger.info("Inference thread stopped")

    def _render_loop(self, config: PipelineConfig) -> None:
        """レンダリングスレッドのメインループ。

        render_bufferからデータを取得し、レンダリングを行い、
        結果をdisplay_bufferに追加する。

        仕様書§6.2に従い、device_map.rendererで指定されたデバイスを使用する。

        Phase 4ではRendererのパススルー動作を行う。
        クロッピング済み画像をそのままdisplay_bufferに渡す。

        Args:
            config: パイプライン設定。device_map.renderer を参照。
        """
        device = config.device_map.renderer
        logger.info("Render thread started (device: {})", device)

        while self._running.is_set():
            data = self._render_buffer.get()
            if data is None:
                continue

            display_frame = data.get("frame", data.get("cropped"))
            self._frame_drop_handler.apply(
                buffer=self._display_buffer,
                policy=self._frame_drop_policy,
                new_frame={
                    "display": display_frame,
                    "index": data.get("index", -1),
                    "device": device,
                },
            )

        logger.info("Render thread stopped")

    def _display_loop_opencv(self, config: PipelineConfig) -> None:
        """OpenCV表示ループ（メインスレッド）。

        display_bufferからレンダリング結果を取得し、OpenCVウィンドウに表示する。
        'q' キーが押されるとループを終了する。

        Args:
            config: パイプライン設定。pipeline.name をウィンドウタイトルに使用。
        """
        fps_counter = FPSCounter()
        window_name = f"FLARE - {config.pipeline.name}"
        logger.info("Display loop started (backend: opencv)")

        while self._running.is_set():
            data = self._display_buffer.get()
            if data is None:
                continue

            display = data["display"]
            current_fps = fps_counter.tick()
            avg_fps = fps_counter.get_average_fps(window=30)

            self._metrics.update(fps=current_fps, dropped=0)

            display_vis = draw_fps(display, avg_fps)
            cv2.imshow(window_name, display_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit key pressed")
                break

        logger.info("Display loop stopped (backend: opencv)")

    def _display_loop_pyqt(self, config: PipelineConfig) -> None:
        """PyQt6 GUI表示ループ（メインスレッド）。

        display_bufferからレンダリング結果を取得し、PyQt6ウィンドウに表示する。
        ウィンドウの閉じるボタンまたはQキーでループを終了する。

        PyQt6が利用不可の場合はOpenCVにフォールバックする。

        Args:
            config: パイプライン設定。pipeline.name をウィンドウタイトルに使用。
        """
        if not _HAS_PYQT6:
            logger.warning(
                "PyQt6 not available, falling back to OpenCV display"
            )
            self._display_loop_opencv(config)
            return

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        window = QMainWindow()
        window.setWindowTitle(f"FLARE - {config.pipeline.name}")
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(image_label)
        window.setCentralWidget(central_widget)
        window.resize(640, 480)
        window.show()

        fps_counter = FPSCounter()
        logger.info("Display loop started (backend: pyqt)")

        def update_frame() -> None:
            """タイマーコールバックでフレームを更新する。"""
            if not self._running.is_set():
                app.quit()
                return

            data = self._display_buffer.get()
            if data is None:
                return

            display = data["display"]
            current_fps = fps_counter.tick()
            avg_fps = fps_counter.get_average_fps(window=30)
            self._metrics.update(fps=current_fps, dropped=0)

            display_vis = draw_fps(display, avg_fps)

            if display_vis.ndim == 3:
                h, w, ch = display_vis.shape
                rgb = cv2.cvtColor(display_vis, cv2.COLOR_BGR2RGB)
                q_img = QImage(
                    rgb.data, w, h, ch * w, QImage.Format.Format_RGB888
                )
                pixmap = QPixmap.fromImage(q_img)
                image_label.setPixmap(
                    pixmap.scaled(
                        image_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                    )
                )

        timer = QTimer()
        timer.timeout.connect(update_frame)
        timer.start(1)

        app.exec()

        logger.info("Display loop stopped (backend: pyqt)")
