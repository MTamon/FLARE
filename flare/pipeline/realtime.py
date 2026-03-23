"""リアルタイムパイプライン。

仕様書6節「リアルタイムモード設計」に基づき、Webカメラからの映像入力を
リアルタイムに処理し、LHGモデル推論後のレンダリング結果を即座に表示する。

5スレッド構成:
    ================= ======================== ===========
    スレッド             機能                     デバイス
    ================= ======================== ===========
    キャプチャスレッド   Webカメラフレーム取得      CPU
    抽出スレッド         顔検出 + 3DMM抽出         GPU (extractor)
    推論スレッド         LHGモデル推論             GPU (lhg_model)
    レンダリングスレッド  画像生成                  GPU (renderer)
    表示スレッド         GUI表示                   CPU
    ================= ======================== ===========

Example:
    >>> from flare.config import load_config
    >>> from flare.pipeline.realtime import RealtimePipeline
    >>> config = load_config("config.yaml")
    >>> pipeline = RealtimePipeline(config)
    >>> pipeline.run(camera_id=0)
"""

from __future__ import annotations

import signal
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from loguru import logger

from flare.config import FLAREConfig
from flare.pipeline.buffer import PipelineBuffer
from flare.utils.errors import FaceNotDetectedError, PipelineErrorHandler
from flare.utils.face_detect import FaceDetector
from flare.utils.metrics import FPSCounter, PipelineMetrics
from flare.utils.video import VideoReader
from flare.utils.visualization import draw_fps_overlay, draw_metrics_overlay


class RealtimePipeline:
    """リアルタイム処理パイプライン。

    5スレッド構成でWebカメラ入力からリアルタイムに顔検出・
    3DMMパラメータ抽出・LHGモデル推論・レンダリング・表示を行う。

    Attributes:
        _config: FLARE統合設定。
        _running: パイプライン稼働フラグ。
        _threads: 起動中のスレッドリスト。
        _capture_buffer: キャプチャ→抽出間バッファ。
        _extract_buffer: 抽出→推論間バッファ。
        _infer_buffer: 推論→レンダリング間バッファ。
        _render_buffer: レンダリング→表示間バッファ。

    Example:
        >>> pipeline = RealtimePipeline(config)
        >>> pipeline.run(camera_id=0)  # 'q'キーで終了
    """

    def __init__(self, config: FLAREConfig) -> None:
        """RealtimePipelineを初期化する。

        Args:
            config: FLARE統合設定。
        """
        self._config: FLAREConfig = config
        self._running: bool = False
        self._threads: list[threading.Thread] = []

        # バッファ群
        buf_cfg = config.buffer
        self._capture_buffer: PipelineBuffer = PipelineBuffer(
            max_size=buf_cfg.max_size,
            timeout=buf_cfg.timeout_sec,
            overflow_policy=buf_cfg.overflow_policy,
        )
        self._extract_buffer: PipelineBuffer = PipelineBuffer(
            max_size=buf_cfg.max_size,
            timeout=buf_cfg.timeout_sec,
            overflow_policy=buf_cfg.overflow_policy,
        )
        self._infer_buffer: PipelineBuffer = PipelineBuffer(
            max_size=buf_cfg.max_size,
            timeout=buf_cfg.timeout_sec,
            overflow_policy=buf_cfg.overflow_policy,
        )
        self._render_buffer: PipelineBuffer = PipelineBuffer(
            max_size=buf_cfg.max_size,
            timeout=buf_cfg.timeout_sec,
            overflow_policy=buf_cfg.overflow_policy,
        )

        # 共有コンポーネント
        self._face_detector: FaceDetector = FaceDetector(
            device="cpu", fallback_to_prev=True
        )
        self._error_handler: PipelineErrorHandler = PipelineErrorHandler()
        self._metrics: PipelineMetrics = PipelineMetrics()
        self._fps_counter: FPSCounter = FPSCounter(window_size=30)

    def run(self, camera_id: int = 0) -> None:
        """リアルタイムパイプラインを起動する。

        5スレッドを起動し、'q'キー押下またはCtrl+Cで終了する。

        Args:
            camera_id: WebカメラのデバイスID。デフォルト0。
        """
        self._running = True
        self._camera_id: int = camera_id

        # SIGINTハンドラ登録
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info(
            "リアルタイムパイプライン起動: camera={} | config={}",
            camera_id,
            self._config.pipeline.name,
        )

        # スレッド定義
        thread_targets = [
            ("capture", self._capture_thread),
            ("extract", self._extract_thread),
            ("infer", self._infer_thread),
            ("render", self._render_thread),
            ("display", self._display_thread),
        ]

        self._threads = []
        for name, target in thread_targets:
            t = threading.Thread(target=target, name=f"flare-{name}", daemon=True)
            self._threads.append(t)

        # 全スレッド起動
        for t in self._threads:
            t.start()
            logger.debug("スレッド起動: {}", t.name)

        # 全スレッド完了待ち
        try:
            for t in self._threads:
                t.join()
        except KeyboardInterrupt:
            self.stop()
        finally:
            signal.signal(signal.SIGINT, original_handler)

        self._log_final_stats()

    def stop(self) -> None:
        """パイプラインを安全に停止する。

        _runningフラグをFalseに設定し、全スレッドの終了を待つ。
        """
        if not self._running:
            return

        logger.info("パイプライン停止中...")
        self._running = False

        for t in self._threads:
            t.join(timeout=3.0)
            if t.is_alive():
                logger.warning("スレッド終了タイムアウト: {}", t.name)

        logger.info("パイプライン停止完了")

    def _signal_handler(self, signum: int, frame: object) -> None:
        """SIGINTシグナルハンドラ。

        Args:
            signum: シグナル番号。
            frame: 現在のスタックフレーム。
        """
        logger.info("SIGINT受信: パイプラインを停止します")
        self.stop()

    def _capture_thread(self) -> None:
        """キャプチャスレッド: Webカメラからフレームを取得する。"""
        try:
            reader: VideoReader = VideoReader(
                self._camera_id,
                width=self._config.renderer.output_size[1],
                height=self._config.renderer.output_size[0],
            )
        except Exception as exc:
            logger.error("カメラオープン失敗: {}", exc)
            self._running = False
            return

        frame_idx: int = 0
        try:
            while self._running:
                frame: Optional[np.ndarray] = reader.read_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                self._capture_buffer.put({
                    "frame": frame,
                    "frame_idx": frame_idx,
                    "timestamp": time.perf_counter(),
                })
                frame_idx += 1
        finally:
            reader.release()

    def _extract_thread(self) -> None:
        """抽出スレッド: 顔検出 + 3DMMパラメータ抽出を行う。"""
        while self._running:
            data: Optional[Dict[str, Any]] = self._capture_buffer.get()
            if data is None:
                continue

            frame: np.ndarray = data["frame"]
            frame_idx: int = data["frame_idx"]

            try:
                bbox = self._face_detector.detect(frame)
                cropped: np.ndarray = self._face_detector.crop_and_align(
                    frame, bbox, size=self._config.extractor.input_size
                )

                # Phase 2で具象Extractorを接続
                # 現時点ではクロップ済み画像と原画像をパススルー
                self._extract_buffer.put({
                    "frame": frame,
                    "cropped": cropped,
                    "bbox": bbox,
                    "frame_idx": frame_idx,
                    "timestamp": data["timestamp"],
                })
            except FaceNotDetectedError:
                self._metrics.record_drop()
            except Exception as exc:
                policy = self._error_handler.handle(exc, {"frame_idx": frame_idx})
                self._metrics.record_drop()

    def _infer_thread(self) -> None:
        """推論スレッド: LHGモデル推論を行う。"""
        while self._running:
            data: Optional[Dict[str, Any]] = self._extract_buffer.get()
            if data is None:
                continue

            # Phase 2で具象LHGModelを接続
            # 現時点ではデータをパススルー
            self._infer_buffer.put(data)

    def _render_thread(self) -> None:
        """レンダリングスレッド: 画像生成を行う。"""
        while self._running:
            data: Optional[Dict[str, Any]] = self._infer_buffer.get()
            if data is None:
                continue

            # Phase 2で具象Rendererを接続
            # 現時点では原画像をそのまま渡す
            self._render_buffer.put({
                "rendered": data.get("frame", data.get("cropped")),
                "original": data.get("frame"),
                "frame_idx": data.get("frame_idx", 0),
                "timestamp": data.get("timestamp", 0.0),
            })

    def _display_thread(self) -> None:
        """表示スレッド: レンダリング結果をGUIに表示する。"""
        window_name: str = f"FLARE - {self._config.pipeline.name}"

        while self._running:
            data: Optional[Dict[str, Any]] = self._render_buffer.get()
            if data is None:
                continue

            rendered: np.ndarray = data["rendered"]

            # レイテンシ計測
            latency_ms: float = (
                time.perf_counter() - data.get("timestamp", time.perf_counter())
            ) * 1000.0
            self._metrics.record_frame(latency_ms=latency_ms)
            self._fps_counter.update()

            # FPSオーバーレイ
            display_frame: np.ndarray = draw_fps_overlay(
                rendered, self._fps_counter.get_fps()
            )

            try:
                cv2.imshow(window_name, display_frame)
                key: int = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("'q'キー押下: パイプラインを停止します")
                    self._running = False
            except Exception:
                # ヘッドレス環境でimshowが使えない場合
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _log_final_stats(self) -> None:
        """終了時の統計情報をログに出力する。"""
        summary: Dict[str, float] = self._metrics.summary()

        capture_stats: Dict[str, int] = self._capture_buffer.get_stats()
        extract_stats: Dict[str, int] = self._extract_buffer.get_stats()

        logger.info(
            "パイプライン統計: "
            "total_frames={:.0f} | "
            "dropped={:.0f} | "
            "drop_rate={:.2%} | "
            "avg_latency={:.1f}ms | "
            "max_latency={:.1f}ms | "
            "capture_drops={} | "
            "extract_drops={}",
            summary.get("total_frames", 0),
            summary.get("dropped_frames", 0),
            summary.get("drop_rate", 0),
            summary.get("avg_latency_ms", 0),
            summary.get("max_latency_ms", 0),
            capture_stats.get("dropped", 0),
            extract_stats.get("dropped", 0),
        )
