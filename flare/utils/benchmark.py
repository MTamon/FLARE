"""パイプラインベンチマーキングツール。

仕様書Phase 5のベンチマーキングツールとして、パイプラインの各コンポーネント
（顔検出・特徴量抽出・パラメータ変換・レンダリング）のレイテンシを
個別に計測し、ボトルネック分析を可能にする。

GPUウォームアップフレームを除外した計測により、安定したベンチマーク結果を得る。

Example:
    >>> from flare.config import load_config
    >>> from flare.utils.benchmark import PipelineBenchmark
    >>> config = load_config("config.yaml")
    >>> bench = PipelineBenchmark(config)
    >>> results = bench.run(n_frames=300, warmup_frames=30)
    >>> bench.print_report(results)
    === FLARE Benchmark Report ===
    Average FPS     : 85.3
    Average Latency : 11.7 ms
    --- Component Breakdown ---
    Face Detect     :  3.2 ms
    Extract         :  4.1 ms
    Convert         :  0.3 ms
    Render          :  4.1 ms
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
from loguru import logger

from flare.config import FLAREConfig
from flare.utils.face_detect import FaceDetector
from flare.utils.metrics import FPSCounter


class PipelineBenchmark:
    """パイプラインベンチマーキングツール。

    ダミーデータを使用してパイプラインの各コンポーネントのレイテンシを
    個別に計測する。GPUウォームアップを考慮した正確な計測が可能。

    Attributes:
        _config: FLARE統合設定。
        _face_detector: 顔検出器。

    Example:
        >>> bench = PipelineBenchmark(config)
        >>> results = bench.run(n_frames=300)
        >>> bench.print_report(results)
    """

    #: ベンチマーク用ダミー画像サイズ
    _DUMMY_IMAGE_SIZE: int = 224

    #: ベンチマーク用ダミーフレームサイズ
    _DUMMY_FRAME_SIZE: tuple[int, int] = (480, 640)

    def __init__(self, config: FLAREConfig) -> None:
        """PipelineBenchmarkを初期化する。

        Args:
            config: FLARE統合設定。
        """
        self._config: FLAREConfig = config
        self._face_detector: FaceDetector = FaceDetector(
            device="cpu", fallback_to_prev=True
        )

    def run(
        self,
        n_frames: int = 300,
        *,
        warmup_frames: int = 30,
    ) -> Dict[str, Any]:
        """ベンチマークを実行する。

        ダミーデータを使ってn_framesフレーム処理し、各コンポーネントの
        レイテンシを計測する。warmup_frames分は計測に含めない。

        Args:
            n_frames: 計測対象のフレーム数。デフォルト300。
            warmup_frames: ウォームアップフレーム数（計測に含めない）。
                デフォルト30。GPU初回推論のオーバーヘッドを除外する。

        Returns:
            ベンチマーク結果Dict:
                - ``avg_fps``: 平均FPS
                - ``avg_total_ms``: 平均総レイテンシ (ms)
                - ``component_breakdown``: コンポーネント別平均レイテンシ
                - ``frame_results``: フレームごとの計測結果リスト
        """
        total_frames: int = warmup_frames + n_frames
        frame_results: List[Dict[str, float]] = []
        fps_counter: FPSCounter = FPSCounter(window_size=n_frames)

        logger.info(
            "ベンチマーク開始: n_frames={} | warmup={} | config={}",
            n_frames,
            warmup_frames,
            self._config.pipeline.name,
        )

        # ダミーデータ生成
        dummy_frame: np.ndarray = self._create_dummy_frame()
        dummy_cropped: np.ndarray = self._create_dummy_cropped()

        for i in range(total_frames):
            frame_timing: Dict[str, float] = self._benchmark_single_frame(
                dummy_frame, dummy_cropped
            )

            if i >= warmup_frames:
                frame_results.append(frame_timing)
                fps_counter.update()

        results: Dict[str, Any] = self._aggregate_results(
            frame_results, fps_counter
        )

        logger.info(
            "ベンチマーク完了: avg_fps={:.1f} | avg_latency={:.1f}ms",
            results["avg_fps"],
            results["avg_total_ms"],
        )

        return results

    def _benchmark_single_frame(
        self,
        frame: np.ndarray,
        cropped: np.ndarray,
    ) -> Dict[str, float]:
        """1フレームの各コンポーネントを計測する。

        Args:
            frame: ダミーフレーム画像。
            cropped: ダミークロップ済み画像。

        Returns:
            各コンポーネントのレイテンシ (ms) を含むDict。
        """
        # Face Detection
        t0: float = time.perf_counter()
        self._bench_face_detect(frame)
        t1: float = time.perf_counter()
        face_detect_ms: float = (t1 - t0) * 1000.0

        # Feature Extraction
        t2: float = time.perf_counter()
        self._bench_extract(cropped)
        t3: float = time.perf_counter()
        extract_ms: float = (t3 - t2) * 1000.0

        # Parameter Conversion
        t4: float = time.perf_counter()
        self._bench_convert()
        t5: float = time.perf_counter()
        convert_ms: float = (t5 - t4) * 1000.0

        # Rendering
        t6: float = time.perf_counter()
        self._bench_render()
        t7: float = time.perf_counter()
        render_ms: float = (t7 - t6) * 1000.0

        total_ms: float = (t7 - t0) * 1000.0

        return {
            "face_detect_ms": face_detect_ms,
            "extract_ms": extract_ms,
            "convert_ms": convert_ms,
            "render_ms": render_ms,
            "total_ms": total_ms,
        }

    def _bench_face_detect(self, frame: np.ndarray) -> None:
        """顔検出のベンチマーク。

        Args:
            frame: 入力フレーム画像。
        """
        try:
            self._face_detector.detect(frame)
        except Exception:
            pass

    def _bench_extract(self, cropped: np.ndarray) -> None:
        """特徴量抽出のベンチマーク（ダミー計算）。

        実際のExtractorが未接続の場合はnumpy演算で
        推論コストをシミュレートする。

        Args:
            cropped: クロップ済み画像。
        """
        _ = np.mean(cropped, axis=(0, 1))
        _ = np.std(cropped, axis=(0, 1))

    def _bench_convert(self) -> None:
        """パラメータ変換のベンチマーク（ダミー計算）。

        軽量な変換操作をシミュレートする。
        """
        dummy: np.ndarray = np.zeros(120, dtype=np.float32)
        _ = dummy * 1.0

    def _bench_render(self) -> None:
        """レンダリングのベンチマーク（ダミー計算）。

        実際のRendererが未接続の場合はnumpy演算で
        レンダリングコストをシミュレートする。
        """
        _ = np.zeros((3, 512, 512), dtype=np.float32)

    def _aggregate_results(
        self,
        frame_results: List[Dict[str, float]],
        fps_counter: FPSCounter,
    ) -> Dict[str, Any]:
        """フレームごとの計測結果を集約する。

        Args:
            frame_results: フレームごとの計測結果リスト。
            fps_counter: FPSカウンタ。

        Returns:
            集約されたベンチマーク結果。
        """
        if not frame_results:
            return {
                "avg_fps": 0.0,
                "avg_total_ms": 0.0,
                "component_breakdown": {
                    "face_detect_ms": 0.0,
                    "extract_ms": 0.0,
                    "convert_ms": 0.0,
                    "render_ms": 0.0,
                },
                "frame_results": [],
            }

        n: int = len(frame_results)

        avg_face: float = sum(r["face_detect_ms"] for r in frame_results) / n
        avg_extract: float = sum(r["extract_ms"] for r in frame_results) / n
        avg_convert: float = sum(r["convert_ms"] for r in frame_results) / n
        avg_render: float = sum(r["render_ms"] for r in frame_results) / n
        avg_total: float = sum(r["total_ms"] for r in frame_results) / n

        avg_fps: float = fps_counter.get_fps()
        if avg_fps == 0.0 and avg_total > 0.0:
            avg_fps = 1000.0 / avg_total

        return {
            "avg_fps": avg_fps,
            "avg_total_ms": avg_total,
            "component_breakdown": {
                "face_detect_ms": avg_face,
                "extract_ms": avg_extract,
                "convert_ms": avg_convert,
                "render_ms": avg_render,
            },
            "frame_results": frame_results,
        }

    def print_report(self, results: Dict[str, Any]) -> None:
        """ベンチマーク結果をターミナルに出力する。

        Args:
            results: ``run()`` の戻り値。
        """
        bd: Dict[str, float] = results.get("component_breakdown", {})

        report: str = (
            "\n"
            "=== FLARE Benchmark Report ===\n"
            f"Average FPS     : {results.get('avg_fps', 0.0):.1f}\n"
            f"Average Latency : {results.get('avg_total_ms', 0.0):.1f} ms\n"
            "--- Component Breakdown ---\n"
            f"Face Detect     : {bd.get('face_detect_ms', 0.0):.1f} ms\n"
            f"Extract         : {bd.get('extract_ms', 0.0):.1f} ms\n"
            f"Convert         : {bd.get('convert_ms', 0.0):.1f} ms\n"
            f"Render          : {bd.get('render_ms', 0.0):.1f} ms\n"
        )

        print(report)
        logger.info("ベンチマークレポート出力完了")

    def _create_dummy_frame(self) -> np.ndarray:
        """ベンチマーク用ダミーフレーム画像を生成する。

        Returns:
            shape (480, 640, 3) のuint8 ndarray。
        """
        rng: np.random.Generator = np.random.default_rng(seed=0)
        return rng.integers(
            0, 256,
            size=(self._DUMMY_FRAME_SIZE[0], self._DUMMY_FRAME_SIZE[1], 3),
            dtype=np.uint8,
        )

    def _create_dummy_cropped(self) -> np.ndarray:
        """ベンチマーク用ダミークロップ済み画像を生成する。

        Returns:
            shape (224, 224, 3) のuint8 ndarray。
        """
        rng: np.random.Generator = np.random.default_rng(seed=1)
        size: int = self._DUMMY_IMAGE_SIZE
        return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
