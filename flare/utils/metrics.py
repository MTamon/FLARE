"""パイプラインメトリクス計測ユーティリティ。

FPS計測（移動平均）とパイプライン全体の統計情報を収集する。
リアルタイムパイプラインの性能監視とボトルネック分析に使用する。

Example:
    >>> fps_counter = FPSCounter(window_size=30)
    >>> metrics = PipelineMetrics()
    >>>
    >>> for frame in pipeline:
    ...     start = time.time()
    ...     process(frame)
    ...     latency = (time.time() - start) * 1000
    ...     fps_counter.update()
    ...     metrics.record_frame(latency_ms=latency)
    ...     print(f"FPS: {fps_counter.get_fps():.1f}")
    >>>
    >>> print(metrics.summary())
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict


class FPSCounter:
    """直近Nフレームの移動平均FPSカウンタ。

    フレーム処理ごとに ``update()`` を呼び出し、``get_fps()`` で
    直近 ``window_size`` フレームの平均FPSを取得する。

    Attributes:
        _window_size: 移動平均のウィンドウサイズ。
        _timestamps: フレーム処理時刻のdeque。

    Example:
        >>> counter = FPSCounter(window_size=60)
        >>> for _ in range(100):
        ...     do_work()
        ...     counter.update()
        >>> print(f"FPS: {counter.get_fps():.1f}")
    """

    def __init__(self, window_size: int = 30) -> None:
        """FPSCounterを初期化する。

        Args:
            window_size: 移動平均を計算するフレーム数。デフォルト30。
        """
        self._window_size: int = window_size
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def update(self) -> None:
        """現在時刻を記録してFPSを更新する。

        フレーム処理の完了時に呼び出す。
        """
        self._timestamps.append(time.perf_counter())

    def get_fps(self) -> float:
        """直近window_sizeフレームの平均FPSを返す。

        タイムスタンプが2つ未満の場合は0.0を返す。

        Returns:
            平均フレームレート（fps）。
        """
        if len(self._timestamps) < 2:
            return 0.0

        elapsed: float = self._timestamps[-1] - self._timestamps[0]

        if elapsed <= 0.0:
            return 0.0

        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """タイムスタンプをクリアしてリセットする。"""
        self._timestamps.clear()


class PipelineMetrics:
    """パイプライン全体の統計情報を収集するクラス。

    フレーム処理のレイテンシ、ドロップフレーム数、ドロップ率等を
    記録し、サマリーとして取得する。

    Attributes:
        _total_frames: 処理済み総フレーム数。
        _dropped_frames: ドロップされたフレーム数。
        _latencies: フレームごとのレイテンシ（ミリ秒）のリスト。

    Example:
        >>> metrics = PipelineMetrics()
        >>> metrics.record_frame(latency_ms=12.5)
        >>> metrics.record_drop()
        >>> print(metrics.summary())
    """

    def __init__(self) -> None:
        """PipelineMetricsを初期化する。"""
        self._total_frames: int = 0
        self._dropped_frames: int = 0
        self._latencies: list[float] = []
        self._max_latency: float = 0.0
        self._sum_latency: float = 0.0

    def record_drop(self) -> None:
        """ドロップフレームを1カウントする。

        フレームドロップポリシーにより破棄されたフレームを記録する。
        """
        self._dropped_frames += 1

    def record_frame(self, latency_ms: float) -> None:
        """フレーム処理を記録する。

        Args:
            latency_ms: フレームの処理レイテンシ（ミリ秒）。
        """
        self._total_frames += 1
        self._sum_latency += latency_ms

        if latency_ms > self._max_latency:
            self._max_latency = latency_ms

    def get_drop_rate(self) -> float:
        """ドロップ率を返す。

        Returns:
            ドロップ率（0.0〜1.0）。フレーム未処理時は0.0。
        """
        total: int = self._total_frames + self._dropped_frames

        if total == 0:
            return 0.0

        return self._dropped_frames / total

    def summary(self) -> Dict[str, float]:
        """パイプライン統計のサマリーを返す。

        Returns:
            以下のキーを含むDict:
                - ``dropped_frames``: ドロップされたフレーム数
                - ``total_frames``: 処理済みフレーム数
                - ``drop_rate``: ドロップ率（0.0〜1.0）
                - ``avg_latency_ms``: 平均レイテンシ（ミリ秒）
                - ``max_latency_ms``: 最大レイテンシ（ミリ秒）

        Example:
            >>> metrics.summary()
            {'dropped_frames': 5.0, 'total_frames': 1000.0, ...}
        """
        avg_latency: float = 0.0
        if self._total_frames > 0:
            avg_latency = self._sum_latency / self._total_frames

        return {
            "dropped_frames": float(self._dropped_frames),
            "total_frames": float(self._total_frames),
            "drop_rate": self.get_drop_rate(),
            "avg_latency_ms": avg_latency,
            "max_latency_ms": self._max_latency,
        }

    def reset(self) -> None:
        """全メトリクスをリセットする。"""
        self._total_frames = 0
        self._dropped_frames = 0
        self._latencies = []
        self._max_latency = 0.0
        self._sum_latency = 0.0
