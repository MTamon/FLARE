"""速度計測・ベンチマークユーティリティ

パイプライン各ステージのレイテンシ計測と統計情報を収集する。
Section 6.5 の速度見積もりの検証に使用する。
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional

from loguru import logger


class LatencyTracker:
    """各ステージの処理時間を記録し、統計を提供する。

    使用例::

        tracker = LatencyTracker()
        with tracker.measure("extractor"):
            result = extractor.extract(image)
        print(tracker.summary())
    """

    def __init__(self, window_size: int = 100) -> None:
        """
        Args:
            window_size: 統計計算に使用する直近サンプル数。
        """
        self._records: Dict[str, List[float]] = defaultdict(list)
        self._window_size = window_size

    @contextmanager
    def measure(self, stage: str) -> Generator[None, None, None]:
        """コンテキストマネージャで処理時間を記録する。"""
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        buf = self._records[stage]
        buf.append(elapsed_ms)
        if len(buf) > self._window_size:
            buf.pop(0)

    def record(self, stage: str, elapsed_ms: float) -> None:
        """手動で記録する場合。"""
        buf = self._records[stage]
        buf.append(elapsed_ms)
        if len(buf) > self._window_size:
            buf.pop(0)

    def stats(self, stage: str) -> Optional[Dict[str, float]]:
        """指定ステージの統計を返す。記録がなければ None。"""
        buf = self._records.get(stage)
        if not buf:
            return None
        n = len(buf)
        mean = sum(buf) / n
        sorted_buf = sorted(buf)
        p50 = sorted_buf[n // 2]
        p95 = sorted_buf[int(n * 0.95)] if n >= 20 else sorted_buf[-1]
        fps = 1000.0 / mean if mean > 0 else float("inf")
        return {
            "count": n,
            "mean_ms": round(mean, 2),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "est_fps": round(fps, 1),
        }

    def summary(self) -> Dict[str, Dict[str, float]]:
        """全ステージの統計を返す。"""
        return {
            stage: self.stats(stage)
            for stage in self._records
            if self.stats(stage) is not None
        }

    def log_summary(self) -> None:
        """統計をログに出力する。"""
        for stage, s in self.summary().items():
            logger.info(
                f"[{stage}] mean={s['mean_ms']:.1f}ms "
                f"p50={s['p50_ms']:.1f}ms p95={s['p95_ms']:.1f}ms "
                f"~{s['est_fps']:.0f}FPS ({s['count']} samples)"
            )

    def reset(self, stage: Optional[str] = None) -> None:
        """記録をリセットする。stage 指定で個別、None で全体。"""
        if stage:
            self._records.pop(stage, None)
        else:
            self._records.clear()


class FPSCounter:
    """シンプルな FPS カウンター。表示スレッド等で使用。"""

    def __init__(self, avg_window: int = 30) -> None:
        self._timestamps: List[float] = []
        self._window = avg_window

    def tick(self) -> float:
        """1 フレーム処理完了を記録し、現在の FPS を返す。"""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed