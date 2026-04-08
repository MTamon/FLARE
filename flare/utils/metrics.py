"""パイプラインパフォーマンス計測ユーティリティモジュール。

リアルタイムパイプラインおよびバッチ処理パイプラインの速度計測・統計収集を提供する。
FPS計測、フレームドロップ統計、処理時間のサマリ生成に使用される。

Example:
    PipelineMetricsによる統計収集::

        metrics = PipelineMetrics()
        for batch in pipeline:
            fps = process(batch)
            metrics.update(fps=fps, dropped=batch.dropped_count)
        summary = metrics.get_summary()
        print(f"Avg FPS: {summary['avg_fps']:.1f}")

    FPSCounterによるリアルタイムFPS計測::

        counter = FPSCounter()
        while running:
            process_frame()
            current_fps = counter.tick()
            avg_fps = counter.get_average_fps(window=30)
"""

from __future__ import annotations

import time
from collections import deque


class PipelineMetrics:
    """パイプラインのパフォーマンス統計を収集するクラス。

    フレームレートやフレームドロップの統計を蓄積し、サマリを生成する。
    リアルタイムモードとバッチモードの両方で使用可能。

    Attributes:
        _fps_history: 記録されたFPS値のリスト。
        _total_dropped: ドロップされたフレームの累計数。
    """

    def __init__(self) -> None:
        """PipelineMetricsを初期化する。"""
        self._fps_history: list[float] = []
        self._total_dropped: int = 0

    def update(self, fps: float, dropped: int = 0) -> None:
        """計測値を記録する。

        Args:
            fps: 直近の計測FPS値。
            dropped: 直近の計測期間でドロップされたフレーム数。
        """
        self._fps_history.append(fps)
        self._total_dropped += dropped

    def get_summary(self) -> dict[str, float]:
        """蓄積された統計のサマリを返す。

        Returns:
            以下のキーを持つ辞書::

                {
                    "avg_fps": 平均FPS,
                    "max_fps": 最大FPS,
                    "min_fps": 最小FPS,
                    "total_dropped": ドロップフレーム累計数,
                }

            FPS履歴が空の場合、avg_fps / max_fps / min_fps は全て ``0.0``。
        """
        if len(self._fps_history) == 0:
            return {
                "avg_fps": 0.0,
                "max_fps": 0.0,
                "min_fps": 0.0,
                "total_dropped": float(self._total_dropped),
            }
        return {
            "avg_fps": sum(self._fps_history) / len(self._fps_history),
            "max_fps": max(self._fps_history),
            "min_fps": min(self._fps_history),
            "total_dropped": float(self._total_dropped),
        }

    def reset(self) -> None:
        """全ての統計をリセットする。"""
        self._fps_history.clear()
        self._total_dropped = 0


class FPSCounter:
    """リアルタイムFPS計測クラス。

    tick()を毎フレーム呼び出すことで、直近のフレームレートを計測する。
    スライディングウィンドウ方式で平均FPSも取得可能。

    Attributes:
        _last_time: 前回のtick()呼び出し時刻。
        _fps_history: 直近のFPS計測値を保持するdeque。
    """

    def __init__(self, max_history: int = 300) -> None:
        """FPSCounterを初期化する。

        Args:
            max_history: FPS履歴の最大保持件数。古い値から自動的に破棄される。
        """
        self._last_time: float | None = None
        self._fps_history: deque[float] = deque(maxlen=max_history)

    def tick(self) -> float:
        """フレームの完了を記録し、瞬時FPSを返す。

        初回呼び出し時はタイムスタンプの初期化のみ行い、``0.0`` を返す。

        Returns:
            前回のtick()からの経過時間に基づく瞬時FPS。
            初回呼び出し時は ``0.0``。
        """
        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
            return 0.0
        elapsed = now - self._last_time
        self._last_time = now
        fps = 1.0 / elapsed if elapsed > 0.0 else 0.0
        self._fps_history.append(fps)
        return fps

    def get_average_fps(self, window: int = 30) -> float:
        """スライディングウィンドウ方式で平均FPSを返す。

        直近のwindowフレーム分のFPS値を平均する。

        Args:
            window: 平均計算に使用する直近フレーム数。
                履歴がwindow未満の場合は、利用可能な全履歴で計算する。

        Returns:
            平均FPS。履歴が空の場合は ``0.0``。
        """
        if len(self._fps_history) == 0:
            return 0.0
        recent = list(self._fps_history)[-window:]
        return sum(recent) / len(recent)
