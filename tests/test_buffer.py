"""flare.pipeline.buffer のテスト。

PipelineBufferのput/get操作、FrameDropPolicy、統計情報収集、
タイムアウト動作、clear操作を検証する。
"""

from __future__ import annotations

import pytest

from flare.pipeline.buffer import FrameDropPolicy, PipelineBuffer


class TestFrameDropPolicy:
    """FrameDropPolicy enumのテスト。"""

    def test_frame_drop_policy_enum(self) -> None:
        """DROP_OLDEST / BLOCK / INTERPOLATEの3値を持つことを確認する。"""
        assert FrameDropPolicy.DROP_OLDEST.value == "drop_oldest"
        assert FrameDropPolicy.BLOCK.value == "block"
        assert FrameDropPolicy.INTERPOLATE.value == "interpolate"
        assert len(FrameDropPolicy) == 3


class TestPipelineBuffer:
    """PipelineBufferのテストスイート。"""

    def test_put_and_get(self) -> None:
        """put()でデータを追加しget()で取得できることを確認する。"""
        buf: PipelineBuffer = PipelineBuffer(max_size=10, timeout=0.1)

        buf.put({"frame": "data_0", "idx": 0})
        buf.put({"frame": "data_1", "idx": 1})

        result = buf.get()
        assert result is not None
        assert result["idx"] == 0

        result = buf.get()
        assert result is not None
        assert result["idx"] == 1

    def test_drop_oldest_policy(self) -> None:
        """max_size=2のバッファに3件put()した場合、最古が破棄されることを確認する。

        最古（idx=0）が破棄され、idx=1とidx=2が残る。
        """
        buf: PipelineBuffer = PipelineBuffer(
            max_size=2, timeout=0.1, overflow_policy="drop_oldest"
        )

        buf.put({"idx": 0})
        buf.put({"idx": 1})
        buf.put({"idx": 2})  # idx=0 がドロップされる

        first = buf.get()
        assert first is not None
        assert first["idx"] == 1

        second = buf.get()
        assert second is not None
        assert second["idx"] == 2

    def test_stats_dropped_count(self) -> None:
        """ドロップ発生時にget_stats()["dropped"]が正しくカウントされることを確認する。"""
        buf: PipelineBuffer = PipelineBuffer(
            max_size=1, timeout=0.1, overflow_policy="drop_oldest"
        )

        buf.put({"idx": 0})
        buf.put({"idx": 1})  # idx=0 ドロップ
        buf.put({"idx": 2})  # idx=1 ドロップ

        stats = buf.get_stats()
        assert stats["dropped"] == 2

    def test_stats_total_put(self) -> None:
        """put() N回でget_stats()["total_put"] == N であることを確認する。"""
        buf: PipelineBuffer = PipelineBuffer(max_size=100, timeout=0.1)

        for i in range(7):
            buf.put({"idx": i})

        stats = buf.get_stats()
        assert stats["total_put"] == 7

    def test_stats_total_get(self) -> None:
        """get()成功回数がget_stats()["total_get"]に反映されることを確認する。"""
        buf: PipelineBuffer = PipelineBuffer(max_size=10, timeout=0.1)

        for i in range(3):
            buf.put({"idx": i})

        buf.get()
        buf.get()

        stats = buf.get_stats()
        assert stats["total_get"] == 2

    def test_get_timeout_returns_none(self) -> None:
        """空バッファからget(timeout=0.1)がNoneを返すことを確認する。"""
        buf: PipelineBuffer = PipelineBuffer(max_size=10, timeout=0.05)

        result = buf.get(timeout=0.1)
        assert result is None

    def test_qsize(self) -> None:
        """put()後にqsize()が正しい値を返すことを確認する。"""
        buf: PipelineBuffer = PipelineBuffer(max_size=10, timeout=0.1)

        assert buf.qsize() == 0

        buf.put({"idx": 0})
        buf.put({"idx": 1})
        buf.put({"idx": 2})

        assert buf.qsize() == 3

        buf.get()
        assert buf.qsize() == 2

    def test_is_empty_and_full(self) -> None:
        """is_empty()とis_full()が正しく動作することを確認する。"""
        buf: PipelineBuffer = PipelineBuffer(max_size=2, timeout=0.1)

        assert buf.is_empty() is True
        assert buf.is_full() is False

        buf.put({"idx": 0})
        buf.put({"idx": 1})

        assert buf.is_empty() is False
        assert buf.is_full() is True

    def test_clear(self) -> None:
        """clear()後にis_empty()がTrueを返すことを確認する。"""
        buf: PipelineBuffer = PipelineBuffer(max_size=10, timeout=0.1)

        buf.put({"idx": 0})
        buf.put({"idx": 1})
        buf.put({"idx": 2})
        assert buf.qsize() == 3

        buf.clear()
        assert buf.is_empty() is True
        assert buf.qsize() == 0

    def test_reset_stats(self) -> None:
        """reset_stats()で統計がゼロリセットされることを確認する。"""
        buf: PipelineBuffer = PipelineBuffer(
            max_size=1, timeout=0.1, overflow_policy="drop_oldest"
        )

        buf.put({"idx": 0})
        buf.put({"idx": 1})  # drop
        buf.get()

        buf.reset_stats()
        stats = buf.get_stats()

        assert stats["dropped"] == 0
        assert stats["total_put"] == 0
        assert stats["total_get"] == 0
