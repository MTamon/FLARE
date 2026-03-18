"""PipelineBuffer のテスト (Section 8.4)"""

from __future__ import annotations

import threading
import time

import pytest

from flare.pipeline.buffer import PipelineBuffer


class TestPipelineBufferBasic:
    def test_put_and_get(self):
        buf = PipelineBuffer(max_size=10, timeout=1.0)
        data = {"frame_idx": 0, "tensor": "dummy"}
        assert buf.put(data) is True
        result = buf.get()
        assert result is not None
        assert result["frame_idx"] == 0

    def test_get_timeout_returns_none(self):
        buf = PipelineBuffer(max_size=10, timeout=0.05)
        result = buf.get()
        assert result is None

    def test_qsize(self):
        buf = PipelineBuffer(max_size=10)
        assert buf.qsize == 0
        buf.put({"a": 1})
        buf.put({"b": 2})
        assert buf.qsize == 2

    def test_empty_and_full(self):
        buf = PipelineBuffer(max_size=2)
        assert buf.empty is True
        assert buf.full is False
        buf.put({"a": 1})
        buf.put({"b": 2})
        assert buf.full is True

    def test_clear(self):
        buf = PipelineBuffer(max_size=10)
        for i in range(5):
            buf.put({"i": i})
        cleared = buf.clear()
        assert cleared == 5
        assert buf.empty is True


class TestPipelineBufferDropOldest:
    """Section 6.4: drop_oldest ポリシー。"""

    def test_overflow_drops_oldest(self):
        buf = PipelineBuffer(max_size=3, overflow_policy="drop_oldest")
        for i in range(5):
            buf.put({"i": i})

        # 最新 3 フレーム (2, 3, 4) が残っているはず
        assert buf.qsize == 3
        first = buf.get()
        assert first is not None
        assert first["i"] == 2

    def test_drop_stats(self):
        buf = PipelineBuffer(max_size=2, overflow_policy="drop_oldest")
        for i in range(5):
            buf.put({"i": i})
        stats = buf.stats
        assert stats["dropped"] == 3
        assert stats["total_put"] == 5


class TestPipelineBufferBlock:
    """Section 6.4: block ポリシー。"""

    def test_block_policy_waits(self):
        buf = PipelineBuffer(max_size=2, timeout=0.1, overflow_policy="block")
        buf.put({"a": 1})
        buf.put({"b": 2})
        # バッファ満杯 → block → timeout → False
        result = buf.put({"c": 3})
        assert result is False

    def test_block_policy_succeeds_when_space(self):
        buf = PipelineBuffer(max_size=2, timeout=1.0, overflow_policy="block")
        buf.put({"a": 1})
        buf.put({"b": 2})

        # 別スレッドで少し待ってから get して空きを作る
        def consumer():
            time.sleep(0.05)
            buf.get()

        t = threading.Thread(target=consumer)
        t.start()
        result = buf.put({"c": 3})
        t.join()
        assert result is True


class TestPipelineBufferStats:
    def test_stats_initial(self):
        buf = PipelineBuffer()
        stats = buf.stats
        assert stats["dropped"] == 0
        assert stats["total_put"] == 0
        assert stats["total_get"] == 0
        assert stats["max_consecutive_drops"] == 0

    def test_consecutive_drops_tracked(self):
        buf = PipelineBuffer(max_size=1, overflow_policy="drop_oldest")
        for i in range(10):
            buf.put({"i": i})
        stats = buf.stats
        assert stats["max_consecutive_drops"] >= 1