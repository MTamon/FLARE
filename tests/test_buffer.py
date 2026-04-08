"""PipelineBuffer の各 overflow_policy テスト。

put/get の正常動作、overflow 時の stats 更新を検証する。
"""

from __future__ import annotations

import threading
import time

import pytest

from flare.pipeline.buffer import PipelineBuffer


class TestPipelineBufferDropOldest:
    """drop_oldest ポリシーのテスト。"""

    def test_basic_put_get(self) -> None:
        """基本的なput/getが正常に動作すること。"""
        buf = PipelineBuffer(max_size=10, timeout=0.1, overflow_policy="drop_oldest")
        buf.put({"value": 1})
        result = buf.get()
        assert result is not None
        assert result["value"] == 1

    def test_fifo_order(self) -> None:
        """FIFO順序が維持されること。"""
        buf = PipelineBuffer(max_size=10, timeout=0.1, overflow_policy="drop_oldest")
        for i in range(5):
            buf.put({"value": i})
        for i in range(5):
            result = buf.get()
            assert result is not None
            assert result["value"] == i

    def test_overflow_drops_oldest(self) -> None:
        """満杯時に最古のアイテムが破棄されること。"""
        buf = PipelineBuffer(max_size=3, timeout=0.1, overflow_policy="drop_oldest")
        buf.put({"value": 0})
        buf.put({"value": 1})
        buf.put({"value": 2})
        buf.put({"value": 3})

        result = buf.get()
        assert result is not None
        assert result["value"] == 1

    def test_stats_dropped_increments(self) -> None:
        """overflow時にstats['dropped']がインクリメントされること。"""
        buf = PipelineBuffer(max_size=2, timeout=0.1, overflow_policy="drop_oldest")
        buf.put({"a": 1})
        buf.put({"a": 2})
        assert buf.stats["dropped"] == 0

        buf.put({"a": 3})
        assert buf.stats["dropped"] == 1

        buf.put({"a": 4})
        assert buf.stats["dropped"] == 2

    def test_stats_total_put(self) -> None:
        """stats['total_put']が正しくカウントされること。"""
        buf = PipelineBuffer(max_size=2, timeout=0.1, overflow_policy="drop_oldest")
        buf.put({"a": 1})
        buf.put({"a": 2})
        buf.put({"a": 3})
        assert buf.stats["total_put"] == 3

    def test_get_timeout_returns_none(self) -> None:
        """空バッファからのgetがtimeout後にNoneを返すこと。"""
        buf = PipelineBuffer(max_size=10, timeout=0.05, overflow_policy="drop_oldest")
        result = buf.get()
        assert result is None


class TestPipelineBufferBlock:
    """block ポリシーのテスト。"""

    def test_block_put_waits(self) -> None:
        """満杯時にput()がブロックし、get()後に通ること。"""
        buf = PipelineBuffer(max_size=1, timeout=0.1, overflow_policy="block")
        buf.put({"x": 1})

        completed = threading.Event()

        def delayed_get() -> None:
            time.sleep(0.05)
            buf.get()
            completed.set()

        thread = threading.Thread(target=delayed_get)
        thread.start()
        buf.put({"x": 2})
        thread.join(timeout=2.0)

        assert completed.is_set()
        assert buf.stats["dropped"] == 0

    def test_block_no_drops(self) -> None:
        """blockポリシーではドロップが発生しないこと。"""
        buf = PipelineBuffer(max_size=5, timeout=0.1, overflow_policy="block")
        for i in range(5):
            buf.put({"v": i})
        assert buf.stats["dropped"] == 0


class TestPipelineBufferInterpolate:
    """interpolate ポリシーのテスト。"""

    def test_interpolate_behaves_like_drop_oldest(self) -> None:
        """Phase 1ではdrop_oldestと同等動作すること。"""
        buf = PipelineBuffer(max_size=2, timeout=0.1, overflow_policy="interpolate")
        buf.put({"v": 0})
        buf.put({"v": 1})
        buf.put({"v": 2})

        assert buf.stats["dropped"] == 1
        result = buf.get()
        assert result is not None
        assert result["v"] == 1


class TestPipelineBufferProperties:
    """バッファプロパティのテスト。"""

    def test_qsize(self) -> None:
        """qsizeが正しいアイテム数を返すこと。"""
        buf = PipelineBuffer(max_size=10, timeout=0.1)
        assert buf.qsize == 0
        buf.put({"a": 1})
        assert buf.qsize == 1
        buf.put({"a": 2})
        assert buf.qsize == 2
        buf.get()
        assert buf.qsize == 1

    def test_full(self) -> None:
        """fullが正しく判定されること。"""
        buf = PipelineBuffer(max_size=2, timeout=0.1)
        assert not buf.full
        buf.put({"a": 1})
        buf.put({"a": 2})
        assert buf.full

    def test_empty(self) -> None:
        """emptyが正しく判定されること。"""
        buf = PipelineBuffer(max_size=10, timeout=0.1)
        assert buf.empty
        buf.put({"a": 1})
        assert not buf.empty

    def test_invalid_policy_raises(self) -> None:
        """不正なポリシーでValueErrorが発生すること。"""
        with pytest.raises(ValueError, match="Unknown overflow_policy"):
            PipelineBuffer(overflow_policy="invalid")
