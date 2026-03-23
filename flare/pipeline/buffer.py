"""パイプラインバッファ。

仕様書8.4節「buffer.py: パラメータバッファ仕様」に基づき、
パイプラインのステージ間でフレームデータを受け渡すための
スレッドセーフなキューバッファを提供する。

リアルタイムモードでは ``DROP_OLDEST`` ポリシーにより最新フレームを優先し、
バッチモードでは ``BLOCK`` ポリシーにより全フレームを確実に処理する。

Example:
    >>> buffer = PipelineBuffer(max_size=256, overflow_policy="drop_oldest")
    >>> buffer.put({"frame": frame_tensor, "index": 0})
    >>> data = buffer.get()
    >>> print(buffer.get_stats())
"""

from __future__ import annotations

import queue
from enum import Enum
from typing import Any, Dict, Optional

from loguru import logger


class FrameDropPolicy(Enum):
    """フレームドロップポリシー。

    リアルタイム処理において処理遅延が発生した場合のバッファ動作を定義する。

    Attributes:
        DROP_OLDEST: 最古フレームを破棄して最新フレームを格納する。
            リアルタイムモードのデフォルト。「最新フレーム優先」方式。
        BLOCK: バッファに空きができるまでブロックする。
            バッチモード用。全フレームを確実に処理する。
        INTERPOLATE: ドロップ後に補間で滑らか化する（将来拡張用）。
            現時点では ``DROP_OLDEST`` と同等の動作。
    """

    DROP_OLDEST = "drop_oldest"
    BLOCK = "block"
    INTERPOLATE = "interpolate"


class PipelineBuffer:
    """パイプラインステージ間のスレッドセーフなフレームバッファ。

    Python ``queue.Queue`` をベースに、オーバーフローポリシーと
    統計情報収集機能を備えたバッファを提供する。

    Attributes:
        _queue: 内部キュー。
        _timeout: get()のデフォルトタイムアウト秒数。
        _policy: オーバーフロー時のフレームドロップポリシー。
        _stats: ドロップ数・put/get総数の統計情報。

    Example:
        >>> buf = PipelineBuffer(max_size=128, timeout=0.5, overflow_policy="drop_oldest")
        >>> buf.put({"params": tensor, "frame_idx": 42})
        True
        >>> data = buf.get()
        >>> buf.get_stats()
        {'dropped': 0, 'total_put': 1, 'total_get': 1}
    """

    def __init__(
        self,
        max_size: int = 256,
        timeout: float = 0.5,
        overflow_policy: str = "drop_oldest",
    ) -> None:
        """PipelineBufferを初期化する。

        Args:
            max_size: バッファの最大フレーム数。デフォルト256。
            timeout: get()操作のデフォルトタイムアウト秒数。デフォルト0.5。
            overflow_policy: オーバーフロー時の方針。
                ``"drop_oldest"``（リアルタイム用）、
                ``"block"``（バッチ用）、
                ``"interpolate"``（将来拡張、現在はdrop_oldest同等）。
        """
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=max_size)
        self._timeout: float = timeout
        self._policy: FrameDropPolicy = FrameDropPolicy(overflow_policy)
        self._max_size: int = max_size
        self._stats: Dict[str, int] = {
            "dropped": 0,
            "total_put": 0,
            "total_get": 0,
        }

    def put(self, frame_data: Dict[str, Any]) -> bool:
        """フレームデータをバッファに格納する。

        オーバーフローポリシーに従ってキューへの格納を行う。

        Args:
            frame_data: 格納するフレームデータ。パラメータDict等。

        Returns:
            格納に成功した場合 ``True``。

        Example:
            >>> success = buffer.put({"frame": tensor, "index": 0})
        """
        if self._policy == FrameDropPolicy.BLOCK:
            self._queue.put(frame_data)
            self._stats["total_put"] += 1
            return True

        # DROP_OLDEST / INTERPOLATE
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._stats["dropped"] += 1
            logger.debug(
                "バッファオーバーフロー: 最古フレーム破棄 | "
                "dropped={}",
                self._stats["dropped"],
            )

        self._queue.put_nowait(frame_data)
        self._stats["total_put"] += 1
        return True

    def get(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """フレームデータをバッファから取得する。

        Args:
            timeout: タイムアウト秒数。Noneの場合はコンストラクタで
                指定したデフォルト値を使用する。

        Returns:
            取得したフレームデータ。タイムアウトした場合は ``None``。

        Example:
            >>> data = buffer.get(timeout=1.0)
            >>> if data is not None:
            ...     process(data)
        """
        effective_timeout: float = timeout if timeout is not None else self._timeout

        try:
            data: Dict[str, Any] = self._queue.get(timeout=effective_timeout)
            self._stats["total_get"] += 1
            return data
        except queue.Empty:
            return None

    def qsize(self) -> int:
        """現在のキューサイズを返す。

        Returns:
            キュー内のアイテム数。
        """
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """バッファが空かどうかを返す。

        Returns:
            バッファが空の場合 ``True``。
        """
        return self._queue.empty()

    def is_full(self) -> bool:
        """バッファが満杯かどうかを返す。

        Returns:
            バッファが満杯の場合 ``True``。
        """
        return self._queue.full()

    def get_stats(self) -> Dict[str, int]:
        """統計情報のコピーを返す。

        Returns:
            以下のキーを含むDict:
                - ``dropped``: ドロップされたフレーム数
                - ``total_put``: put()が呼ばれた総回数
                - ``total_get``: get()で取得に成功した総回数

        Example:
            >>> buffer.get_stats()
            {'dropped': 5, 'total_put': 1000, 'total_get': 995}
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        """統計情報をゼロリセットする。"""
        self._stats["dropped"] = 0
        self._stats["total_put"] = 0
        self._stats["total_get"] = 0

    def clear(self) -> None:
        """キューを空にする。

        キュー内の全アイテムを破棄する。統計情報は変更しない。
        """
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
