"""PipelineBuffer (Section 8.4)

Python の queue.Queue ベースの設定可能なバッファ。
Section 8.4 のサンプルコードに準拠し、以下を実装:
  - max_size: バッファの最大フレーム数 (default=256)
  - timeout_sec: get() のタイムアウト秒数 (default=0.5)
  - overflow_policy: "drop_oldest" | "block"
  - 統計情報 (dropped, total_put, total_get)

Section 6.4:
  リアルタイムモードでは DROP_OLDEST をデフォルトとし、
  バッチモードでは BLOCK を使用する。
  ドロップ発生時にはログに記録し、統計情報を収集する。
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, Literal, Optional

from loguru import logger


class PipelineBuffer:
    """スレッド間パラメータ受け渡し用バッファ。

    Section 8.4 の仕様に準拠。
    """

    def __init__(
        self,
        max_size: int = 256,
        timeout: float = 0.5,
        overflow_policy: Literal["drop_oldest", "block"] = "drop_oldest",
    ) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self._timeout = timeout
        self._policy = overflow_policy
        self._lock = threading.Lock()
        self._stats = {
            "dropped": 0,
            "total_put": 0,
            "total_get": 0,
            "max_consecutive_drops": 0,
        }
        self._consecutive_drops = 0

    # ------------------------------------------------------------------
    # put / get
    # ------------------------------------------------------------------

    def put(self, frame_data: Dict[str, Any]) -> bool:
        """フレームデータをバッファに投入する。

        Returns:
            True: 正常投入。
            False: ドロップが発生した場合も True を返す
                   （投入自体は成功する）。block ポリシーで
                   タイムアウトした場合のみ False。
        """
        with self._lock:
            self._stats["total_put"] += 1

        if self._queue.full():
            if self._policy == "drop_oldest":
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                with self._lock:
                    self._stats["dropped"] += 1
                    self._consecutive_drops += 1
                    self._stats["max_consecutive_drops"] = max(
                        self._stats["max_consecutive_drops"],
                        self._consecutive_drops,
                    )
                logger.debug(
                    f"Buffer overflow: dropped oldest frame "
                    f"(total dropped={self._stats['dropped']})"
                )
            elif self._policy == "block":
                try:
                    self._queue.put(frame_data, timeout=self._timeout)
                    self._reset_consecutive_drops()
                    return True
                except queue.Full:
                    logger.warning("Buffer put timed out (block policy)")
                    return False

        try:
            self._queue.put_nowait(frame_data)
            self._reset_consecutive_drops()
        except queue.Full:
            # drop_oldest で race condition が起きた場合のフォールバック
            logger.warning("Buffer put failed after drop_oldest (race)")
            return False
        return True

    def get(self) -> Optional[Dict[str, Any]]:
        """フレームデータを取得する。タイムアウト時は None。"""
        try:
            data = self._queue.get(timeout=self._timeout)
            with self._lock:
                self._stats["total_get"] += 1
            return data
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # 統計・状態
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, int]:
        """統計情報のコピーを返す。"""
        with self._lock:
            return dict(self._stats)

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    @property
    def full(self) -> bool:
        return self._queue.full()

    def clear(self) -> int:
        """バッファを空にして、破棄したアイテム数を返す。"""
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        return count

    def log_stats(self) -> None:
        """統計をログに出力する。"""
        s = self.stats
        logger.info(
            f"Buffer stats: put={s['total_put']} get={s['total_get']} "
            f"dropped={s['dropped']} max_consecutive_drops={s['max_consecutive_drops']} "
            f"current_size={self.qsize}"
        )

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _reset_consecutive_drops(self) -> None:
        with self._lock:
            self._consecutive_drops = 0