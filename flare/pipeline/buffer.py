"""パラメータバッファモジュール。

Python の ``queue.Queue`` ベースの設定可能なバッファ。
リアルタイムパイプラインにおけるスレッド間のフレームデータ受け渡しに使用する。

仕様書§8.4に基づく3つのオーバーフローポリシーをサポートする:
    - ``drop_oldest``: 満杯時に最古のアイテムを破棄して新アイテムを追加。
    - ``block``: 満杯時に空きが出るまでブロック待機。
    - ``interpolate``: Phase 1 では ``drop_oldest`` と同等動作（将来の補間対応）。

Example:
    リアルタイムモードでの使用::

        buffer = PipelineBuffer(max_size=256, overflow_policy="drop_oldest")
        buffer.put({"frame": frame_tensor, "index": 0})
        data = buffer.get()
        if data is not None:
            process(data)
        print(buffer.stats)  # {"dropped": 0, "total_put": 1}
"""

from __future__ import annotations

import queue
from typing import Any, Optional


class PipelineBuffer:
    """スレッドセーフなパイプラインバッファ。

    ``queue.Queue`` をラップし、オーバーフローポリシーに応じた
    put/get 動作とドロップ統計の収集を提供する。

    Attributes:
        _queue: 内部キュー。
        _timeout: ``get()`` のタイムアウト秒数。
        _policy: オーバーフローポリシー文字列。
        _stats: ドロップ数・put総数の統計辞書。
    """

    def __init__(
        self,
        max_size: int = 256,
        timeout: float = 0.5,
        overflow_policy: str = "drop_oldest",
    ) -> None:
        """PipelineBufferを初期化する。

        Args:
            max_size: バッファの最大フレーム数。``queue.Queue`` の maxsize に対応。
            timeout: ``get()`` のタイムアウト秒数。タイムアウト後は ``None`` を返す。
            overflow_policy: オーバーフロー時の方針。以下のいずれか:
                - ``"drop_oldest"``: 最古アイテムを破棄して新アイテムを追加
                  （リアルタイムモードのデフォルト）。
                - ``"block"``: 空きが出るまでブロック待機
                  （バッチモード向け）。
                - ``"interpolate"``: Phase 1 では ``drop_oldest`` と同等動作。
                  将来のフェーズでドロップ後の補間処理を追加予定。

        Raises:
            ValueError: 未知の overflow_policy が指定された場合。
        """
        valid_policies = {"drop_oldest", "block", "interpolate"}
        if overflow_policy not in valid_policies:
            raise ValueError(
                f"Unknown overflow_policy: {overflow_policy!r}. "
                f"Must be one of {valid_policies}"
            )
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_size)
        self._timeout = timeout
        self._policy = overflow_policy
        self._stats = {"dropped": 0, "total_put": 0}

    def put(self, frame_data: dict[str, Any]) -> bool:
        """フレームデータをバッファに追加する。

        オーバーフローポリシーに応じた動作を行う:
            - ``drop_oldest``: キューが満杯の場合、``get_nowait()`` で最古の
              アイテムを取り出して破棄し、``stats["dropped"]`` をインクリメント
              した後、新アイテムを追加する。
            - ``block``: キューの ``put()`` でブロック待機する。
            - ``interpolate``: Phase 1 では ``drop_oldest`` と同等の動作を行う。

        Args:
            frame_data: 追加するフレームデータの辞書。
                例: ``{"frame": Tensor, "index": int, "timestamp": float}``。

        Returns:
            追加に成功した場合は ``True``。
        """
        if self._policy in ("drop_oldest", "interpolate"):
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._stats["dropped"] += 1
            self._queue.put_nowait(frame_data)
        elif self._policy == "block":
            self._queue.put(frame_data)

        self._stats["total_put"] += 1
        return True

    def get(self) -> Optional[dict[str, Any]]:
        """バッファからフレームデータを取得する。

        ``timeout`` 秒間待機し、データが取得できない場合は ``None`` を返す。

        Returns:
            フレームデータの辞書。タイムアウト時は ``None``。
        """
        try:
            return self._queue.get(timeout=self._timeout)
        except queue.Empty:
            return None

    @property
    def stats(self) -> dict[str, int]:
        """バッファの統計情報を返す。

        Returns:
            以下のキーを持つ辞書::

                {
                    "dropped": ドロップされたフレーム数（int）,
                    "total_put": put() の成功回数（int）,
                }
        """
        return dict(self._stats)

    @property
    def qsize(self) -> int:
        """現在のキュー内アイテム数を返す。

        Returns:
            キュー内のアイテム数。スレッド安全だが概算値。
        """
        return self._queue.qsize()

    @property
    def full(self) -> bool:
        """キューが満杯かどうかを返す。

        Returns:
            満杯なら ``True``。スレッド安全だが概算値。
        """
        return self._queue.full()

    @property
    def empty(self) -> bool:
        """キューが空かどうかを返す。

        Returns:
            空なら ``True``。スレッド安全だが概算値。
        """
        return self._queue.empty()
