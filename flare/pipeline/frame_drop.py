"""フレームドロップポリシーモジュール。

リアルタイムパイプラインにおけるバッファオーバーフロー時の
フレーム処理戦略を定義する。

仕様書§6.4に基づく3つのポリシー:
    - ``DROP_OLDEST``: 最古のフレームを破棄して最新フレームを追加する。
      リアルタイム処理のデフォルトポリシー（Latest-Frame-Wins）。
    - ``BLOCK``: バッファに空きが出るまでブロック待機する。
      バッチモード向け。
    - ``INTERPOLATE``: ドロップ後に補間処理を行うポリシー。
      Phase 4ではDROP_OLDESTと同等の動作を行い、
      将来のフェーズで補間アルゴリズムを実装する。

FrameDropHandlerは各ポリシーに応じた具体的なバッファ操作を
統一インターフェースで提供する。

Example:
    FrameDropHandlerの使用::

        handler = FrameDropHandler()
        buffer = PipelineBuffer(max_size=2, overflow_policy="drop_oldest")
        success = handler.apply(
            buffer=buffer,
            policy=FrameDropPolicy.DROP_OLDEST,
            new_frame={"frame": tensor, "index": 42},
        )
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from flare.pipeline.buffer import PipelineBuffer


class FrameDropPolicy(Enum):
    """フレームドロップポリシーの列挙型。

    リアルタイムパイプラインのバッファオーバーフロー時に
    どのようにフレームを処理するかを定義する。

    仕様書§6.4で定義された3つの戦略に対応する。

    Attributes:
        DROP_OLDEST: 最古フレームを破棄して最新フレームを追加する。
        BLOCK: バッファに空きが出るまでブロック待機する。
        INTERPOLATE: ドロップ後に補間処理を行う（将来実装予定）。
    """

    DROP_OLDEST = "drop_oldest"
    BLOCK = "block"
    INTERPOLATE = "interpolate"


class FrameDropHandler:
    """フレームドロップポリシーに基づくバッファ操作ハンドラ。

    FrameDropPolicyに応じた具体的なバッファ操作を
    統一インターフェース ``apply()`` で提供する。

    各ポリシーの動作:
        - ``DROP_OLDEST``: PipelineBuffer.put()をそのまま呼び出す。
          PipelineBufferの``drop_oldest``ポリシーにより最古フレームが自動破棄される。
        - ``BLOCK``: PipelineBuffer.put()をそのまま呼び出す。
          PipelineBufferの``block``ポリシーにより空きが出るまで待機する。
        - ``INTERPOLATE``: Phase 4ではDROP_OLDESTと同等の動作を行う。
          将来のフェーズでドロップされたフレーム区間を検出し、
          前後のフレームから線形補間で中間フレームを生成する
          アルゴリズムを実装予定。

    Example:
        ハンドラの基本的な使用::

            handler = FrameDropHandler()
            success = handler.apply(buffer, FrameDropPolicy.DROP_OLDEST, frame_data)
    """

    def apply(
        self,
        buffer: PipelineBuffer,
        policy: FrameDropPolicy,
        new_frame: dict[str, Any],
    ) -> bool:
        """ポリシーに基づいてフレームをバッファに追加する。

        指定されたFrameDropPolicyに従い、新しいフレームデータを
        PipelineBufferに追加する。バッファオーバーフロー時の処理は
        PipelineBuffer内部のoverflow_policyと連携して行われる。

        Args:
            buffer: フレームデータを追加するPipelineBuffer。
                bufferのoverflow_policyとFrameDropPolicyが対応している
                ことが期待されるが、本メソッドはバッファの既存ポリシーに
                関わらず直接put()を呼び出す。
            policy: 適用するフレームドロップポリシー。
            new_frame: 追加するフレームデータの辞書。
                例: ``{"frame": ndarray, "index": int}``。

        Returns:
            フレームの追加に成功した場合は ``True``。
            DROP_OLDESTおよびINTERPOLATEでは常にTrueを返す。
            BLOCKではバッファに空きが出た後にTrueを返す。

        Note:
            INTERPOLATEポリシーは現在DROP_OLDESTと同等の動作を行う。
            将来のフェーズで、ドロップされたフレーム区間を検出し
            前後フレームから線形補間で中間フレームを生成する
            アルゴリズムを実装する予定である。
        """
        if policy == FrameDropPolicy.DROP_OLDEST:
            return buffer.put(new_frame)

        if policy == FrameDropPolicy.BLOCK:
            return buffer.put(new_frame)

        if policy == FrameDropPolicy.INTERPOLATE:
            # Phase 4: DROP_OLDESTと同等の動作を行う。
            # 将来のフェーズでドロップ検出と線形補間を実装予定。
            return buffer.put(new_frame)

        return buffer.put(new_frame)

    @staticmethod
    def policy_from_string(policy_str: str) -> FrameDropPolicy:
        """文字列からFrameDropPolicyを生成する。

        PipelineConfigのbuffer.overflow_policy文字列を
        FrameDropPolicy列挙値に変換するユーティリティ。

        Args:
            policy_str: ポリシー文字列。
                ``"drop_oldest"``, ``"block"``, ``"interpolate"`` のいずれか。

        Returns:
            対応するFrameDropPolicy列挙値。

        Raises:
            ValueError: 未知のポリシー文字列が指定された場合。
        """
        policy_map = {
            "drop_oldest": FrameDropPolicy.DROP_OLDEST,
            "block": FrameDropPolicy.BLOCK,
            "interpolate": FrameDropPolicy.INTERPOLATE,
        }
        if policy_str not in policy_map:
            raise ValueError(
                f"Unknown policy string: {policy_str!r}. "
                f"Must be one of {list(policy_map.keys())}"
            )
        return policy_map[policy_str]
