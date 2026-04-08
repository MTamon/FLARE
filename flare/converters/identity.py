"""Identity（パススルー）変換モジュール。

パラメータ形式が変換不要な場合に使用するアダプタ。
入力辞書をそのまま返す。converter_chainに組み込むことで、
チェーンの構造を維持しつつ変換をスキップできる。

Example:
    IdentityAdapterの使用::

        adapter = IdentityAdapter()
        output = adapter.convert({"expr": tensor, "pose": tensor})
        # output is the same dict as input
"""

from __future__ import annotations

import torch

from flare.converters.base import BaseAdapter


class IdentityAdapter(BaseAdapter):
    """パススルー変換アダプタ。

    入力パラメータ辞書をそのまま出力として返す。
    パラメータ形式が既にターゲット形式と一致している場合や、
    converter_chain内でプレースホルダとして使用する。

    source_formatおよびtarget_formatは共に ``"any"`` であり、
    任意の形式ペアに対してマッチする。
    """

    def convert(
        self, source_params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """入力辞書をそのまま返す。

        Args:
            source_params: 任意のパラメータ辞書。

        Returns:
            入力と同一の辞書（コピーなし）。
        """
        return source_params

    @property
    def source_format(self) -> str:
        """変換元のパラメータ形式名を返す。

        Returns:
            ``"any"``。任意の形式にマッチ。
        """
        return "any"

    @property
    def target_format(self) -> str:
        """変換先のパラメータ形式名を返す。

        Returns:
            ``"any"``。任意の形式にマッチ。
        """
        return "any"
