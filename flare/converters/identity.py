"""IdentityAdapter: パラメータ変換不要時のパススルーアダプタ。

Extractorの出力形式とRendererの入力形式が一致している場合に使用する。
source_paramsをそのまま（シャローコピーで）返す。

AdapterRegistryのauto_select()で適切なAdapterが見つからない場合の
フォールバックとしても使用される。

Example:
    >>> adapter = IdentityAdapter()
    >>> result = adapter.convert(params)
    >>> result is not params  # シャローコピー
    True
    >>> result["exp"] is params["exp"]  # テンソル自体は共有
    True
"""

from __future__ import annotations

from typing import Dict

import torch

from flare.converters.base import BaseAdapter
from flare.converters.registry import AdapterRegistry

_registry: AdapterRegistry = AdapterRegistry.get_instance()


@_registry.register
class IdentityAdapter(BaseAdapter):
    """パラメータ変換不要時のパススルーアダプタ。

    入力Dictをそのままシャローコピーして返す。
    source_formatとtarget_formatはコンストラクタ引数で指定可能であり、
    デフォルトではともに ``"identity"`` となる。

    Attributes:
        _source_fmt: 変換元フォーマット名。
        _target_fmt: 変換先フォーマット名。

    Example:
        >>> adapter = IdentityAdapter()
        >>> adapter.source_format  # "identity"
        >>> adapter.target_format  # "identity"
        >>>
        >>> # カスタムフォーマット名で使用
        >>> adapter = IdentityAdapter(source_fmt="smirk", target_fmt="smirk")
    """

    def __init__(
        self,
        source_fmt: str = "identity",
        target_fmt: str = "identity",
    ) -> None:
        """IdentityAdapterを初期化する。

        Args:
            source_fmt: 変換元フォーマット名。デフォルト ``"identity"``。
            target_fmt: 変換先フォーマット名。デフォルト ``"identity"``。
        """
        self._source_fmt: str = source_fmt
        self._target_fmt: str = target_fmt

    @property
    def source_format(self) -> str:
        """変換元フォーマット名を返す。

        Returns:
            コンストラクタで指定されたsource_fmt。
        """
        return self._source_fmt

    @property
    def target_format(self) -> str:
        """変換先フォーマット名を返す。

        Returns:
            コンストラクタで指定されたtarget_fmt。
        """
        return self._target_fmt

    def convert(
        self, source_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """パラメータをそのまま返す（シャローコピー）。

        Dictオブジェクト自体は新しいインスタンスを返すが、
        内部のテンソル参照は共有される（不要なメモリコピーを避ける）。

        Args:
            source_params: 入力パラメータDict。

        Returns:
            source_paramsのシャローコピー。
        """
        return dict(source_params)
