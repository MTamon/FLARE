"""Adapterレジストリモジュール。

AdapterRegistryパターンにより、登録されたBaseAdapter実装の中から
(source_format, target_format)の組み合わせで適切なAdapterを自動選択する。
config.yamlのconverter_chainリストからAdapterインスタンスのチェーンを構築する
機能も提供する。

Example:
    Adapterの登録と取得::

        registry = AdapterRegistry()
        registry.register(DECAToFlameAdapter())
        adapter = registry.get("deca", "flash_avatar")
        converted = adapter.convert(deca_params)

    converter_chainからのチェーン構築::

        chain = registry.build_chain([
            {"type": "deca_to_flame"},
            {"type": "identity"},
        ])
        for adapter in chain:
            params = adapter.convert(params)
"""

from __future__ import annotations

from flare.converters.base import BaseAdapter


class AdapterRegistry:
    """Adapterの登録・検索を管理するレジストリ。

    (source_format, target_format)のタプルをキーとしてAdapterインスタンスを管理する。
    また、Adapterのtype名（例: "deca_to_flame"）からの検索もサポートし、
    config.yamlのconverter_chainリストからAdapterチェーンを構築できる。

    Attributes:
        _adapters: (source_format, target_format) → BaseAdapter のマッピング。
        _by_type: type名 → BaseAdapter のマッピング。
    """

    def __init__(self) -> None:
        """AdapterRegistryを初期化する。"""
        self._adapters: dict[tuple[str, str], BaseAdapter] = {}
        self._by_type: dict[str, BaseAdapter] = {}

    def register(self, adapter: BaseAdapter) -> None:
        """Adapterをレジストリに登録する。

        同一の(source_format, target_format)ペアが既に登録されている場合は上書きする。
        type名は ``"{source_format}_to_{target_format}"`` の形式で自動生成される。

        Args:
            adapter: 登録するBaseAdapterインスタンス。source_formatと
                target_formatプロパティが実装済みであること。
        """
        key = (adapter.source_format, adapter.target_format)
        self._adapters[key] = adapter
        type_name = f"{adapter.source_format}_to_{adapter.target_format}"
        self._by_type[type_name] = adapter

    def get(self, source_format: str, target_format: str) -> BaseAdapter:
        """指定された形式ペアに対応するAdapterを取得する。

        Args:
            source_format: 変換元のパラメータ形式名。例: ``"deca"``。
            target_format: 変換先のパラメータ形式名。例: ``"flash_avatar"``。

        Returns:
            対応するBaseAdapterインスタンス。

        Raises:
            KeyError: 指定された形式ペアに対応するAdapterが登録されていない場合。
        """
        key = (source_format, target_format)
        if key not in self._adapters:
            registered = list(self._adapters.keys())
            raise KeyError(
                f"Adapter not found for ({source_format!r}, {target_format!r}). "
                f"Registered adapters: {registered}"
            )
        return self._adapters[key]

    def build_chain(
        self, converter_chain: list[dict[str, str]]
    ) -> list[BaseAdapter]:
        """config.yamlのconverter_chainリストからAdapterチェーンを構築する。

        converter_chainの各要素は ``{"type": "<adapter_type_name>"}`` の形式。
        type名は ``register()`` 時に自動生成された
        ``"{source_format}_to_{target_format}"`` と一致する必要がある。

        Args:
            converter_chain: Adapter定義のリスト。各要素は ``"type"`` キーを
                持つ辞書。例::

                    [
                        {"type": "deca_to_flash_avatar"},
                        {"type": "identity_to_identity"},
                    ]

        Returns:
            変換チェーンを構成するBaseAdapterインスタンスのリスト。
            チェーンが空の場合は空リストを返す。

        Raises:
            KeyError: 指定されたtype名に対応するAdapterが登録されていない場合。
            ValueError: converter_chainの要素に ``"type"`` キーが存在しない場合。
        """
        chain: list[BaseAdapter] = []
        for entry in converter_chain:
            if "type" not in entry:
                raise ValueError(
                    f"converter_chain entry must have a 'type' key, got: {entry!r}"
                )
            type_name = entry["type"]
            if type_name not in self._by_type:
                registered = list(self._by_type.keys())
                raise KeyError(
                    f"Adapter type {type_name!r} not found. "
                    f"Registered types: {registered}"
                )
            chain.append(self._by_type[type_name])
        return chain
