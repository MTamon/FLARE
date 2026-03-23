"""AdapterRegistry: Adapterの自動選択レジストリ。

config.yamlのextractor.typeとrenderer.typeの組み合わせから
適切なAdapterを自動選択するシングルトンレジストリを提供する。

登録キーは ``"{source_format}_to_{target_format}"`` の形式で生成される。
例: ``"deca_to_flash_avatar"``、``"flame_to_pirender"``、``"identity"``。

Example:
    >>> registry = AdapterRegistry.get_instance()
    >>>
    >>> @registry.register
    ... class DECAToFlameAdapter(BaseAdapter):
    ...     ...
    >>>
    >>> adapter = registry.get("deca", "flash_avatar")
    >>> # または auto_select で自動選択
    >>> adapter = registry.auto_select("deca", "flash_avatar")
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Type

from flare.converters.base import BaseAdapter


class AdapterRegistry:
    """Adapterクラスの登録・検索を行うシングルトンレジストリ。

    スレッドセーフなシングルトンパターンを採用し、アプリケーション全体で
    単一のレジストリインスタンスを共有する。

    ``register()`` でAdapterクラスを登録し、``get()`` または ``auto_select()``
    で適切なAdapterインスタンスを取得する。

    Attributes:
        _instance: シングルトンインスタンス。
        _lock: インスタンス生成時のスレッドセーフティ用ロック。
        _adapters: 登録済みAdapterクラスのDict。
            キーは ``"{source}_to_{target}"`` 形式。

    Example:
        >>> registry = AdapterRegistry.get_instance()
        >>> registry.list_adapters()
        ['deca_to_flash_avatar', 'identity_to_identity']
    """

    _instance: Optional[AdapterRegistry] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """レジストリを初期化する。

        直接インスタンス化せず、``get_instance()`` を使用すること。

        Raises:
            RuntimeError: ``get_instance()`` を経由せず直接呼び出した場合。
                ただし初回の内部呼び出しは許可される。
        """
        self._adapters: Dict[str, Type[BaseAdapter]] = {}

    @classmethod
    def get_instance(cls) -> AdapterRegistry:
        """シングルトンインスタンスを返す。

        スレッドセーフにシングルトンインスタンスを生成・取得する。
        初回呼び出し時にインスタンスを生成し、以降は同一インスタンスを返す。

        Returns:
            AdapterRegistryのシングルトンインスタンス。

        Example:
            >>> registry = AdapterRegistry.get_instance()
            >>> registry is AdapterRegistry.get_instance()  # True
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """シングルトンインスタンスをリセットする。

        テスト時にレジストリを初期状態に戻すために使用する。
        本番コードでは通常使用しない。
        """
        with cls._lock:
            cls._instance = None

    def register(self, adapter_cls: Type[BaseAdapter]) -> Type[BaseAdapter]:
        """Adapterクラスをレジストリに登録する。

        クラスデコレータとしても使用できる。一時的にインスタンスを生成して
        ``source_format`` と ``target_format`` を取得し、
        ``"{source}_to_{target}"`` をキーとして登録する。

        Args:
            adapter_cls: 登録するBaseAdapterのサブクラス。

        Returns:
            引数で受け取ったadapter_clsをそのまま返す（デコレータ対応）。

        Raises:
            TypeError: adapter_clsがBaseAdapterのサブクラスでない場合。

        Example:
            >>> @registry.register
            ... class MyAdapter(BaseAdapter):
            ...     @property
            ...     def source_format(self) -> str:
            ...         return "deca"
            ...     @property
            ...     def target_format(self) -> str:
            ...         return "flash_avatar"
            ...     def convert(self, source_params):
            ...         ...
        """
        if not (isinstance(adapter_cls, type) and issubclass(adapter_cls, BaseAdapter)):
            raise TypeError(
                f"登録対象はBaseAdapterのサブクラスである必要があります。"
                f"受け取った型: {type(adapter_cls)}"
            )

        temp_instance: BaseAdapter = adapter_cls()
        key: str = self._make_key(
            temp_instance.source_format, temp_instance.target_format
        )
        self._adapters[key] = adapter_cls
        return adapter_cls

    def get(self, source_format: str, target_format: str) -> BaseAdapter:
        """登録済みAdapterのインスタンスを返す。

        ``source_format`` と ``target_format`` の組み合わせに対応する
        Adapterクラスをインスタンス化して返す。

        Args:
            source_format: 変換元フォーマット名。例: ``"deca"``。
            target_format: 変換先フォーマット名。例: ``"flash_avatar"``。

        Returns:
            対応するAdapterの新規インスタンス。

        Raises:
            KeyError: 対応するAdapterが未登録の場合。
                エラーメッセージに登録済みキー一覧を含む。

        Example:
            >>> adapter = registry.get("deca", "flash_avatar")
            >>> adapter.source_format  # "deca"
        """
        key: str = self._make_key(source_format, target_format)
        adapter_cls: Optional[Type[BaseAdapter]] = self._adapters.get(key)

        if adapter_cls is None:
            available: List[str] = self.list_adapters()
            raise KeyError(
                f"Adapterが見つかりません: '{key}'。"
                f"登録済みAdapter: {available}"
            )

        return adapter_cls()

    def list_adapters(self) -> List[str]:
        """登録済みAdapterのキー一覧を返す。

        Returns:
            ``"{source}_to_{target}"`` 形式のキー文字列のリスト。

        Example:
            >>> registry.list_adapters()
            ['deca_to_flash_avatar', 'identity_to_identity']
        """
        return list(self._adapters.keys())

    def auto_select(
        self, extractor_type: str, renderer_type: str
    ) -> BaseAdapter:
        """Extractor/Rendererの型名から適切なAdapterを自動選択する。

        config.yamlの ``extractor.type`` と ``renderer.type`` から
        対応するAdapterを検索する。完全一致が見つからない場合は
        ``"identity"`` アダプタへのフォールバックを試みる。

        Args:
            extractor_type: Extractorの種別名。例: ``"deca"``、``"deep3d"``。
            renderer_type: Rendererの種別名。例: ``"flash_avatar"``、``"pirender"``。

        Returns:
            選択されたAdapterの新規インスタンス。

        Raises:
            KeyError: 対応するAdapterが見つからず、identityアダプタも
                未登録の場合。

        Example:
            >>> adapter = registry.auto_select("deca", "flash_avatar")
            >>> adapter.source_format  # "deca"
        """
        key: str = self._make_key(extractor_type, renderer_type)

        # 完全一致で検索
        if key in self._adapters:
            return self._adapters[key]()

        # identity フォールバック
        identity_key: str = self._make_key("identity", "identity")
        if identity_key in self._adapters:
            return self._adapters[identity_key]()

        available: List[str] = self.list_adapters()
        raise KeyError(
            f"Adapterが見つかりません: '{key}'。"
            f"identityアダプタも未登録です。"
            f"登録済みAdapter: {available}"
        )

    @staticmethod
    def _make_key(source_format: str, target_format: str) -> str:
        """source_formatとtarget_formatからレジストリキーを生成する。

        Args:
            source_format: 変換元フォーマット名。
            target_format: 変換先フォーマット名。

        Returns:
            ``"{source}_to_{target}"`` 形式のキー文字列。
        """
        return f"{source_format}_to_{target_format}"
