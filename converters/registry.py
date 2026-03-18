"""AdapterRegistry: config の extractor.type / renderer.type の組み合わせから
適切な Adapter を自動選択する。

Section 8.2:
  AdapterRegistry パターンにより、config.yaml の extractor.type と
  renderer.type の組み合わせから適切な Adapter を自動選択する。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Type

from lhg_toolkit.converters.base import BaseAdapter


class AdapterRegistry:
    """(source_format, target_format) → Adapter クラスのレジストリ。

    使用例::

        registry = AdapterRegistry()

        # 登録（デコレータ）
        @registry.register("deca", "flash_avatar")
        class DECAToFlameAdapter(BaseAdapter):
            ...

        # 取得
        adapter_cls = registry.get("deca", "flash_avatar")
        adapter = adapter_cls()

        # 同一形式の場合は IdentityAdapter を返す
        adapter_cls = registry.get("flash_avatar", "flash_avatar")
    """

    def __init__(self) -> None:
        self._registry: Dict[Tuple[str, str], Type[BaseAdapter]] = {}
        self._identity_factory: Optional[Callable[[], BaseAdapter]] = None

    # ------------------------------------------------------------------
    # 登録
    # ------------------------------------------------------------------

    def register(
        self,
        source_format: str,
        target_format: str,
    ) -> Callable[[Type[BaseAdapter]], Type[BaseAdapter]]:
        """クラスデコレータとして使用する登録メソッド。"""

        def decorator(cls: Type[BaseAdapter]) -> Type[BaseAdapter]:
            key = (source_format.lower(), target_format.lower())
            if key in self._registry:
                raise ValueError(
                    f"Adapter already registered for {key}: "
                    f"{self._registry[key].__name__}"
                )
            self._registry[key] = cls
            return cls

        return decorator

    def register_class(
        self,
        source_format: str,
        target_format: str,
        cls: Type[BaseAdapter],
    ) -> None:
        """命令的に Adapter クラスを登録する。"""
        key = (source_format.lower(), target_format.lower())
        if key in self._registry:
            raise ValueError(
                f"Adapter already registered for {key}: "
                f"{self._registry[key].__name__}"
            )
        self._registry[key] = cls

    def set_identity_factory(self, factory: Callable[[], BaseAdapter]) -> None:
        """source_format == target_format の場合に返す IdentityAdapter の
        ファクトリを設定する。"""
        self._identity_factory = factory

    # ------------------------------------------------------------------
    # 取得
    # ------------------------------------------------------------------

    def get(
        self,
        source_format: str,
        target_format: str,
    ) -> Type[BaseAdapter]:
        """(source_format, target_format) に対応する Adapter クラスを返す。

        同一形式の場合は IdentityAdapter クラスを返す。

        Raises:
            KeyError: 該当する Adapter が登録されていない場合。
        """
        src = source_format.lower()
        tgt = target_format.lower()

        if src == tgt:
            if self._identity_factory is not None:
                return type(self._identity_factory())  # type: ignore[return-value]
            raise KeyError(
                f"No identity adapter registered. "
                f"Call set_identity_factory() first, or register ({src}, {tgt})."
            )

        key = (src, tgt)
        if key not in self._registry:
            raise KeyError(
                f"No adapter registered for ({src} → {tgt}). "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[key]

    def get_instance(
        self,
        source_format: str,
        target_format: str,
    ) -> BaseAdapter:
        """Adapter のインスタンスを直接返すヘルパー。"""
        cls = self.get(source_format, target_format)
        return cls()

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    def available(self) -> list[Tuple[str, str]]:
        """登録済みの (source, target) ペア一覧を返す。"""
        return list(self._registry.keys())


# グローバルレジストリインスタンス
# 各 Adapter モジュール (deca_to_flame.py 等) がインポート時に登録する
adapter_registry = AdapterRegistry()