"""BaseAdapter: パラメータ変換の抽象基底クラス。

仕様書8.2節「BaseAdapter（v2.0: converters/に移動）」に基づき、
Extractor出力形式とRenderer入力形式の間のパラメータ変換を行う
Adapterの統一インターフェースを定義する。

converters/モジュールの責務分離:
    ============== ====================================== ==============================
    モジュール       責務                                    入出力
    ============== ====================================== ==============================
    converters/    パラメータ空間の変換（DECA→FLAME等）     Dict[str, Tensor] → Dict[str, Tensor]
    extractors/    画像→3DMMパラメータ抽出                  image → Dict[str, Tensor]
    renderers/     3DMMパラメータ→画像レンダリング           Dict[str, Tensor] → image
    ============== ====================================== ==============================

Example:
    >>> class DECAToFlameAdapter(BaseAdapter):
    ...     def convert(self, source_params):
    ...         expr_100d = F.pad(source_params['exp'], (0, 50), value=0.0)
    ...         return {'expr': expr_100d, ...}
    ...     @property
    ...     def source_format(self) -> str:
    ...         return "deca"
    ...     @property
    ...     def target_format(self) -> str:
    ...         return "flash_avatar"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch


class BaseAdapter(ABC):
    """パラメータ変換の抽象基底クラス。

    全てのAdapter実装（DECAToFlameAdapter, FlameToPIRenderAdapter,
    IdentityAdapter等）はこのクラスを継承し、``convert()``・
    ``source_format``・``target_format`` を実装する。

    AdapterRegistryにより、``source_format`` と ``target_format`` の
    組み合わせから適切なAdapterが自動選択される。

    Example:
        >>> adapter = DECAToFlameAdapter()
        >>> flame_params = adapter.convert(deca_output)
        >>> # または callable ショートカット
        >>> flame_params = adapter(deca_output)
    """

    @abstractmethod
    def convert(
        self, source_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """パラメータ形式を変換する。

        変換元Extractorの出力Dict を、変換先Rendererが受け付ける
        入力Dict 形式に変換する。

        Args:
            source_params: 変換元のパラメータDict。
                キーと次元数はExtractor実装に依存する。
                例（DECA）: ``{"exp": (B, 50), "pose": (B, 6), ...}``

        Returns:
            変換先のパラメータDict。
            キーと次元数はRenderer実装に依存する。
            例（FlashAvatar）: ``{"expr": (B, 100), "jaw_pose": (B, 6), ...}``
        """

    @property
    @abstractmethod
    def source_format(self) -> str:
        """変換元フォーマット名を返す。

        AdapterRegistryでのキー生成に使用される。

        Returns:
            変換元のフォーマット識別子。例: ``"deca"``、``"deep3d"``。
        """

    @property
    @abstractmethod
    def target_format(self) -> str:
        """変換先フォーマット名を返す。

        AdapterRegistryでのキー生成に使用される。

        Returns:
            変換先のフォーマット識別子。例: ``"flash_avatar"``、``"pirender"``。
        """

    def __call__(
        self, source_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """convert()を呼び出すショートカット。

        Adapterインスタンスを関数のように呼び出せるようにする。

        Args:
            source_params: 変換元のパラメータDict。

        Returns:
            変換後のパラメータDict。

        Example:
            >>> adapter = DECAToFlameAdapter()
            >>> result = adapter(deca_output)  # adapter.convert(deca_output) と同等
        """
        return self.convert(source_params)
