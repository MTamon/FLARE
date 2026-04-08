"""パラメータ変換の抽象基底クラス。

異なる3DMMパラメータ形式間の変換（DECA→FLAME等）を行うAdapterの
共通インターフェースを定義する。全てのAdapter実装はBaseAdapterを継承し、
抽象メソッドを実装しなければならない。

v2.0変更点:
    - ``adapter.py`` を ``model_interface/`` から ``converters/`` パッケージに移動し、
      パラメータ変換専用モジュールとして独立。
    - AdapterRegistryパターンにより、設定ファイルのextractor.typeとrenderer.typeの
      組み合わせから適切なAdapterを自動選択。

Example:
    BaseAdapterを継承した具体クラスの実装::

        class DECAToFlameAdapter(BaseAdapter):
            def convert(self, source_params):
                expr_100d = F.pad(source_params["exp"], (0, 50), value=0.0)
                ...
                return {"expr": expr_100d, ...}

            @property
            def source_format(self) -> str:
                return "deca"

            @property
            def target_format(self) -> str:
                return "flash_avatar"
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseAdapter(ABC):
    """パラメータ変換の抽象基底クラス。

    source_formatからtarget_formatへのパラメータ辞書の変換を行う。
    AdapterRegistryに登録することで、パイプラインから自動選択される。

    Attributes:
        source_format: 変換元のパラメータ形式名。
        target_format: 変換先のパラメータ形式名。
    """

    @abstractmethod
    def convert(
        self, source_params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """パラメータ形式を変換する。

        Args:
            source_params: 変換元のパラメータ辞書。キーはパラメータ名、
                値はテンソル。形式はsource_formatに依存する。
                例（DECA）::

                    {
                        "exp":  Tensor(B, 50),
                        "pose": Tensor(B, 6),
                    }

        Returns:
            変換後のパラメータ辞書。キーはパラメータ名、値はテンソル。
            形式はtarget_formatに依存する。
            例（FlashAvatar）::

                {
                    "expr":      Tensor(B, 100),
                    "jaw_pose":  Tensor(B, 6),
                    "eyes_pose": Tensor(B, 12),
                    "eyelids":   Tensor(B, 2),
                }

        Raises:
            KeyError: source_paramsに必要なキーが存在しない場合。
            RuntimeError: 変換処理に失敗した場合。
        """

    @property
    @abstractmethod
    def source_format(self) -> str:
        """変換元のパラメータ形式名を返す。

        Returns:
            形式名の文字列。例: ``"deca"``, ``"deep3d"``, ``"flash_avatar"``。
        """

    @property
    @abstractmethod
    def target_format(self) -> str:
        """変換先のパラメータ形式名を返す。

        Returns:
            形式名の文字列。例: ``"flash_avatar"``, ``"pirender"``, ``"deca"``。
        """
