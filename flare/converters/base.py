"""BaseAdapter ABC

Section 8.2 BaseAdapter (v2.0: converters/ に移動):
- convert(): パラメータ形式を変換する
- source_format: 変換元の形式名
- target_format: 変換先の形式名

converters/ モジュールの責務 (Section 8.2):
  パラメータ空間の変換（DECA→FLAME 等）。
  Dict[str, Tensor] → Dict[str, Tensor]。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch


class BaseAdapter(ABC):
    """パラメータ形式変換の抽象基底クラス。"""

    @abstractmethod
    def convert(
        self,
        source_params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """パラメータ形式を変換する。

        Args:
            source_params: 変換元のパラメータ辞書。

        Returns:
            変換先の形式に合わせたパラメータ辞書。
        """
        ...

    @property
    @abstractmethod
    def source_format(self) -> str:
        """変換元の形式名（例: "deca"）。"""
        ...

    @property
    @abstractmethod
    def target_format(self) -> str:
        """変換先の形式名（例: "flash_avatar"）。"""
        ...