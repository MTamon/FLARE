"""BaseExtractor ABC (torch.Tensor 版)

Section 8.2 BaseExtractor:
- extract(): 1 フレームから 3DMM パラメータを抽出 → Dict[str, Tensor]
- extract_batch(): デフォルト実装はループ。サブクラスでバッチ最適化可能 (Section 2.3)
- param_dim: 出力パラメータの総次元数
- param_keys: 出力 Dict のキーリスト
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import torch


class BaseExtractor(ABC):
    """3DMM パラメータ抽出の抽象基底クラス。

    サブクラスは extract(), param_dim, param_keys を実装する。
    extract_batch() はデフォルトでループ版が提供されるが、
    サブクラスでバッチ最適化のためにオーバーライドしてよい。
    """

    @abstractmethod
    def extract(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """1 フレーム（またはバッチサイズ 1）から 3DMM パラメータを抽出する。

        Args:
            image: 顔クロップ済みテンソル (1, C, H, W)

        Returns:
            キー名と Tensor の辞書。キー構成は param_keys に従う。
        """
        ...

    def extract_batch(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """バッチ処理。デフォルトはループ実装。サブクラスで最適化可能。

        Args:
            images: (B, C, H, W)

        Returns:
            各キーが (B, D_k) の Tensor を持つ辞書。
        """
        results: Dict[str, list] = {}
        for i in range(images.shape[0]):
            single = self.extract(images[i : i + 1])
            for k, v in single.items():
                results.setdefault(k, []).append(v)
        return {k: torch.cat(v, dim=0) for k, v in results.items()}

    @property
    @abstractmethod
    def param_dim(self) -> int:
        """出力パラメータの総次元数。"""
        ...

    @property
    @abstractmethod
    def param_keys(self) -> List[str]:
        """出力 Dict のキーリスト。"""
        ...