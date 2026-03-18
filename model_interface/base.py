"""BaseLHGModel ABC (2 引数版)

Section 8.2 BaseLHGModel:
- predict(audio_features, speaker_motion): 音声特徴量 + 話者動作からリスナー動作を予測
- requires_window: ウィンドウレベル入力か、フレームレベル入力かを示す
- window_size: ウィンドウレベルの場合のフレーム数（例: L2L なら 64）

設計根拠 (Section 8.2):
  L2L をはじめ主要 LHG モデルは音声特徴量と話者動作を別チャネルで受け取る。
  元仕様の 1 引数版は廃止し、2 引数版に統一することで責務を明確化する。
  requires_window プロパティにより、パイプラインがバッファリング戦略を自動切替する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseLHGModel(ABC):
    """LHG モデル推論の抽象基底クラス。"""

    @abstractmethod
    def predict(
        self,
        audio_features: torch.Tensor,   # (B, T_a, D_a) or (B, D_a)
        speaker_motion: torch.Tensor,    # (B, T_s, D_s) or (B, D_s)
    ) -> torch.Tensor:                   # (B, T_out, D_out) or (B, D_out)
        """音声特徴量 + 話者動作からリスナー動作を予測する。

        Args:
            audio_features: 音声特徴量テンソル。
            speaker_motion: 話者の 3DMM パラメータ系列テンソル。

        Returns:
            予測されたリスナーの動作パラメータテンソル。
        """
        ...

    @property
    @abstractmethod
    def requires_window(self) -> bool:
        """True → ウィンドウレベル入力、False → フレームレベル入力。"""
        ...

    @property
    @abstractmethod
    def window_size(self) -> Optional[int]:
        """ウィンドウレベルの場合のフレーム数（例: L2L なら 64）。

        フレームレベル (requires_window=False) の場合は None を返す。
        """
        ...