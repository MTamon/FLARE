"""LHGモデルインターフェースの抽象基底クラス。

音声特徴量と話者動作からリスナー動作を予測するLHGモデルの共通インターフェースを
定義する。全てのLHGモデル実装（Learning to Listen等）はBaseLHGModelを継承し、
抽象メソッドを実装しなければならない。

v2.0変更点:
    - ``predict()`` を1引数版から2引数版
      ``(audio_features, speaker_motion)`` に変更（元の1引数版は廃止）。
    - ``requires_window`` プロパティの追加により、パイプラインが
      バッファリング戦略を自動切替可能に。

設計根拠:
    L2Lをはじめ主要LHGモデルは音声特徴量と話者動作を別チャネルで受け取る。
    2引数版に統一することで責務を明確化する。requires_windowプロパティにより、
    パイプラインがバッファリング戦略を自動切替する。

Example:
    BaseLHGModelを継承した具体クラスの実装::

        class L2LModel(BaseLHGModel):
            def predict(self, audio_features, speaker_motion):
                # VQ-VAEベースのリスナー動作予測
                ...

            @property
            def requires_window(self) -> bool:
                return True

            @property
            def window_size(self) -> int:
                return 64
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseLHGModel(ABC):
    """LHGモデルの抽象基底クラス。

    音声特徴量と話者動作の2入力からリスナー動作を予測する。
    ウィンドウレベル（複数フレーム一括）とフレームレベル（1フレームずつ）の
    両方の入力形式に対応し、requires_windowプロパティで切り替える。

    典型的な使用フロー:
        1. ``requires_window`` / ``window_size`` を確認してバッファリング戦略を決定
        2. ``predict(audio_features, speaker_motion)`` でリスナー動作を予測

    Attributes:
        requires_window: ウィンドウレベル入力が必要かどうか。
        window_size: ウィンドウレベルの場合のフレーム数。
    """

    @abstractmethod
    def predict(
        self,
        audio_features: torch.Tensor,
        speaker_motion: torch.Tensor,
    ) -> torch.Tensor:
        """音声特徴量と話者動作からリスナー動作を予測する。

        Args:
            audio_features: 音声特徴量テンソル。
                ウィンドウレベル: ``(B, T_a, D_a)``、
                フレームレベル: ``(B, D_a)``。
                T_aは音声フレーム数、D_aは音声特徴量の次元数。
            speaker_motion: 話者の動作パラメータテンソル。
                ウィンドウレベル: ``(B, T_s, D_s)``、
                フレームレベル: ``(B, D_s)``。
                T_sは動作フレーム数、D_sは動作パラメータの次元数。

        Returns:
            予測されたリスナー動作テンソル。
            ウィンドウレベル: ``(B, T_out, D_out)``、
            フレームレベル: ``(B, D_out)``。
            T_outは出力フレーム数、D_outは出力パラメータの次元数。

        Raises:
            RuntimeError: モデル推論に失敗した場合。
        """

    @property
    @abstractmethod
    def requires_window(self) -> bool:
        """ウィンドウレベル入力が必要かどうかを返す。

        Returns:
            Trueの場合、predict()はウィンドウレベル入力 ``(B, T, D)`` を期待する。
            Falseの場合、フレームレベル入力 ``(B, D)`` を期待する。
        """

    @property
    @abstractmethod
    def window_size(self) -> Optional[int]:
        """ウィンドウレベル入力のフレーム数を返す。

        Returns:
            ``requires_window`` がTrueの場合、ウィンドウに含まれるフレーム数
            （例: L2Lでは64）。Falseの場合はNone。
        """
