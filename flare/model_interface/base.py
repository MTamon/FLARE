"""BaseLHGModel: LHGモデル推論の抽象基底クラス。

仕様書8.2節「BaseLHGModel（v2.0: 2引数版）」に基づき、
音声特徴量と話者動作からリスナー動作を予測するモデルの
統一インターフェースを定義する。

設計根拠:
    L2Lをはじめ主要LHGモデルは音声特徴量と話者動作を別チャネルで受け取る。
    元仕様の1引数版は廃止し、2引数版に統一することで責務を明確化する。
    ``requires_window`` プロパティにより、パイプラインがバッファリング戦略を
    自動切替する。

入力形式:
    - ウィンドウレベル（requires_window=True）: (B, T, D) の3次元テンソル
    - フレームレベル（requires_window=False）: (B, D) の2次元テンソル

Example:
    >>> class L2LModel(BaseLHGModel):
    ...     # 抽象メソッド・プロパティを全て実装
    ...     ...
    >>> model = L2LModel(device=torch.device("cuda:0"))
    >>> output = model.predict(audio_features, speaker_motion)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseLHGModel(ABC):
    """LHGモデル推論の抽象基底クラス。

    全てのLHGモデル実装（L2L, VQ-VAEベース等）はこのクラスを継承し、
    ``predict()``・``requires_window``・``window_size`` を実装する。

    ``requires_window`` により、パイプラインはウィンドウレベルの入力バッファリング
    （例: L2Lの64フレーム蓄積）とフレームレベルの逐次処理を自動で切り替える。

    Attributes:
        _device: モデルが配置されるCUDAデバイス。サブクラスの ``__init__`` で設定する。

    Example:
        >>> class MyLHGModel(BaseLHGModel):
        ...     def __init__(self, device: torch.device) -> None:
        ...         self._device = device
        ...
        ...     def predict(self, audio_features, speaker_motion):
        ...         self.validate_inputs(audio_features, speaker_motion)
        ...         # モデル推論処理...
        ...         return predicted_motion
        ...
        ...     @property
        ...     def requires_window(self) -> bool:
        ...         return True
        ...
        ...     @property
        ...     def window_size(self) -> Optional[int]:
        ...         return 64
    """

    _device: torch.device

    @abstractmethod
    def predict(
        self,
        audio_features: torch.Tensor,
        speaker_motion: torch.Tensor,
    ) -> torch.Tensor:
        """音声特徴量と話者動作からリスナー動作を予測する。

        Args:
            audio_features: 音声特徴量テンソル。
                ウィンドウレベル: shape ``(B, T_a, D_a)``。
                フレームレベル: shape ``(B, D_a)``。
                D_aは特徴量の次元数（mel: 128, HuBERT: 768 等）。
            speaker_motion: 話者の3DMM動作パラメータテンソル。
                ウィンドウレベル: shape ``(B, T_s, D_s)``。
                フレームレベル: shape ``(B, D_s)``。
                D_sはExtractorの出力次元数に依存する。

        Returns:
            予測されたリスナー動作テンソル。
            ウィンドウレベル: shape ``(B, T_out, D_out)``。
            フレームレベル: shape ``(B, D_out)``。

        Example:
            >>> # L2L: ウィンドウレベル入力
            >>> audio = torch.randn(2, 64, 128)    # (B, T, D_a)
            >>> motion = torch.randn(2, 64, 56)    # (B, T, D_s)
            >>> output = model.predict(audio, motion)  # (B, T_out, D_out)
        """

    @property
    @abstractmethod
    def requires_window(self) -> bool:
        """ウィンドウレベル入力が必要かどうかを返す。

        Returns:
            ``True``: ウィンドウレベル入力（複数フレーム蓄積）が必要。
                パイプラインは ``window_size`` フレーム分をバッファリングしてから
                ``predict()`` を呼び出す。
            ``False``: フレームレベル入力（1フレームずつ処理）。
        """

    @property
    @abstractmethod
    def window_size(self) -> Optional[int]:
        """ウィンドウレベル入力時のフレーム数を返す。

        Returns:
            ``requires_window`` が ``True`` の場合: ウィンドウのフレーム数
            （例: L2Lなら64）。
            ``requires_window`` が ``False`` の場合: ``None``。

        Example:
            >>> model.requires_window  # True
            >>> model.window_size      # 64
        """

    @property
    def device(self) -> torch.device:
        """モデルが配置されているデバイスを返す。

        サブクラスの ``__init__`` で設定された ``self._device`` を参照する。

        Returns:
            モデルのデバイス（例: ``torch.device("cuda:0")``）。

        Raises:
            AttributeError: サブクラスで ``_device`` が設定されていない場合。
        """
        return self._device

    def validate_inputs(
        self,
        audio_features: torch.Tensor,
        speaker_motion: torch.Tensor,
    ) -> None:
        """入力テンソルの次元数を検証する。

        ``audio_features`` と ``speaker_motion`` の ``ndim`` がそれぞれ
        2（フレームレベル）または3（ウィンドウレベル）であることを確認する。

        Args:
            audio_features: 検証対象の音声特徴量テンソル。
            speaker_motion: 検証対象の話者動作テンソル。

        Raises:
            ValueError: いずれかのテンソルの次元数が2でも3でもない場合。

        Example:
            >>> model.validate_inputs(
            ...     torch.randn(2, 64, 128),  # 3D: OK
            ...     torch.randn(2, 56),        # 2D: OK
            ... )
        """
        if audio_features.ndim not in (2, 3):
            raise ValueError(
                f"audio_featuresは2次元 (B, D) または3次元 (B, T, D) である"
                f"必要があります。受け取った次元数: {audio_features.ndim} "
                f"(shape: {audio_features.shape})"
            )

        if speaker_motion.ndim not in (2, 3):
            raise ValueError(
                f"speaker_motionは2次元 (B, D) または3次元 (B, T, D) である"
                f"必要があります。受け取った次元数: {speaker_motion.ndim} "
                f"(shape: {speaker_motion.shape})"
            )
