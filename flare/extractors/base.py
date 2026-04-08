"""特徴量抽出の抽象基底クラス。

画像から3DMMパラメータを抽出するExtractorの共通インターフェースを定義する。
全てのExtractor実装（DECA, Deep3DFaceRecon, SMIRK等）はBaseExtractorを継承し、
抽象メソッドを実装しなければならない。

v2.0変更点:
    - 入出力の型を ``np.ndarray`` から ``torch.Tensor`` に統一（GPU処理完結のため）。
    - ``extract_batch`` を ``@abstractmethod`` からデフォルト実装を持つconcreteメソッドに変更。
      サブクラスでバッチ最適化のオーバーライドが可能。

Example:
    BaseExtractorを継承した具体クラスの実装::

        class DECAExtractor(BaseExtractor):
            def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
                ...

            @property
            def param_dim(self) -> int:
                return 236

            @property
            def param_keys(self) -> list[str]:
                return ["shape", "tex", "exp", "pose", "cam", "light", "detail"]
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseExtractor(ABC):
    """3DMMパラメータ抽出の抽象基底クラス。

    画像テンソルを受け取り、3DMMパラメータをキー付き辞書として返す。
    Route A（BFM系: Deep3DFaceRecon）およびRoute B（FLAME系: DECA, SMIRK等）の
    両方のExtractorがこのインターフェースを実装する。

    入力画像はface_detect.pyによる顔検出・クロッピング済みであることを前提とする。
    Extractorは顔検出を行わない。

    Attributes:
        param_dim: 出力パラメータの総次元数。
        param_keys: 出力辞書のキーリスト。
    """

    @abstractmethod
    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """1フレームの画像から3DMMパラメータを抽出する。

        Args:
            image: 入力画像テンソル。形状は ``(1, 3, H, W)`` または ``(3, H, W)``。
                face_detect.pyでクロッピング済みの顔画像。値域は ``[0, 1]``。

        Returns:
            キーがパラメータ名、値がテンソルの辞書。
            各テンソルの形状は ``(1, D)`` （Dはパラメータ固有の次元数）。
            例（DECA）::

                {
                    "shape": Tensor(1, 100),
                    "tex":   Tensor(1, 50),
                    "exp":   Tensor(1, 50),
                    "pose":  Tensor(1, 6),
                    "cam":   Tensor(1, 3),
                    "light": Tensor(1, 27),
                    "detail": Tensor(1, 128),
                }

        Raises:
            RuntimeError: モデル推論に失敗した場合。
        """

    def extract_batch(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """バッチ画像から3DMMパラメータを一括抽出する。

        デフォルト実装ではextract()をループで呼び出す。
        サブクラスでバッチ最適化されたオーバーライドが可能。

        Args:
            images: バッチ画像テンソル。形状は ``(B, 3, H, W)``。
                Bはバッチサイズ。各画像はface_detect.pyでクロッピング済み。

        Returns:
            キーがパラメータ名、値がバッチ結合済みテンソルの辞書。
            各テンソルの形状は ``(B, D)`` （Dはパラメータ固有の次元数）。

        Raises:
            RuntimeError: モデル推論に失敗した場合。
        """
        results: dict[str, list[torch.Tensor]] = {}
        for i in range(images.shape[0]):
            single = self.extract(images[i : i + 1])
            for k, v in single.items():
                results.setdefault(k, []).append(v)
        return {k: torch.cat(v, dim=0) for k, v in results.items()}

    @property
    @abstractmethod
    def param_dim(self) -> int:
        """出力パラメータの総次元数。

        Returns:
            全パラメータキーの次元数を合算した整数値。
            例（DECA）: 100 + 50 + 50 + 6 + 3 + 27 + 128 = 364。
        """

    @property
    @abstractmethod
    def param_keys(self) -> list[str]:
        """出力辞書のキーリスト。

        Returns:
            extract()が返す辞書のキー名リスト。
            例（DECA）: ``["shape", "tex", "exp", "pose", "cam", "light", "detail"]``。
        """
