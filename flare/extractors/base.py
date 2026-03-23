"""BaseExtractor: 3DMMパラメータ抽出の抽象基底クラス。

仕様書8.2節に基づき、全Extractorが実装すべきインターフェースを定義する。
入出力は全て ``torch.Tensor`` で統一し、GPU上での処理完結を前提とする。

ルートA（BFM）の場合:
    - Deep3DFaceRecon: id(80D) + exp(64D) + tex(80D) + pose(6D) + lighting(27D) = 257D

ルートB（FLAME）の場合:
    - DECA: shape(100D) + tex(50D) + exp(50D) + pose(6D) + cam(3D) + light(27D) + detail(128D) = 364D
    - SMIRK: exp(50D) + pose(6D) 等

Example:
    >>> class DECAExtractor(BaseExtractor):
    ...     # 抽象メソッド・プロパティを全て実装
    ...     ...
    >>> extractor = DECAExtractor(device=torch.device("cuda:0"))
    >>> result = extractor.extract(image_tensor)  # Dict[str, Tensor]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import torch


class BaseExtractor(ABC):
    """3DMMパラメータ抽出の抽象基底クラス。

    全てのExtractor実装（DECA, Deep3DFaceRecon, SMIRK等）はこのクラスを継承し、
    ``extract()``・``param_dim``・``param_keys`` を実装する。

    ``extract_batch()`` はデフォルトでループ版を提供し、サブクラスで
    バッチ最適化のためにオーバーライドできる。

    Attributes:
        _device: モデルが配置されるCUDAデバイス。サブクラスの ``__init__`` で設定する。

    Example:
        >>> class MyExtractor(BaseExtractor):
        ...     def __init__(self, device: torch.device) -> None:
        ...         self._device = device
        ...
        ...     def extract(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        ...         return {"exp": torch.zeros(1, 50, device=self._device)}
        ...
        ...     @property
        ...     def param_dim(self) -> int:
        ...         return 50
        ...
        ...     @property
        ...     def param_keys(self) -> List[str]:
        ...         return ["exp"]
    """

    _device: torch.device

    @abstractmethod
    def extract(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """1フレームから3DMMパラメータを抽出する。

        入力画像は ``face_detect.py`` によって顔検出・クロッピング済みであることを
        前提とする。Extractorは顔検出を行わない。

        Args:
            image: 前処理済み顔画像テンソル。shape ``(1, 3, H, W)``。
                通常 H=W=224。デバイスはGPU上を想定。

        Returns:
            パラメータ名をキー、対応するテンソルを値とするDict。
            各テンソルのbatch次元は入力に合わせて ``(1, D)`` となる。
            具体的なキーはExtractor実装に依存する（``param_keys`` で取得可能）。

        Raises:
            RuntimeError: モデル推論に失敗した場合。

        Example:
            >>> result = extractor.extract(image)
            >>> result["exp"].shape  # DECA: (1, 50)
        """

    def extract_batch(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """バッチ画像から3DMMパラメータを一括抽出する。

        デフォルト実装は ``extract()`` をフレームごとにループで呼び出し、
        結果をbatch次元で結合して返す。サブクラスでバッチ処理に最適化された
        実装にオーバーライドできる。

        Args:
            images: バッチ画像テンソル。shape ``(B, 3, H, W)``。
                Bはバッチサイズ。通常 H=W=224。

        Returns:
            パラメータ名をキー、batch次元で結合されたテンソルを値とするDict。
            各テンソルのshapeは ``(B, D)`` となる。

        Example:
            >>> batch = torch.randn(8, 3, 224, 224, device=device)
            >>> results = extractor.extract_batch(batch)
            >>> results["exp"].shape  # DECA: (8, 50)
        """
        self.validate_image(images)

        accumulator: Dict[str, List[torch.Tensor]] = {}

        for i in range(images.shape[0]):
            single_image: torch.Tensor = images[i : i + 1]  # (1, 3, H, W)
            single_result: Dict[str, torch.Tensor] = self.extract(single_image)

            for key, value in single_result.items():
                accumulator.setdefault(key, []).append(value)

        batched: Dict[str, torch.Tensor] = {
            key: torch.cat(tensors, dim=0) for key, tensors in accumulator.items()
        }
        return batched

    @property
    @abstractmethod
    def param_dim(self) -> int:
        """出力パラメータの総次元数を返す。

        Extractorが出力するDictの全パラメータの次元数の合計を返す。
        例えばDECAの場合、shape(100) + tex(50) + exp(50) + pose(6) +
        cam(3) + light(27) + detail(128) = 364。

        Returns:
            パラメータの総次元数。
        """

    @property
    @abstractmethod
    def param_keys(self) -> List[str]:
        """出力Dictのキーリストを返す。

        ``extract()`` が返すDictに含まれるキーの一覧を返す。
        順序はパラメータの連結順序に対応する。

        Returns:
            パラメータキー名のリスト。

        Example:
            >>> extractor.param_keys
            ['shape', 'tex', 'exp', 'pose', 'cam', 'light', 'detail']
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

    def validate_image(self, image: torch.Tensor) -> None:
        """入力画像テンソルの形状を検証する。

        shape が ``(B, 3, H, W)`` の4次元テンソルであり、
        チャネル数が3であることを確認する。

        Args:
            image: 検証対象の画像テンソル。

        Raises:
            ValueError: テンソルが4次元でない場合、
                またはチャネル数が3でない場合。

        Example:
            >>> extractor.validate_image(torch.randn(1, 3, 224, 224))  # OK
            >>> extractor.validate_image(torch.randn(3, 224, 224))     # ValueError
        """
        if image.ndim != 4:
            raise ValueError(
                f"画像テンソルは4次元 (B, 3, H, W) である必要があります。"
                f"受け取った次元数: {image.ndim} (shape: {image.shape})"
            )

        channels: int = image.shape[1]
        if channels != 3:
            raise ValueError(
                f"画像テンソルのチャネル数は3である必要があります。"
                f"受け取ったチャネル数: {channels} (shape: {image.shape})"
            )
