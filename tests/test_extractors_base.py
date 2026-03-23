"""flare.extractors.base.BaseExtractor のテスト。

テスト用ConcreteExtractorを定義し、ABCインターフェースの
extract / extract_batch / param_dim / param_keys / validate_image を検証する。
"""

from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from flare.extractors.base import BaseExtractor


class ConcreteExtractor(BaseExtractor):
    """テスト用のBaseExtractor具象実装。

    DECA風の最小パラメータ（exp: 50D + pose: 6D = 56D）を返す。

    Attributes:
        _device: モデルのデバイス。
    """

    def __init__(self, device: torch.device | None = None) -> None:
        """ConcreteExtractorを初期化する。

        Args:
            device: モデルのデバイス。Noneの場合はCPU。
        """
        self._device: torch.device = device or torch.device("cpu")

    def extract(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ダミーの3DMMパラメータを返す。

        Args:
            image: 入力画像テンソル。shape (1, 3, H, W)。

        Returns:
            exp(50D)とpose(6D)を含むDict。
        """
        batch_size: int = image.shape[0]
        return {
            "exp": torch.zeros(batch_size, 50, device=self._device),
            "pose": torch.zeros(batch_size, 6, device=self._device),
        }

    @property
    def param_dim(self) -> int:
        """出力パラメータの総次元数。

        Returns:
            56（exp: 50 + pose: 6）。
        """
        return 56

    @property
    def param_keys(self) -> List[str]:
        """出力Dictのキーリスト。

        Returns:
            ["exp", "pose"]。
        """
        return ["exp", "pose"]


class TestBaseExtractor:
    """BaseExtractor ABCのテストスイート。"""

    @pytest.fixture()
    def extractor(self) -> ConcreteExtractor:
        """ConcreteExtractorインスタンスを返すフィクスチャ。

        Returns:
            CPUデバイスのConcreteExtractor。
        """
        return ConcreteExtractor(device=torch.device("cpu"))

    def test_extract_single_frame(
        self, extractor: ConcreteExtractor, dummy_image_tensor: torch.Tensor
    ) -> None:
        """extract()が正しいDictを返すことを確認する。

        Args:
            extractor: テスト対象のExtractor。
            dummy_image_tensor: shape (1, 3, 224, 224) のテスト画像。
        """
        result: Dict[str, torch.Tensor] = extractor.extract(dummy_image_tensor)

        assert isinstance(result, dict)
        assert "exp" in result
        assert "pose" in result
        assert result["exp"].shape == (1, 50)
        assert result["pose"].shape == (1, 6)

    def test_extract_batch_default(
        self, extractor: ConcreteExtractor
    ) -> None:
        """extract_batch()のデフォルト実装がbatch次元でcatされたDictを返すことを確認する。

        バッチサイズ3で検証する。
        """
        batch: torch.Tensor = torch.rand(3, 3, 224, 224)
        result: Dict[str, torch.Tensor] = extractor.extract_batch(batch)

        assert isinstance(result, dict)
        assert "exp" in result
        assert "pose" in result
        assert result["exp"].shape == (3, 50)
        assert result["pose"].shape == (3, 6)

    def test_param_dim(self, extractor: ConcreteExtractor) -> None:
        """param_dimが56であることを確認する。

        Args:
            extractor: テスト対象のExtractor。
        """
        assert extractor.param_dim == 56

    def test_param_keys(self, extractor: ConcreteExtractor) -> None:
        """param_keysが["exp", "pose"]であることを確認する。

        Args:
            extractor: テスト対象のExtractor。
        """
        assert extractor.param_keys == ["exp", "pose"]

    def test_device_property(self, extractor: ConcreteExtractor) -> None:
        """deviceプロパティがCPUデバイスを返すことを確認する。

        Args:
            extractor: テスト対象のExtractor。
        """
        assert extractor.device == torch.device("cpu")

    def test_validate_image_valid(
        self, extractor: ConcreteExtractor, dummy_image_tensor: torch.Tensor
    ) -> None:
        """正しい形状のTensorでエラーが出ないことを確認する。

        Args:
            extractor: テスト対象のExtractor。
            dummy_image_tensor: shape (1, 3, 224, 224) のテスト画像。
        """
        extractor.validate_image(dummy_image_tensor)

    def test_validate_image_valid_batch(
        self, extractor: ConcreteExtractor, dummy_batch_tensor: torch.Tensor
    ) -> None:
        """バッチ画像テンソルでもエラーが出ないことを確認する。

        Args:
            extractor: テスト対象のExtractor。
            dummy_batch_tensor: shape (4, 3, 224, 224) のバッチ画像。
        """
        extractor.validate_image(dummy_batch_tensor)

    def test_validate_image_invalid_no_batch_dim(
        self, extractor: ConcreteExtractor
    ) -> None:
        """shape (3, 224, 224)（バッチ次元なし）でValueErrorが送出されることを確認する。

        Args:
            extractor: テスト対象のExtractor。
        """
        invalid: torch.Tensor = torch.rand(3, 224, 224)

        with pytest.raises(ValueError, match="4次元"):
            extractor.validate_image(invalid)

    def test_validate_image_invalid_channels(
        self, extractor: ConcreteExtractor
    ) -> None:
        """チャネル数が3でないTensorでValueErrorが送出されることを確認する。

        Args:
            extractor: テスト対象のExtractor。
        """
        invalid: torch.Tensor = torch.rand(1, 4, 224, 224)

        with pytest.raises(ValueError, match="チャネル数"):
            extractor.validate_image(invalid)

    def test_cannot_instantiate_abc(self) -> None:
        """BaseExtractorを直接インスタンス化できないことを確認する。"""
        with pytest.raises(TypeError, match="abstract"):
            BaseExtractor()  # type: ignore[abstract]
