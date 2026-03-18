"""BaseExtractor ABC のテスト (Section 8.2)"""

from __future__ import annotations

import pytest
import torch

from flare.extractors.base import BaseExtractor


class TestBaseExtractorABC:
    """ABC がインスタンス化できないことを確認。"""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseExtractor()  # type: ignore[abstract]


class TestDummyExtractor:
    """conftest.DummyExtractor を使ったスモークテスト。"""

    def test_extract_returns_dict(self, dummy_extractor):
        image = torch.randn(1, 3, 224, 224)
        result = dummy_extractor.extract(image)
        assert isinstance(result, dict)
        for key in dummy_extractor.param_keys:
            assert key in result
            assert isinstance(result[key], torch.Tensor)

    def test_extract_batch_shape(self, dummy_extractor):
        """extract_batch のデフォルトループ実装 (Section 2.3) が正しく連結する。"""
        B = 4
        images = torch.randn(B, 3, 224, 224)
        result = dummy_extractor.extract_batch(images)
        for key in dummy_extractor.param_keys:
            assert result[key].shape[0] == B

    def test_param_dim(self, dummy_extractor):
        assert dummy_extractor.param_dim == 284  # 100+50+6+128

    def test_param_keys(self, dummy_extractor):
        assert dummy_extractor.param_keys == ["shape", "exp", "pose", "detail"]

    def test_extract_batch_single(self, dummy_extractor):
        """バッチサイズ 1 でも動作する。"""
        images = torch.randn(1, 3, 224, 224)
        result = dummy_extractor.extract_batch(images)
        assert result["exp"].shape == (1, 50)