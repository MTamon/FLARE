"""BaseExtractor の ABC 制約・デフォルト実装テスト。

抽象メソッドの制約、extract_batch のデフォルトループ実装を検証する。
"""

from __future__ import annotations

import pytest
import torch

from flare.extractors.base import BaseExtractor


class _CompleteExtractor(BaseExtractor):
    """テスト用の完全実装Extractor。"""

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """ダミー抽出: 入力に依存しない固定値を返す。"""
        return {
            "exp": torch.ones(1, 50),
            "pose": torch.ones(1, 6),
        }

    @property
    def param_dim(self) -> int:
        """パラメータ総次元数。"""
        return 56

    @property
    def param_keys(self) -> list[str]:
        """パラメータキーリスト。"""
        return ["exp", "pose"]


class TestBaseExtractorABC:
    """BaseExtractor の ABC 制約テスト。"""

    def test_cannot_instantiate_abstract(self) -> None:
        """抽象メソッドを未実装のサブクラスはインスタンス化できないこと。"""
        with pytest.raises(TypeError):
            BaseExtractor()  # type: ignore[abstract]

    def test_missing_extract_raises(self) -> None:
        """extractのみ未実装でもインスタンス化できないこと。"""

        class _Incomplete(BaseExtractor):
            @property
            def param_dim(self) -> int:
                return 0

            @property
            def param_keys(self) -> list[str]:
                return []

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_missing_param_dim_raises(self) -> None:
        """param_dimのみ未実装でもインスタンス化できないこと。"""

        class _Incomplete(BaseExtractor):
            def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
                return {}

            @property
            def param_keys(self) -> list[str]:
                return []

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_complete_implementation(self) -> None:
        """全メソッド実装済みのサブクラスはインスタンス化できること。"""
        ext = _CompleteExtractor()
        assert ext.param_dim == 56
        assert ext.param_keys == ["exp", "pose"]


class TestExtractBatchDefault:
    """extract_batch のデフォルト実装テスト。"""

    def test_batch_calls_extract_per_sample(self) -> None:
        """extract_batchがバッチ内の各サンプルに対してextractを呼ぶこと。"""
        ext = _CompleteExtractor()
        images = torch.randn(4, 3, 224, 224)
        result = ext.extract_batch(images)

        assert "exp" in result
        assert "pose" in result
        assert result["exp"].shape == (4, 50)
        assert result["pose"].shape == (4, 6)

    def test_batch_single_sample(self) -> None:
        """バッチサイズ1の場合も正しく動作すること。"""
        ext = _CompleteExtractor()
        images = torch.randn(1, 3, 224, 224)
        result = ext.extract_batch(images)

        assert result["exp"].shape == (1, 50)

    def test_batch_values_concatenated(self) -> None:
        """各サンプルの結果がdim=0で正しく結合されること。"""
        ext = _CompleteExtractor()
        images = torch.randn(3, 3, 224, 224)
        result = ext.extract_batch(images)

        assert torch.allclose(result["exp"], torch.ones(3, 50))
