"""pytest共通フィクスチャ定義モジュール。

全テストファイルで共有されるフィクスチャを提供する。
ダミーテンソル、DECA出力辞書、最小設定インスタンス等。

Example:
    テスト関数でのフィクスチャ使用::

        def test_something(dummy_tensor, minimal_config):
            assert dummy_tensor.shape == (1, 3, 224, 224)
            assert minimal_config.pipeline.fps == 30
"""

from __future__ import annotations

import pytest
import torch

from flare.config import PipelineConfig


@pytest.fixture()
def dummy_tensor() -> torch.Tensor:
    """テスト用のダミー画像テンソルを返す。

    Returns:
        形状 ``(1, 3, 224, 224)`` のランダムテンソル。値域は ``[0, 1)``。
    """
    return torch.rand(1, 3, 224, 224)


@pytest.fixture()
def dummy_deca_output() -> dict[str, torch.Tensor]:
    """テスト用のDECA出力辞書を返す。

    仕様書§8.2のDECAキー定義に基づくダミー出力。
    バッチサイズ2で生成。

    Returns:
        以下のキーを持つ辞書::

            {
                "shape": (2, 100), "tex": (2, 50), "exp": (2, 50),
                "pose": (2, 6), "cam": (2, 3), "light": (2, 27),
                "detail": (2, 128),
            }
    """
    batch_size = 2
    return {
        "shape": torch.randn(batch_size, 100),
        "tex": torch.randn(batch_size, 50),
        "exp": torch.randn(batch_size, 50),
        "pose": torch.randn(batch_size, 6),
        "cam": torch.randn(batch_size, 3),
        "light": torch.randn(batch_size, 27),
        "detail": torch.randn(batch_size, 128),
    }


@pytest.fixture()
def minimal_config() -> PipelineConfig:
    """テスト用の最小PipelineConfigインスタンスを返す。

    Returns:
        全フィールドがデフォルト値のPipelineConfig。
    """
    return PipelineConfig()
