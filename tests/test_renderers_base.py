"""BaseRenderer ABC のテスト (Section 8.2: setup/render 分離)"""

from __future__ import annotations

import pytest
import torch

from flare.renderers.base import BaseRenderer


class TestBaseRendererABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseRenderer()  # type: ignore[abstract]


class TestDummyRenderer:
    def test_not_initialized_before_setup(self, dummy_renderer):
        assert not dummy_renderer.is_initialized

    def test_initialized_after_setup(self, dummy_renderer):
        dummy_renderer.setup(source_image=torch.randn(1, 3, 512, 512))
        assert dummy_renderer.is_initialized

    def test_setup_none_source(self, dummy_renderer):
        """FlashAvatar 風: source_image=None でも setup() が通る。"""
        dummy_renderer.setup(source_image=None)
        assert dummy_renderer.is_initialized

    def test_render_returns_tensor(self, dummy_renderer):
        dummy_renderer.setup()
        params = {"expr": torch.randn(1, 100)}
        output = dummy_renderer.render(params)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 3, 512, 512)

    def test_render_before_setup_raises(self, dummy_renderer):
        """setup() 前の render() 呼び出しはエラーになるべき。"""
        with pytest.raises(RuntimeError):
            dummy_renderer.render({"expr": torch.randn(1, 100)})