"""BaseRenderer の ABC 制約・setup/render 分離テスト。

抽象メソッドの制約、is_initialized の状態遷移を検証する。
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch

from flare.renderers.base import BaseRenderer


class _CompleteRenderer(BaseRenderer):
    """テスト用の完全実装Renderer。"""

    def __init__(self) -> None:
        """初期化。未セットアップ状態。"""
        self._initialized = False

    def setup(
        self, source_image: Optional[torch.Tensor] = None, **kwargs: object
    ) -> None:
        """ダミーセットアップ。"""
        self._initialized = True

    def render(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """ダミーレンダリング。未初期化時はRuntimeErrorを送出。"""
        if not self._initialized:
            raise RuntimeError("Renderer not initialized")
        return torch.zeros(1, 3, 512, 512)

    @property
    def is_initialized(self) -> bool:
        """セットアップ状態。"""
        return self._initialized


class TestBaseRendererABC:
    """BaseRenderer の ABC 制約テスト。"""

    def test_cannot_instantiate_abstract(self) -> None:
        """抽象メソッドを未実装のサブクラスはインスタンス化できないこと。"""
        with pytest.raises(TypeError):
            BaseRenderer()  # type: ignore[abstract]

    def test_missing_setup_raises(self) -> None:
        """setupのみ未実装でもインスタンス化できないこと。"""

        class _Incomplete(BaseRenderer):
            def render(
                self, params: dict[str, torch.Tensor]
            ) -> torch.Tensor:
                return torch.zeros(1)

            @property
            def is_initialized(self) -> bool:
                return False

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_missing_is_initialized_raises(self) -> None:
        """is_initializedのみ未実装でもインスタンス化できないこと。"""

        class _Incomplete(BaseRenderer):
            def setup(
                self,
                source_image: Optional[torch.Tensor] = None,
                **kwargs: object,
            ) -> None:
                pass

            def render(
                self, params: dict[str, torch.Tensor]
            ) -> torch.Tensor:
                return torch.zeros(1)

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_complete_implementation(self) -> None:
        """全メソッド実装済みのサブクラスはインスタンス化できること。"""
        renderer = _CompleteRenderer()
        assert not renderer.is_initialized


class TestRendererSetupRenderSeparation:
    """setup/render 分離パターンのテスト。"""

    def test_is_initialized_false_before_setup(self) -> None:
        """setup()前はis_initializedがFalseであること。"""
        renderer = _CompleteRenderer()
        assert renderer.is_initialized is False

    def test_is_initialized_true_after_setup(self) -> None:
        """setup()後はis_initializedがTrueであること。"""
        renderer = _CompleteRenderer()
        renderer.setup(source_image=torch.randn(1, 3, 256, 256))
        assert renderer.is_initialized is True

    def test_render_before_setup_raises(self) -> None:
        """setup()前にrender()を呼ぶとRuntimeErrorが発生すること。"""
        renderer = _CompleteRenderer()
        with pytest.raises(RuntimeError, match="not initialized"):
            renderer.render({"expr": torch.zeros(1, 100)})

    def test_render_after_setup_succeeds(self) -> None:
        """setup()後のrender()が正常に動作すること。"""
        renderer = _CompleteRenderer()
        renderer.setup()
        output = renderer.render({"expr": torch.zeros(1, 100)})
        assert output.shape == (1, 3, 512, 512)
