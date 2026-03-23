"""flare.renderers.base.BaseRenderer のテスト。

テスト用ConcreteRendererを定義し、setup/render分離パターンの
動作を検証する。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest
import torch

from flare.renderers.base import BaseRenderer


class ConcreteRenderer(BaseRenderer):
    """テスト用のBaseRenderer具象実装。

    FlashAvatar風の最小レンダラー。setup()後にrender()が
    (1, 3, 512, 512) のゼロテンソルを返す。

    Attributes:
        _device: レンダラーのデバイス。
        _initialized: setup()完了フラグ。
    """

    def __init__(self, device: torch.device | None = None) -> None:
        """ConcreteRendererを初期化する。

        Args:
            device: レンダラーのデバイス。Noneの場合はCPU。
        """
        self._device: torch.device = device or torch.device("cpu")
        self._initialized: bool = False

    def setup(
        self,
        source_image: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> None:
        """セッション初期化を行う。

        Args:
            source_image: ソース画像（テスト用では未使用）。
            **kwargs: 追加パラメータ。
        """
        self._initialized = True

    def render(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """ダミーのレンダリング結果を返す。

        Args:
            params: レンダリングパラメータ。

        Returns:
            shape (1, 3, 512, 512) のゼロテンソル。

        Raises:
            RuntimeError: setup()未呼び出しの場合。
        """
        self.ensure_initialized()
        return torch.zeros(1, 3, 512, 512, device=self._device)

    @property
    def is_initialized(self) -> bool:
        """setup()完了済みかどうかを返す。

        Returns:
            初期化済みならTrue。
        """
        return self._initialized

    @property
    def required_keys(self) -> List[str]:
        """必須パラメータキーを返す。

        Returns:
            テスト用の必須キーリスト。
        """
        return ["expr", "jaw_pose"]


class TestBaseRenderer:
    """BaseRenderer ABCのテストスイート。"""

    @pytest.fixture()
    def renderer(self) -> ConcreteRenderer:
        """ConcreteRendererインスタンスを返すフィクスチャ。

        Returns:
            CPUデバイスのConcreteRenderer。
        """
        return ConcreteRenderer(device=torch.device("cpu"))

    def test_render_before_setup_raises(self, renderer: ConcreteRenderer) -> None:
        """setup()前にrender()を呼ぶとRuntimeErrorが送出されることを確認する。

        Args:
            renderer: テスト対象のRenderer。
        """
        with pytest.raises(RuntimeError, match="初期化"):
            renderer.render({"expr": torch.zeros(1, 100)})

    def test_render_after_setup(self, renderer: ConcreteRenderer) -> None:
        """setup()後にrender()が(1,3,512,512)のTensorを返すことを確認する。

        Args:
            renderer: テスト対象のRenderer。
        """
        renderer.setup()
        result: torch.Tensor = renderer.render({
            "expr": torch.zeros(1, 100),
            "jaw_pose": torch.zeros(1, 6),
        })

        assert result.shape == (1, 3, 512, 512)

    def test_is_initialized_false_before_setup(
        self, renderer: ConcreteRenderer
    ) -> None:
        """初期状態でis_initializedがFalseであることを確認する。

        Args:
            renderer: テスト対象のRenderer。
        """
        assert renderer.is_initialized is False

    def test_is_initialized_true_after_setup(
        self, renderer: ConcreteRenderer
    ) -> None:
        """setup()後にis_initializedがTrueであることを確認する。

        Args:
            renderer: テスト対象のRenderer。
        """
        renderer.setup()
        assert renderer.is_initialized is True

    def test_validate_params_missing_key(self, renderer: ConcreteRenderer) -> None:
        """必須キーが不足している場合にKeyErrorが送出されることを確認する。

        Args:
            renderer: テスト対象のRenderer。
        """
        with pytest.raises(KeyError, match="jaw_pose"):
            renderer.validate_params({"expr": torch.zeros(1, 100)})

    def test_cannot_instantiate_abc(self) -> None:
        """BaseRendererを直接インスタンス化できないことを確認する。"""
        with pytest.raises(TypeError, match="abstract"):
            BaseRenderer()  # type: ignore[abstract]
