"""flare.model_interface.base.BaseLHGModel のテスト。

テスト用ConcreteLHGModelを定義し、2引数版predict()および
requires_window / window_size / validate_inputs を検証する。
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch

from flare.model_interface.base import BaseLHGModel


class ConcreteLHGModel(BaseLHGModel):
    """テスト用のBaseLHGModel具象実装。

    L2L風のウィンドウレベルモデル。predict()は (1, 64, 184) の
    ゼロテンソルを返す。

    Attributes:
        _device: モデルのデバイス。
    """

    def __init__(self, device: torch.device | None = None) -> None:
        """ConcreteLHGModelを初期化する。

        Args:
            device: モデルのデバイス。Noneの場合はCPU。
        """
        self._device: torch.device = device or torch.device("cpu")

    def predict(
        self,
        audio_features: torch.Tensor,
        speaker_motion: torch.Tensor,
    ) -> torch.Tensor:
        """ダミーのリスナー動作予測を返す。

        Args:
            audio_features: 音声特徴量テンソル。
            speaker_motion: 話者動作テンソル。

        Returns:
            shape (1, 64, 184) のゼロテンソル。
        """
        self.validate_inputs(audio_features, speaker_motion)
        return torch.zeros(1, 64, 184, device=self._device)

    @property
    def requires_window(self) -> bool:
        """ウィンドウレベル入力が必要。

        Returns:
            True。
        """
        return True

    @property
    def window_size(self) -> Optional[int]:
        """ウィンドウサイズ64フレーム。

        Returns:
            64。
        """
        return 64


class TestBaseLHGModel:
    """BaseLHGModel ABCのテストスイート。"""

    @pytest.fixture()
    def model(self) -> ConcreteLHGModel:
        """ConcreteLHGModelインスタンスを返すフィクスチャ。

        Returns:
            CPUデバイスのConcreteLHGModel。
        """
        return ConcreteLHGModel(device=torch.device("cpu"))

    def test_predict_two_args(self, model: ConcreteLHGModel) -> None:
        """predict(audio_features, speaker_motion)が正しい形状を返すことを確認する。

        Args:
            model: テスト対象のLHGモデル。
        """
        audio: torch.Tensor = torch.rand(1, 64, 128)
        motion: torch.Tensor = torch.rand(1, 64, 56)
        result: torch.Tensor = model.predict(audio, motion)

        assert result.shape == (1, 64, 184)

    def test_requires_window(self, model: ConcreteLHGModel) -> None:
        """requires_windowがTrueであることを確認する。

        Args:
            model: テスト対象のLHGモデル。
        """
        assert model.requires_window is True

    def test_window_size(self, model: ConcreteLHGModel) -> None:
        """window_sizeが64であることを確認する。

        Args:
            model: テスト対象のLHGモデル。
        """
        assert model.window_size == 64

    def test_validate_inputs_valid_3d(self, model: ConcreteLHGModel) -> None:
        """shape (1, 64, 128) の3次元入力でエラーが出ないことを確認する。

        Args:
            model: テスト対象のLHGモデル。
        """
        audio: torch.Tensor = torch.rand(1, 64, 128)
        motion: torch.Tensor = torch.rand(1, 64, 56)
        model.validate_inputs(audio, motion)

    def test_validate_inputs_valid_2d(self, model: ConcreteLHGModel) -> None:
        """shape (1, 128) の2次元入力でエラーが出ないことを確認する。

        Args:
            model: テスト対象のLHGモデル。
        """
        audio: torch.Tensor = torch.rand(1, 128)
        motion: torch.Tensor = torch.rand(1, 56)
        model.validate_inputs(audio, motion)

    def test_validate_inputs_invalid_ndim(self, model: ConcreteLHGModel) -> None:
        """shape (128,) のTensor（1次元）でValueErrorが送出されることを確認する。

        Args:
            model: テスト対象のLHGモデル。
        """
        invalid: torch.Tensor = torch.rand(128)
        valid: torch.Tensor = torch.rand(1, 64, 56)

        with pytest.raises(ValueError, match="audio_features"):
            model.validate_inputs(invalid, valid)

    def test_validate_inputs_invalid_motion_ndim(
        self, model: ConcreteLHGModel
    ) -> None:
        """speaker_motionが1次元の場合にValueErrorが送出されることを確認する。

        Args:
            model: テスト対象のLHGモデル。
        """
        valid: torch.Tensor = torch.rand(1, 64, 128)
        invalid: torch.Tensor = torch.rand(56)

        with pytest.raises(ValueError, match="speaker_motion"):
            model.validate_inputs(valid, invalid)

    def test_cannot_instantiate_abc(self) -> None:
        """BaseLHGModelを直接インスタンス化できないことを確認する。"""
        with pytest.raises(TypeError, match="abstract"):
            BaseLHGModel()  # type: ignore[abstract]
