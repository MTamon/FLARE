"""BaseLHGModel の ABC 制約・シグネチャテスト。

抽象メソッドの制約、predict の2引数シグネチャを検証する。
"""

from __future__ import annotations

import inspect
from typing import Optional

import pytest
import torch

from flare.model_interface.base import BaseLHGModel


class _CompleteModel(BaseLHGModel):
    """テスト用の完全実装LHGModel。"""

    def predict(
        self,
        audio_features: torch.Tensor,
        speaker_motion: torch.Tensor,
    ) -> torch.Tensor:
        """ダミー予測: 入力と同形状のゼロテンソルを返す。"""
        return torch.zeros_like(speaker_motion)

    @property
    def requires_window(self) -> bool:
        """ウィンドウレベル入力が必要。"""
        return True

    @property
    def window_size(self) -> Optional[int]:
        """ウィンドウサイズ。"""
        return 64


class TestBaseLHGModelABC:
    """BaseLHGModel の ABC 制約テスト。"""

    def test_cannot_instantiate_abstract(self) -> None:
        """抽象メソッドを未実装のサブクラスはインスタンス化できないこと。"""
        with pytest.raises(TypeError):
            BaseLHGModel()  # type: ignore[abstract]

    def test_missing_predict_raises(self) -> None:
        """predictのみ未実装でもインスタンス化できないこと。"""

        class _Incomplete(BaseLHGModel):
            @property
            def requires_window(self) -> bool:
                return False

            @property
            def window_size(self) -> Optional[int]:
                return None

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_missing_requires_window_raises(self) -> None:
        """requires_windowのみ未実装でもインスタンス化できないこと。"""

        class _Incomplete(BaseLHGModel):
            def predict(
                self,
                audio_features: torch.Tensor,
                speaker_motion: torch.Tensor,
            ) -> torch.Tensor:
                return torch.zeros(1)

            @property
            def window_size(self) -> Optional[int]:
                return None

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_complete_implementation(self) -> None:
        """全メソッド実装済みのサブクラスはインスタンス化できること。"""
        model = _CompleteModel()
        assert model.requires_window is True
        assert model.window_size == 64


class TestPredictSignature:
    """predict の2引数シグネチャ確認テスト。"""

    def test_predict_has_two_positional_params(self) -> None:
        """predictがaudio_featuresとspeaker_motionの2引数を持つこと。"""
        sig = inspect.signature(BaseLHGModel.predict)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "audio_features" in params
        assert "speaker_motion" in params
        non_self = [p for p in params if p != "self"]
        assert len(non_self) == 2

    def test_predict_accepts_two_tensors(self) -> None:
        """predictが2つのテンソルを受け取って結果を返すこと。"""
        model = _CompleteModel()
        audio = torch.randn(2, 64, 128)
        motion = torch.randn(2, 64, 56)
        output = model.predict(audio, motion)
        assert output.shape == motion.shape

    def test_predict_window_level(self) -> None:
        """ウィンドウレベル入力(B, T, D)が正しく処理されること。"""
        model = _CompleteModel()
        audio = torch.randn(1, 64, 128)
        motion = torch.randn(1, 64, 56)
        output = model.predict(audio, motion)
        assert output.ndim == 3
        assert output.shape[1] == 64

    def test_predict_frame_level(self) -> None:
        """フレームレベル入力(B, D)も受け付けること。"""
        model = _CompleteModel()
        audio = torch.randn(1, 128)
        motion = torch.randn(1, 56)
        output = model.predict(audio, motion)
        assert output.shape == (1, 56)
