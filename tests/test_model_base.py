"""BaseLHGModel ABC のテスト (Section 8.2: 2 引数版 predict)"""

from __future__ import annotations

import pytest
import torch

from flare.model_interface.base import BaseLHGModel


class TestBaseLHGModelABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseLHGModel()  # type: ignore[abstract]


class TestDummyLHGModel:
    def test_predict_returns_tensor(self, dummy_lhg_model):
        audio = torch.randn(2, 64, 128)   # (B, T_a, D_a)
        motion = torch.randn(2, 64, 56)   # (B, T_s, D_s)
        output = dummy_lhg_model.predict(audio, motion)
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2  # batch dim preserved

    def test_requires_window(self, dummy_lhg_model):
        assert dummy_lhg_model.requires_window is True

    def test_window_size(self, dummy_lhg_model):
        assert dummy_lhg_model.window_size == 64

    def test_frame_level_model(self):
        """requires_window=False のモデルでは window_size=None。"""

        class FrameLevelModel(BaseLHGModel):
            def predict(self, audio_features, speaker_motion):
                return torch.zeros(audio_features.shape[0], 56)

            @property
            def requires_window(self):
                return False

            @property
            def window_size(self):
                return None

        model = FrameLevelModel()
        assert not model.requires_window
        assert model.window_size is None
        out = model.predict(torch.randn(1, 128), torch.randn(1, 56))
        assert out.shape == (1, 56)