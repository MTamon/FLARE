"""Phase 2 の統合テスト。

DECAToFlameAdapter / FlashAvatarRenderer / IdentityAdapter / L2LModel の
インターフェースと動作を検証する。外部モデル依存のクラスはモックを使用する。
"""

from __future__ import annotations

import inspect
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

from flare.converters.deca_to_flame import (
    DECAToFlameAdapter,
    _aa_to_mat,
    _mat_to_6d,
)
from flare.converters.identity import IdentityAdapter
from flare.converters.registry import AdapterRegistry
from flare.renderers.flashavatar import (
    FlashAvatarRenderer,
    _CONDITION_DIM,
    _CONDITION_KEYS,
)
from flare.utils.errors import ModelLoadError, RendererNotInitializedError


# =========================================================================
# DECAToFlameAdapter
# =========================================================================


class TestDECAToFlameAdapterConversion:
    """DECAToFlameAdapter の完全な変換テスト。"""

    def test_expr_shape_and_padding(self) -> None:
        """exp 50D → expr 100D: 形状が (B, 100) であること。"""
        adapter = DECAToFlameAdapter()
        source = {"exp": torch.randn(4, 50), "pose": torch.randn(4, 6)}
        result = adapter.convert(source)
        assert result["expr"].shape == (4, 100)

    def test_expr_first_50_match_input(self) -> None:
        """expr の先頭50Dが入力 exp と一致すること。"""
        adapter = DECAToFlameAdapter()
        exp = torch.randn(2, 50)
        source = {"exp": exp, "pose": torch.randn(2, 6)}
        result = adapter.convert(source)
        assert torch.allclose(result["expr"][:, :50], exp)

    def test_expr_last_50_are_zero(self) -> None:
        """expr の後半50Dがゼロパディングであること。"""
        adapter = DECAToFlameAdapter()
        source = {"exp": torch.randn(3, 50), "pose": torch.randn(3, 6)}
        result = adapter.convert(source)
        assert torch.allclose(result["expr"][:, 50:], torch.zeros(3, 50))

    def test_jaw_pose_shape(self) -> None:
        """jaw_pose が (B, 6) の rotation_6d であること。"""
        adapter = DECAToFlameAdapter()
        source = {"exp": torch.randn(2, 50), "pose": torch.randn(2, 6)}
        result = adapter.convert(source)
        assert result["jaw_pose"].shape == (2, 6)

    def test_jaw_pose_identity_for_zero_input(self) -> None:
        """axis-angle がゼロのとき jaw_pose が単位回転の6D表現であること。"""
        adapter = DECAToFlameAdapter()
        source = {"exp": torch.zeros(1, 50), "pose": torch.zeros(1, 6)}
        result = adapter.convert(source)
        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        assert torch.allclose(result["jaw_pose"], identity_6d, atol=1e-6)

    def test_jaw_pose_rotation_6d_valid_range(self) -> None:
        """jaw_pose の各成分が合理的な範囲内であること。"""
        adapter = DECAToFlameAdapter()
        source = {"exp": torch.randn(8, 50), "pose": torch.randn(8, 6) * 0.5}
        result = adapter.convert(source)
        assert result["jaw_pose"].shape == (8, 6)
        assert torch.all(result["jaw_pose"].abs() <= 2.0)

    def test_eyes_pose_shape(self) -> None:
        """eyes_pose が (B, 12) であること。"""
        adapter = DECAToFlameAdapter()
        source = {"exp": torch.randn(5, 50), "pose": torch.randn(5, 6)}
        result = adapter.convert(source)
        assert result["eyes_pose"].shape == (5, 12)

    def test_eyes_pose_is_identity_repeated(self) -> None:
        """eyes_pose が identity_6d を2回repeatした値であること。"""
        adapter = DECAToFlameAdapter()
        source = {"exp": torch.zeros(1, 50), "pose": torch.zeros(1, 6)}
        result = adapter.convert(source)
        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        expected = identity_6d.repeat(1, 2)
        assert torch.allclose(result["eyes_pose"], expected, atol=1e-6)

    def test_eyelids_shape(self) -> None:
        """eyelids が (B, 2) であること。"""
        adapter = DECAToFlameAdapter()
        source = {"exp": torch.randn(3, 50), "pose": torch.randn(3, 6)}
        result = adapter.convert(source)
        assert result["eyelids"].shape == (3, 2)

    def test_eyelids_all_zero(self) -> None:
        """eyelids が全てゼロであること。"""
        adapter = DECAToFlameAdapter()
        source = {"exp": torch.randn(4, 50), "pose": torch.randn(4, 6)}
        result = adapter.convert(source)
        assert torch.allclose(result["eyelids"], torch.zeros(4, 2))

    def test_condition_vector_total_dim(self) -> None:
        """出力の総次元数が120D (100+6+12+2) であること。"""
        adapter = DECAToFlameAdapter()
        source = {"exp": torch.randn(2, 50), "pose": torch.randn(2, 6)}
        result = adapter.convert(source)
        total = sum(v.shape[-1] for v in result.values())
        assert total == _CONDITION_DIM

    def test_missing_exp_key_raises(self) -> None:
        """expキーが欠損している場合にKeyErrorが発生すること。"""
        adapter = DECAToFlameAdapter()
        with pytest.raises(KeyError):
            adapter.convert({"pose": torch.zeros(1, 6)})

    def test_missing_pose_key_raises(self) -> None:
        """poseキーが欠損している場合にKeyErrorが発生すること。"""
        adapter = DECAToFlameAdapter()
        with pytest.raises(KeyError):
            adapter.convert({"exp": torch.zeros(1, 50)})


class TestDECAToFlameRotationUtils:
    """回転変換ユーティリティのテスト。"""

    def test_aa_to_mat_identity(self) -> None:
        """ゼロ入力で単位行列が返ること。"""
        aa = torch.zeros(1, 3)
        mat = _aa_to_mat(aa)
        assert mat.shape == (1, 3, 3)
        assert torch.allclose(mat, torch.eye(3).unsqueeze(0), atol=1e-6)

    def test_mat_to_6d_identity(self) -> None:
        """単位行列の6D表現が [1,0,0,0,1,0] であること。"""
        mat = torch.eye(3).unsqueeze(0)
        r6d = _mat_to_6d(mat)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        assert torch.allclose(r6d, expected, atol=1e-6)

    def test_roundtrip_preserves_rotation(self) -> None:
        """axis-angle → 回転行列の変換が正しい回転を表すこと。"""
        aa = torch.tensor([[0.0, 0.0, 3.14159 / 2]])
        mat = _aa_to_mat(aa)
        assert mat.shape == (1, 3, 3)
        det = torch.det(mat)
        assert torch.allclose(det, torch.ones(1), atol=1e-4)


# =========================================================================
# FlashAvatarRenderer
# =========================================================================


class TestFlashAvatarRendererInit:
    """FlashAvatarRenderer の初期化テスト。"""

    def test_not_initialized_on_creation(self) -> None:
        """生成直後はis_initializedがFalseであること。"""
        renderer = FlashAvatarRenderer()
        assert renderer.is_initialized is False

    def test_default_output_size(self) -> None:
        """デフォルト出力サイズが512x512であること。"""
        renderer = FlashAvatarRenderer()
        assert renderer._output_size == [512, 512]

    def test_custom_output_size(self) -> None:
        """カスタム出力サイズが反映されること。"""
        renderer = FlashAvatarRenderer(output_size=[1024, 768])
        assert renderer._output_size == [1024, 768]


class TestFlashAvatarRendererSetupRender:
    """FlashAvatarRenderer の setup/render 状態遷移テスト。"""

    def test_render_before_setup_raises(self) -> None:
        """setup前のrenderでRendererNotInitializedErrorが発生すること。"""
        renderer = FlashAvatarRenderer()
        params = {
            "expr": torch.zeros(1, 100),
            "jaw_pose": torch.zeros(1, 6),
            "eyes_pose": torch.zeros(1, 12),
            "eyelids": torch.zeros(1, 2),
        }
        with pytest.raises(RendererNotInitializedError):
            renderer.render(params)

    def test_setup_failure_raises_model_load_error(self) -> None:
        """モジュールインポート失敗でModelLoadErrorが発生すること。"""
        renderer = FlashAvatarRenderer(model_path="./nonexistent/")
        with pytest.raises(ModelLoadError):
            renderer.setup()

    def test_render_with_mock_model(self) -> None:
        """モック化されたモデルでrenderが正しい形状を返すこと。"""
        renderer = FlashAvatarRenderer()
        renderer._initialized = True
        renderer._device = torch.device("cpu")

        fake_render_output = {"render": torch.rand(3, 512, 512)}
        renderer._renderer_internal = MagicMock(return_value=fake_render_output)
        renderer._model = MagicMock()
        renderer._output_size = [512, 512]

        params = {
            "expr": torch.randn(1, 100),
            "jaw_pose": torch.randn(1, 6),
            "eyes_pose": torch.randn(1, 12),
            "eyelids": torch.randn(1, 2),
        }
        output = renderer.render(params)
        assert output.shape == (1, 3, 512, 512)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_render_missing_key_raises(self) -> None:
        """必要なキーが欠損している場合にKeyErrorが発生すること。"""
        renderer = FlashAvatarRenderer()
        renderer._initialized = True
        with pytest.raises(KeyError):
            renderer.render({"expr": torch.zeros(1, 100)})

    def test_condition_vector_keys(self) -> None:
        """condition vectorに必要なキーが4つであること。"""
        assert len(_CONDITION_KEYS) == 4
        assert "expr" in _CONDITION_KEYS
        assert "jaw_pose" in _CONDITION_KEYS
        assert "eyes_pose" in _CONDITION_KEYS
        assert "eyelids" in _CONDITION_KEYS

    def test_condition_dim_is_120(self) -> None:
        """condition vectorの総次元数が120であること。"""
        assert _CONDITION_DIM == 120


# =========================================================================
# IdentityAdapter
# =========================================================================


class TestIdentityAdapter:
    """IdentityAdapter のパススルーテスト。"""

    def test_convert_returns_same_dict(self) -> None:
        """convert()が入力辞書と同一のオブジェクトを返すこと。"""
        adapter = IdentityAdapter()
        data = {"a": torch.randn(2, 3), "b": torch.randn(2, 5)}
        result = adapter.convert(data)
        assert result is data

    def test_convert_preserves_all_keys(self) -> None:
        """convert()が全キーを保持すること。"""
        adapter = IdentityAdapter()
        data = {
            "expr": torch.randn(1, 100),
            "jaw_pose": torch.randn(1, 6),
            "eyes_pose": torch.randn(1, 12),
        }
        result = adapter.convert(data)
        assert set(result.keys()) == {"expr", "jaw_pose", "eyes_pose"}

    def test_convert_preserves_tensor_values(self) -> None:
        """convert()がテンソル値を変更しないこと。"""
        adapter = IdentityAdapter()
        t = torch.randn(3, 50)
        data = {"param": t}
        result = adapter.convert(data)
        assert torch.equal(result["param"], t)

    def test_source_format(self) -> None:
        """source_formatが'any'であること。"""
        adapter = IdentityAdapter()
        assert adapter.source_format == "any"

    def test_target_format(self) -> None:
        """target_formatが'any'であること。"""
        adapter = IdentityAdapter()
        assert adapter.target_format == "any"

    def test_convert_empty_dict(self) -> None:
        """空辞書でもエラーなく動作すること。"""
        adapter = IdentityAdapter()
        result = adapter.convert({})
        assert result == {}

    def test_registry_integration(self) -> None:
        """AdapterRegistryに登録・取得できること。"""
        registry = AdapterRegistry()
        adapter = IdentityAdapter()
        registry.register(adapter)
        got = registry.get("any", "any")
        assert got is adapter


# =========================================================================
# L2LModel
# =========================================================================


class TestL2LModelInterface:
    """L2LModel のインターフェーステスト。"""

    def test_import_succeeds(self) -> None:
        """L2LModelがインポートできること。"""
        from flare.model_interface.l2l import L2LModel

        assert L2LModel is not None

    def test_requires_window_is_true(self) -> None:
        """requires_windowがTrueであること。"""
        from flare.model_interface.l2l import L2LModel

        with pytest.raises(ModelLoadError):
            L2LModel(model_path="./nonexistent.pth", device="cpu")

    def test_predict_signature(self) -> None:
        """predictのシグネチャがaudio_features, speaker_motionの2引数であること。"""
        from flare.model_interface.l2l import L2LModel

        sig = inspect.signature(L2LModel.predict)
        params = [p for p in sig.parameters.keys() if p != "self"]
        assert params == ["audio_features", "speaker_motion"]

    def test_window_size_default(self) -> None:
        """window_sizeのデフォルトが64であること。"""
        from flare.model_interface.l2l import L2LModel

        with patch.object(L2LModel, "_load_model", return_value=None):
            model = object.__new__(L2LModel)
            model._window_size = 64
            assert model.window_size == 64

    def test_requires_window_property(self) -> None:
        """requires_windowプロパティがTrueを返すこと。"""
        from flare.model_interface.l2l import L2LModel

        with patch.object(L2LModel, "_load_model", return_value=None):
            model = object.__new__(L2LModel)
            model._window_size = 64
            assert model.requires_window is True

    def test_predict_with_mock(self) -> None:
        """モック化されたモデルでpredictが正しい形状を返すこと。"""
        from flare.model_interface.l2l import L2LModel

        model = object.__new__(L2LModel)
        model._device = torch.device("cpu")
        model._window_size = 64

        fake_output = torch.randn(2, 64, 56)
        mock_vqvae = MagicMock(return_value=fake_output)
        model._model = mock_vqvae

        audio = torch.randn(2, 64, 128)
        motion = torch.randn(2, 64, 56)
        result = model.predict(audio, motion)

        assert result.shape == (2, 64, 56)
        mock_vqvae.assert_called_once()


# =========================================================================
# DECAExtractor / SMIRKExtractor (import & interface tests)
# =========================================================================


class TestDECAExtractorInterface:
    """DECAExtractor のインターフェーステスト。"""

    def test_import_succeeds(self) -> None:
        """DECAExtractorがインポートできること。"""
        from flare.extractors.deca import DECAExtractor

        assert DECAExtractor is not None

    def test_init_raises_without_deca(self) -> None:
        """DECAモジュールが無い場合にModelLoadErrorが発生すること。"""
        from flare.extractors.deca import DECAExtractor

        with pytest.raises(ModelLoadError):
            DECAExtractor(model_path="./nonexistent.tar", device="cpu")

    def test_param_keys_defined(self) -> None:
        """パラメータキーが仕様書通りに定義されていること。"""
        from flare.extractors.deca import _DECA_PARAM_KEYS

        assert "shape" in _DECA_PARAM_KEYS
        assert "tex" in _DECA_PARAM_KEYS
        assert "exp" in _DECA_PARAM_KEYS
        assert "pose" in _DECA_PARAM_KEYS
        assert "cam" in _DECA_PARAM_KEYS
        assert "light" in _DECA_PARAM_KEYS
        assert "detail" in _DECA_PARAM_KEYS

    def test_param_dim_is_364(self) -> None:
        """param_dimが364 (100+50+50+6+3+27+128) であること。"""
        from flare.extractors.deca import _DECA_TOTAL_DIM

        assert _DECA_TOTAL_DIM == 364


class TestSMIRKExtractorInterface:
    """SMIRKExtractor のインターフェーステスト。"""

    def test_import_succeeds(self) -> None:
        """SMIRKExtractorがインポートできること。"""
        from flare.extractors.smirk import SMIRKExtractor

        assert SMIRKExtractor is not None

    def test_init_raises_without_smirk(self) -> None:
        """SMIRKモジュールが無い場合にModelLoadErrorが発生すること。"""
        from flare.extractors.smirk import SMIRKExtractor

        with pytest.raises(ModelLoadError):
            SMIRKExtractor(model_path="./nonexistent.pt", device="cpu")

    def test_param_keys_defined(self) -> None:
        """パラメータキーが定義されていること。"""
        from flare.extractors.smirk import _SMIRK_PARAM_KEYS

        assert "shape" in _SMIRK_PARAM_KEYS
        assert "exp" in _SMIRK_PARAM_KEYS
        assert "pose" in _SMIRK_PARAM_KEYS

    def test_param_dim(self) -> None:
        """param_dimが361 (300+50+6+3+2) であること。"""
        from flare.extractors.smirk import _SMIRK_TOTAL_DIM

        assert _SMIRK_TOTAL_DIM == 361
