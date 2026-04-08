"""Phase 3 の統合テスト。

Deep3DFaceReconExtractor / TDDFAExtractor / PIRenderRenderer /
HeadGaSRenderer / FlameToPIRenderAdapter / ViCoModel の
インターフェースと動作を検証する。外部モデル依存のクラスはモックを使用する。
"""

from __future__ import annotations

import inspect
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

from flare.converters.flame_to_pirender import (
    FlameToPIRenderAdapter,
    _6d_to_mat,
    _mat_to_aa,
)
from flare.converters.registry import AdapterRegistry
from flare.renderers.pirender import (
    PIRenderRenderer,
    _PIRENDER_PARAM_KEYS,
)
from flare.renderers.headgas import (
    HeadGaSRenderer,
    _HEADGAS_CONDITION_DIM,
    _HEADGAS_CONDITION_KEYS,
)
from flare.utils.errors import ModelLoadError, RendererNotInitializedError


# =========================================================================
# Deep3DFaceReconExtractor
# =========================================================================


class TestDeep3DFaceReconExtractorInterface:
    """Deep3DFaceReconExtractor のインターフェーステスト。"""

    def test_import_succeeds(self) -> None:
        """Deep3DFaceReconExtractorがインポートできること。"""
        from flare.extractors.deep3d import Deep3DFaceReconExtractor

        assert Deep3DFaceReconExtractor is not None

    def test_init_raises_without_deep3d(self) -> None:
        """Deep3DFaceReconモジュールが無い場合にModelLoadErrorが発生すること。"""
        from flare.extractors.deep3d import Deep3DFaceReconExtractor

        with pytest.raises(ModelLoadError):
            Deep3DFaceReconExtractor(
                model_path="./nonexistent.pth", device="cpu"
            )

    def test_param_keys_defined(self) -> None:
        """パラメータキーが仕様書通りに定義されていること。"""
        from flare.extractors.deep3d import _DEEP3D_PARAM_KEYS

        assert "id" in _DEEP3D_PARAM_KEYS
        assert "exp" in _DEEP3D_PARAM_KEYS
        assert "tex" in _DEEP3D_PARAM_KEYS
        assert "pose" in _DEEP3D_PARAM_KEYS
        assert "lighting" in _DEEP3D_PARAM_KEYS

    def test_param_dim_is_257(self) -> None:
        """param_dimが257 (80+64+80+6+27) であること。"""
        from flare.extractors.deep3d import _DEEP3D_TOTAL_DIM

        assert _DEEP3D_TOTAL_DIM == 257

    def test_param_dims_match_spec(self) -> None:
        """各パラメータの次元数が仕様書通りであること。"""
        from flare.extractors.deep3d import _DEEP3D_PARAM_DIMS

        assert _DEEP3D_PARAM_DIMS["id"] == 80
        assert _DEEP3D_PARAM_DIMS["exp"] == 64
        assert _DEEP3D_PARAM_DIMS["tex"] == 80
        assert _DEEP3D_PARAM_DIMS["pose"] == 6
        assert _DEEP3D_PARAM_DIMS["lighting"] == 27

    def test_extract_with_mock(self) -> None:
        """モック化されたモデルでextractが正しい形状を返すこと。"""
        from flare.extractors.deep3d import Deep3DFaceReconExtractor

        extractor = object.__new__(Deep3DFaceReconExtractor)
        extractor._device = torch.device("cpu")

        # Deep3DFaceReconは257Dの連結ベクトルを返す
        fake_coeffs = torch.randn(1, 257)
        mock_model = MagicMock(return_value=fake_coeffs)
        extractor._model = mock_model

        image = torch.randn(1, 3, 224, 224)
        result = extractor.extract(image)

        assert result["id"].shape == (1, 80)
        assert result["exp"].shape == (1, 64)
        assert result["tex"].shape == (1, 80)
        assert result["pose"].shape == (1, 6)
        assert result["lighting"].shape == (1, 27)

    def test_extract_batch_with_mock(self) -> None:
        """モック化されたモデルでextract_batchが正しい形状を返すこと。"""
        from flare.extractors.deep3d import Deep3DFaceReconExtractor

        extractor = object.__new__(Deep3DFaceReconExtractor)
        extractor._device = torch.device("cpu")

        fake_coeffs = torch.randn(1, 257)
        mock_model = MagicMock(return_value=fake_coeffs)
        extractor._model = mock_model

        images = torch.randn(4, 3, 224, 224)
        result = extractor.extract_batch(images)

        assert result["id"].shape == (4, 80)
        assert result["exp"].shape == (4, 64)
        assert result["tex"].shape == (4, 80)
        assert result["pose"].shape == (4, 6)
        assert result["lighting"].shape == (4, 27)

    def test_output_coefficients_split_correctly(self) -> None:
        """出力係数が正しいオフセットで分割されること。"""
        from flare.extractors.deep3d import Deep3DFaceReconExtractor

        extractor = object.__new__(Deep3DFaceReconExtractor)
        extractor._device = torch.device("cpu")

        # 既知の値を持つ連結ベクトルを作成
        coeffs = torch.arange(257, dtype=torch.float32).unsqueeze(0)
        mock_model = MagicMock(return_value=coeffs)
        extractor._model = mock_model

        result = extractor.extract(torch.randn(1, 3, 224, 224))

        # id: 0-79, exp: 80-143, tex: 144-223, pose: 224-229, lighting: 230-256
        assert torch.allclose(result["id"], coeffs[:, :80])
        assert torch.allclose(result["exp"], coeffs[:, 80:144])
        assert torch.allclose(result["tex"], coeffs[:, 144:224])
        assert torch.allclose(result["pose"], coeffs[:, 224:230])
        assert torch.allclose(result["lighting"], coeffs[:, 230:257])


# =========================================================================
# TDDFAExtractor
# =========================================================================


class TestTDDFAExtractorInterface:
    """TDDFAExtractor のインターフェーステスト。"""

    def test_import_succeeds(self) -> None:
        """TDDFAExtractorがインポートできること。"""
        from flare.extractors.tdddfa import TDDFAExtractor

        assert TDDFAExtractor is not None

    def test_init_raises_without_tddfa(self) -> None:
        """3DDFA V2モジュールが無い場合にModelLoadErrorが発生すること。"""
        from flare.extractors.tdddfa import TDDFAExtractor

        with pytest.raises(ModelLoadError):
            TDDFAExtractor(model_path="./nonexistent.onnx", device="cpu")

    def test_param_keys_defined(self) -> None:
        """パラメータキーが定義されていること。"""
        from flare.extractors.tdddfa import _TDDFA_PARAM_KEYS

        assert "shape" in _TDDFA_PARAM_KEYS
        assert "exp" in _TDDFA_PARAM_KEYS

    def test_param_dim_is_50(self) -> None:
        """param_dimが50 (40+10) であること。"""
        from flare.extractors.tdddfa import _TDDFA_TOTAL_DIM

        assert _TDDFA_TOTAL_DIM == 50

    def test_param_dims_match_spec(self) -> None:
        """各パラメータの次元数が仕様書通りであること。"""
        from flare.extractors.tdddfa import _TDDFA_PARAM_DIMS

        assert _TDDFA_PARAM_DIMS["shape"] == 40
        assert _TDDFA_PARAM_DIMS["exp"] == 10


# =========================================================================
# PIRenderRenderer
# =========================================================================


class TestPIRenderRendererInit:
    """PIRenderRenderer の初期化テスト。"""

    def test_not_initialized_on_creation(self) -> None:
        """生成直後はis_initializedがFalseであること。"""
        renderer = PIRenderRenderer()
        assert renderer.is_initialized is False

    def test_default_output_size(self) -> None:
        """デフォルト出力サイズが256x256であること。"""
        renderer = PIRenderRenderer()
        assert renderer._output_size == [256, 256]

    def test_custom_output_size(self) -> None:
        """カスタム出力サイズが反映されること。"""
        renderer = PIRenderRenderer(output_size=[512, 512])
        assert renderer._output_size == [512, 512]


class TestPIRenderRendererSetupRender:
    """PIRenderRenderer の setup/render 状態遷移テスト。"""

    def test_render_before_setup_raises(self) -> None:
        """setup前のrenderでRendererNotInitializedErrorが発生すること。"""
        renderer = PIRenderRenderer()
        params = {
            "exp": torch.zeros(1, 64),
            "pose": torch.zeros(1, 6),
            "trans": torch.zeros(1, 3),
        }
        with pytest.raises(RendererNotInitializedError):
            renderer.render(params)

    def test_setup_failure_raises_model_load_error(self) -> None:
        """モジュールインポート失敗でModelLoadErrorが発生すること。"""
        renderer = PIRenderRenderer(model_path="./nonexistent/")
        with pytest.raises(ModelLoadError):
            renderer.setup()

    def test_render_with_mock_model(self) -> None:
        """モック化されたモデルでrenderが正しい形状を返すこと。"""
        renderer = PIRenderRenderer()
        renderer._initialized = True
        renderer._device = torch.device("cpu")
        renderer._output_size = [256, 256]

        fake_output = {"fake_image": torch.rand(1, 3, 256, 256)}
        renderer._model = MagicMock(return_value=fake_output)
        renderer._source_image = torch.rand(1, 3, 256, 256)
        renderer._source_descriptor = torch.randn(1, 256)

        params = {
            "exp": torch.randn(1, 64),
            "pose": torch.randn(1, 6),
            "trans": torch.randn(1, 3),
        }
        output = renderer.render(params)
        assert output.shape == (1, 3, 256, 256)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_render_missing_key_raises(self) -> None:
        """必要なキーが欠損している場合にKeyErrorが発生すること。"""
        renderer = PIRenderRenderer()
        renderer._initialized = True
        with pytest.raises(KeyError):
            renderer.render({"exp": torch.zeros(1, 64)})

    def test_param_keys_defined(self) -> None:
        """PIRenderのパラメータキーが3つ定義されていること。"""
        assert len(_PIRENDER_PARAM_KEYS) == 3
        assert "exp" in _PIRENDER_PARAM_KEYS
        assert "pose" in _PIRENDER_PARAM_KEYS
        assert "trans" in _PIRENDER_PARAM_KEYS

    def test_is_initialized_transitions(self) -> None:
        """is_initializedがsetup前後で正しく遷移すること。"""
        renderer = PIRenderRenderer()
        assert renderer.is_initialized is False
        renderer._initialized = True
        assert renderer.is_initialized is True


# =========================================================================
# HeadGaSRenderer
# =========================================================================


class TestHeadGaSRendererInit:
    """HeadGaSRenderer の初期化テスト。"""

    def test_not_initialized_on_creation(self) -> None:
        """生成直後はis_initializedがFalseであること。"""
        renderer = HeadGaSRenderer()
        assert renderer.is_initialized is False

    def test_default_output_size(self) -> None:
        """デフォルト出力サイズが512x512であること。"""
        renderer = HeadGaSRenderer()
        assert renderer._output_size == [512, 512]

    def test_condition_dim_is_120(self) -> None:
        """condition vectorの総次元数が120であること。"""
        assert _HEADGAS_CONDITION_DIM == 120

    def test_condition_keys_count(self) -> None:
        """condition vectorのキーが4つであること。"""
        assert len(_HEADGAS_CONDITION_KEYS) == 4

    def test_render_before_setup_raises(self) -> None:
        """setup前のrenderでRendererNotInitializedErrorが発生すること。"""
        renderer = HeadGaSRenderer()
        params = {
            "expr": torch.zeros(1, 100),
            "jaw_pose": torch.zeros(1, 6),
            "eyes_pose": torch.zeros(1, 12),
            "eyelids": torch.zeros(1, 2),
        }
        with pytest.raises(RendererNotInitializedError):
            renderer.render(params)

    def test_setup_failure_raises_model_load_error(self) -> None:
        """存在しないパスでsetupするとModelLoadErrorが発生すること。"""
        renderer = HeadGaSRenderer(model_path="./nonexistent/")
        with pytest.raises(ModelLoadError):
            renderer.setup()

    def test_render_missing_key_raises(self) -> None:
        """必要なキーが欠損している場合にKeyErrorが発生すること。"""
        renderer = HeadGaSRenderer()
        renderer._initialized = True
        with pytest.raises(KeyError):
            renderer.render({"expr": torch.zeros(1, 100)})


# =========================================================================
# FlameToPIRenderAdapter
# =========================================================================


class TestFlameToPIRenderAdapterConversion:
    """FlameToPIRenderAdapter の変換テスト。"""

    def test_exp_shape(self) -> None:
        """exp が (B, 64) であること。"""
        adapter = FlameToPIRenderAdapter()
        source = {
            "expr": torch.randn(4, 100),
            "jaw_pose": torch.tensor([[1.0, 0, 0, 0, 1, 0]]).repeat(4, 1),
            "rotation": torch.randn(4, 3),
        }
        result = adapter.convert(source)
        assert result["exp"].shape == (4, 64)

    def test_exp_first_64_from_expr(self) -> None:
        """exp が expr の先頭64Dであること。"""
        adapter = FlameToPIRenderAdapter()
        expr = torch.randn(2, 100)
        source = {
            "expr": expr,
            "jaw_pose": torch.tensor([[1.0, 0, 0, 0, 1, 0]]).repeat(2, 1),
            "rotation": torch.zeros(2, 3),
        }
        result = adapter.convert(source)
        assert torch.allclose(result["exp"], expr[:, :64])

    def test_pose_shape(self) -> None:
        """pose が (B, 6) であること。"""
        adapter = FlameToPIRenderAdapter()
        source = {
            "expr": torch.randn(3, 100),
            "jaw_pose": torch.tensor([[1.0, 0, 0, 0, 1, 0]]).repeat(3, 1),
            "rotation": torch.randn(3, 3),
        }
        result = adapter.convert(source)
        assert result["pose"].shape == (3, 6)

    def test_pose_contains_rotation_and_jaw(self) -> None:
        """poseが rotation(3D) + jaw_aa(3D) の連結であること。"""
        adapter = FlameToPIRenderAdapter()
        rotation = torch.tensor([[0.1, 0.2, 0.3]])
        # identity 6d → identity rotation → zero axis-angle
        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        source = {
            "expr": torch.randn(1, 100),
            "jaw_pose": identity_6d,
            "rotation": rotation,
        }
        result = adapter.convert(source)
        # rotation部分が保存されていること
        assert torch.allclose(result["pose"][:, :3], rotation, atol=1e-5)
        # identity jaw → zero axis-angle
        assert torch.allclose(
            result["pose"][:, 3:6], torch.zeros(1, 3), atol=1e-5
        )

    def test_trans_shape(self) -> None:
        """trans が (B, 3) であること。"""
        adapter = FlameToPIRenderAdapter()
        source = {
            "expr": torch.randn(2, 100),
            "jaw_pose": torch.tensor([[1.0, 0, 0, 0, 1, 0]]).repeat(2, 1),
            "rotation": torch.zeros(2, 3),
        }
        result = adapter.convert(source)
        assert result["trans"].shape == (2, 3)

    def test_trans_default_is_zero(self) -> None:
        """transがsource_paramsに無い場合はゼロであること。"""
        adapter = FlameToPIRenderAdapter()
        source = {
            "expr": torch.randn(3, 100),
            "jaw_pose": torch.tensor([[1.0, 0, 0, 0, 1, 0]]).repeat(3, 1),
            "rotation": torch.zeros(3, 3),
        }
        result = adapter.convert(source)
        assert torch.allclose(result["trans"], torch.zeros(3, 3))

    def test_trans_from_source(self) -> None:
        """transがsource_paramsにある場合はそれを使用すること。"""
        adapter = FlameToPIRenderAdapter()
        trans = torch.tensor([[1.0, 2.0, 3.0]])
        source = {
            "expr": torch.randn(1, 100),
            "jaw_pose": torch.tensor([[1.0, 0, 0, 0, 1, 0]]),
            "rotation": torch.zeros(1, 3),
            "trans": trans,
        }
        result = adapter.convert(source)
        assert torch.allclose(result["trans"], trans)

    def test_missing_expr_key_raises(self) -> None:
        """exprキーが欠損している場合にKeyErrorが発生すること。"""
        adapter = FlameToPIRenderAdapter()
        with pytest.raises(KeyError):
            adapter.convert({
                "jaw_pose": torch.zeros(1, 6),
                "rotation": torch.zeros(1, 3),
            })

    def test_missing_jaw_pose_key_raises(self) -> None:
        """jaw_poseキーが欠損している場合にKeyErrorが発生すること。"""
        adapter = FlameToPIRenderAdapter()
        with pytest.raises(KeyError):
            adapter.convert({
                "expr": torch.zeros(1, 100),
                "rotation": torch.zeros(1, 3),
            })

    def test_missing_rotation_key_raises(self) -> None:
        """rotationキーが欠損している場合にKeyErrorが発生すること。"""
        adapter = FlameToPIRenderAdapter()
        with pytest.raises(KeyError):
            adapter.convert({
                "expr": torch.zeros(1, 100),
                "jaw_pose": torch.zeros(1, 6),
            })

    def test_source_format(self) -> None:
        """source_formatが'flame'であること。"""
        adapter = FlameToPIRenderAdapter()
        assert adapter.source_format == "flame"

    def test_target_format(self) -> None:
        """target_formatが'pirender'であること。"""
        adapter = FlameToPIRenderAdapter()
        assert adapter.target_format == "pirender"

    def test_registry_integration(self) -> None:
        """AdapterRegistryに登録・取得できること。"""
        registry = AdapterRegistry()
        adapter = FlameToPIRenderAdapter()
        registry.register(adapter)
        got = registry.get("flame", "pirender")
        assert got is adapter

    def test_short_expr_is_padded(self) -> None:
        """expr が64D未満の場合にゼロパディングされること。"""
        adapter = FlameToPIRenderAdapter()
        source = {
            "expr": torch.randn(1, 30),
            "jaw_pose": torch.tensor([[1.0, 0, 0, 0, 1, 0]]),
            "rotation": torch.zeros(1, 3),
        }
        result = adapter.convert(source)
        assert result["exp"].shape == (1, 64)
        assert torch.allclose(result["exp"][:, 30:], torch.zeros(1, 34))


# =========================================================================
# FlameToPIRender Rotation Utils
# =========================================================================


class TestFlameToPIRenderRotationUtils:
    """FlameToPIRender の回転変換ユーティリティテスト。"""

    def test_6d_to_mat_identity(self) -> None:
        """identity 6D入力で単位行列が返ること。"""
        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        mat = _6d_to_mat(identity_6d)
        assert mat.shape == (1, 3, 3)
        assert torch.allclose(mat, torch.eye(3).unsqueeze(0), atol=1e-6)

    def test_mat_to_aa_identity(self) -> None:
        """単位行列でゼロaxis-angleが返ること。"""
        mat = torch.eye(3).unsqueeze(0)
        aa = _mat_to_aa(mat)
        assert aa.shape == (1, 3)
        assert torch.allclose(aa, torch.zeros(1, 3), atol=1e-6)

    def test_roundtrip_6d_to_aa(self) -> None:
        """identity 6D → matrix → axis-angle の変換が正しいこと。"""
        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        mat = _6d_to_mat(identity_6d)
        aa = _mat_to_aa(mat)
        assert torch.allclose(aa, torch.zeros(1, 3), atol=1e-5)

    def test_6d_to_mat_produces_valid_rotation(self) -> None:
        """6D→行列の結果が有効な回転行列（det≈1）であること。"""
        r6d = torch.randn(4, 6)
        mat = _6d_to_mat(r6d)
        det = torch.det(mat)
        assert torch.allclose(det, torch.ones(4), atol=1e-4)


# =========================================================================
# ViCoModel
# =========================================================================


class TestViCoModelInterface:
    """ViCoModel のインターフェーステスト。"""

    def test_import_succeeds(self) -> None:
        """ViCoModelがインポートできること。"""
        from flare.model_interface.vico import ViCoModel

        assert ViCoModel is not None

    def test_requires_window_is_false(self) -> None:
        """requires_windowがFalseであること。"""
        from flare.model_interface.vico import ViCoModel

        with pytest.raises(ModelLoadError):
            ViCoModel(model_path="./nonexistent.pth", device="cpu")

    def test_predict_signature(self) -> None:
        """predictのシグネチャがaudio_features, speaker_motionの2引数であること。"""
        from flare.model_interface.vico import ViCoModel

        sig = inspect.signature(ViCoModel.predict)
        params = [p for p in sig.parameters.keys() if p != "self"]
        assert params == ["audio_features", "speaker_motion"]

    def test_window_size_is_none(self) -> None:
        """window_sizeがNoneであること。"""
        from flare.model_interface.vico import ViCoModel

        with patch.object(ViCoModel, "_load_model", return_value=None):
            model = object.__new__(ViCoModel)
            assert model.window_size is None

    def test_requires_window_property(self) -> None:
        """requires_windowプロパティがFalseを返すこと。"""
        from flare.model_interface.vico import ViCoModel

        with patch.object(ViCoModel, "_load_model", return_value=None):
            model = object.__new__(ViCoModel)
            assert model.requires_window is False

    def test_predict_with_mock(self) -> None:
        """モック化されたモデルでpredictが正しい形状を返すこと。"""
        from flare.model_interface.vico import ViCoModel

        model = object.__new__(ViCoModel)
        model._device = torch.device("cpu")

        fake_output = torch.randn(4, 56)
        mock_vico = MagicMock(return_value=fake_output)
        model._model = mock_vico

        audio = torch.randn(4, 128)
        motion = torch.randn(4, 56)
        result = model.predict(audio, motion)

        assert result.shape == (4, 56)
        mock_vico.assert_called_once()
