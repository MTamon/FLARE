"""Phase 5 の統合テスト。

BFMToFlameAdapter / detect_eye_pose / CLI dry-run の
動作を検証する。外部モデル依存はモック・フォールバックで回避する。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from flare.converters.bfm_to_flame import (
    BFMToFlameAdapter,
    _BFM_2017_EXP_DIM,
    _BFM_CROPPED_EXP_DIM,
    _BFM_CROPPED_SHAPE_DIM,
    _FLAME_EXPR_DIM,
    _FLAME_SHAPE_DIM,
)
from flare.converters.registry import AdapterRegistry
from flare.utils.face_detect import (
    _axis_angle_to_rotation_6d,
    _blendshapes_to_flame_eye,
    _default_eyelids,
    _default_eyes_pose,
)


# =========================================================================
# BFMToFlameAdapter - 基本形状テスト
# =========================================================================


class TestBFMToFlameAdapterShapes:
    """BFMToFlameAdapter の出力形状テスト。"""

    def test_expr_shape_bfm2017(self) -> None:
        """BFM 2017 exp 64D → FLAME expr 100D の形状が正しいこと。"""
        adapter = BFMToFlameAdapter(bfm_variant="bfm2017")
        source = {
            "exp": torch.randn(2, 64),
            "pose": torch.randn(2, 6),
            "id": torch.randn(2, 80),
        }
        result = adapter.convert(source)
        assert result["expr"].shape == (2, _FLAME_EXPR_DIM)

    def test_shape_output_dim(self) -> None:
        """id 80D → FLAME shape 300D の形状が正しいこと。"""
        adapter = BFMToFlameAdapter()
        source = {
            "exp": torch.randn(3, 64),
            "pose": torch.randn(3, 6),
            "id": torch.randn(3, 80),
        }
        result = adapter.convert(source)
        assert result["shape"].shape == (3, _FLAME_SHAPE_DIM)

    def test_pose_passthrough(self) -> None:
        """pose が (B, 6) でパススルーされること。"""
        adapter = BFMToFlameAdapter()
        pose = torch.randn(4, 6)
        source = {
            "exp": torch.randn(4, 64),
            "pose": pose,
            "id": torch.randn(4, 80),
        }
        result = adapter.convert(source)
        assert result["pose"].shape == (4, 6)
        assert torch.allclose(result["pose"], pose)

    def test_cropped_bfm_exp_10d(self) -> None:
        """cropped BFM 2009 exp 10D → FLAME expr 100D が変換されること。"""
        adapter = BFMToFlameAdapter(bfm_variant="cropped_bfm2009")
        source = {
            "exp": torch.randn(1, 10),
            "pose": torch.randn(1, 6),
            "shape": torch.randn(1, 40),
        }
        result = adapter.convert(source)
        assert result["expr"].shape == (1, _FLAME_EXPR_DIM)

    def test_cropped_bfm_shape_40d(self) -> None:
        """cropped BFM 2009 shape 40D → FLAME shape 300D が変換されること。"""
        adapter = BFMToFlameAdapter(bfm_variant="cropped_bfm2009")
        source = {
            "exp": torch.randn(1, 10),
            "pose": torch.randn(1, 6),
            "shape": torch.randn(1, 40),
        }
        result = adapter.convert(source)
        assert result["shape"].shape == (1, _FLAME_SHAPE_DIM)

    def test_no_id_returns_zero_shape(self) -> None:
        """idキーもshapeキーも無い場合にゼロのFLAME shapeが返ること。"""
        adapter = BFMToFlameAdapter()
        source = {
            "exp": torch.randn(2, 64),
            "pose": torch.randn(2, 6),
        }
        result = adapter.convert(source)
        assert result["shape"].shape == (2, _FLAME_SHAPE_DIM)
        assert torch.allclose(result["shape"], torch.zeros(2, _FLAME_SHAPE_DIM))


# =========================================================================
# BFMToFlameAdapter - フォールバック変換値テスト
# =========================================================================


class TestBFMToFlameAdapterValues:
    """BFMToFlameAdapter のフォールバック変換値テスト。"""

    def test_expr_first_64_preserved(self) -> None:
        """フォールバック時にexprの先頭64Dが入力expと一致すること。"""
        adapter = BFMToFlameAdapter()
        exp = torch.randn(2, 64)
        source = {"exp": exp, "pose": torch.randn(2, 6)}
        result = adapter.convert(source)
        assert torch.allclose(result["expr"][:, :64], exp)

    def test_expr_padding_is_zero(self) -> None:
        """フォールバック時にexprの後半がゼロパディングであること。"""
        adapter = BFMToFlameAdapter()
        source = {"exp": torch.randn(3, 64), "pose": torch.randn(3, 6)}
        result = adapter.convert(source)
        assert torch.allclose(
            result["expr"][:, 64:], torch.zeros(3, _FLAME_EXPR_DIM - 64)
        )

    def test_shape_padding_is_zero(self) -> None:
        """フォールバック時にshapeの後半がゼロパディングであること。"""
        adapter = BFMToFlameAdapter()
        id_coeff = torch.randn(2, 80)
        source = {"exp": torch.randn(2, 64), "pose": torch.randn(2, 6), "id": id_coeff}
        result = adapter.convert(source)
        assert torch.allclose(result["shape"][:, :80], id_coeff)
        assert torch.allclose(
            result["shape"][:, 80:], torch.zeros(2, _FLAME_SHAPE_DIM - 80)
        )

    def test_small_exp_padded(self) -> None:
        """10Dのexpがゼロパディングで100Dになること（3DDFA形式）。"""
        adapter = BFMToFlameAdapter(bfm_variant="cropped_bfm2009")
        exp = torch.randn(1, 10)
        source = {"exp": exp, "pose": torch.randn(1, 6)}
        result = adapter.convert(source)
        assert torch.allclose(result["expr"][:, :10], exp)
        assert torch.allclose(result["expr"][:, 10:], torch.zeros(1, 90))

    def test_missing_exp_key_raises(self) -> None:
        """expキーが欠損している場合にKeyErrorが発生すること。"""
        adapter = BFMToFlameAdapter()
        with pytest.raises(KeyError):
            adapter.convert({"pose": torch.zeros(1, 6)})

    def test_missing_pose_key_raises(self) -> None:
        """poseキーが欠損している場合にKeyErrorが発生すること。"""
        adapter = BFMToFlameAdapter()
        with pytest.raises(KeyError):
            adapter.convert({"exp": torch.zeros(1, 64)})


# =========================================================================
# BFMToFlameAdapter - プロパティテスト
# =========================================================================


class TestBFMToFlameAdapterProperties:
    """BFMToFlameAdapter のプロパティテスト。"""

    def test_source_format(self) -> None:
        """source_formatが'bfm'であること。"""
        adapter = BFMToFlameAdapter()
        assert adapter.source_format == "bfm"

    def test_target_format(self) -> None:
        """target_formatが'flame'であること。"""
        adapter = BFMToFlameAdapter()
        assert adapter.target_format == "flame"

    def test_bfm_variant_default(self) -> None:
        """デフォルトのbfm_variantが'bfm2017'であること。"""
        adapter = BFMToFlameAdapter()
        assert adapter.bfm_variant == "bfm2017"

    def test_bfm_variant_custom(self) -> None:
        """カスタムbfm_variantが設定できること。"""
        adapter = BFMToFlameAdapter(bfm_variant="cropped_bfm2009")
        assert adapter.bfm_variant == "cropped_bfm2009"

    def test_registry_integration(self) -> None:
        """AdapterRegistryに登録・取得できること。"""
        registry = AdapterRegistry()
        adapter = BFMToFlameAdapter()
        registry.register(adapter)
        got = registry.get("bfm", "flame")
        assert got is adapter


# =========================================================================
# BFMToFlameAdapter - バッチサイズテスト
# =========================================================================


class TestBFMToFlameAdapterBatch:
    """BFMToFlameAdapter のバッチサイズテスト。"""

    def test_batch_size_1(self) -> None:
        """バッチサイズ1で正しく動作すること。"""
        adapter = BFMToFlameAdapter()
        source = {
            "exp": torch.randn(1, 64),
            "pose": torch.randn(1, 6),
            "id": torch.randn(1, 80),
        }
        result = adapter.convert(source)
        assert result["expr"].shape[0] == 1

    def test_batch_size_16(self) -> None:
        """バッチサイズ16で正しく動作すること。"""
        adapter = BFMToFlameAdapter()
        source = {
            "exp": torch.randn(16, 64),
            "pose": torch.randn(16, 6),
            "id": torch.randn(16, 80),
        }
        result = adapter.convert(source)
        assert result["expr"].shape[0] == 16
        assert result["shape"].shape[0] == 16


# =========================================================================
# detect_eye_pose ヘルパー関数テスト
# =========================================================================


class TestDetectEyePoseHelpers:
    """detect_eye_pose のヘルパー関数テスト。"""

    def test_default_eyes_pose_shape(self) -> None:
        """デフォルトeyes_poseが (1, 12) であること。"""
        result = _default_eyes_pose()
        assert result.shape == (1, 12)

    def test_default_eyes_pose_is_identity(self) -> None:
        """デフォルトeyes_poseがidentity rotation × 2であること。"""
        result = _default_eyes_pose()
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        expected = torch.cat([identity_6d, identity_6d]).unsqueeze(0)
        assert torch.allclose(result, expected)

    def test_default_eyelids_shape(self) -> None:
        """デフォルトeyelidsが (1, 2) であること。"""
        result = _default_eyelids()
        assert result.shape == (1, 2)

    def test_default_eyelids_is_zero(self) -> None:
        """デフォルトeyelidsがゼロであること。"""
        result = _default_eyelids()
        assert torch.allclose(result, torch.zeros(1, 2))

    def test_axis_angle_to_6d_identity(self) -> None:
        """ゼロaxis-angleでidentity 6D rotationが返ること。"""
        aa = torch.zeros(3)
        r6d = _axis_angle_to_rotation_6d(aa)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        assert torch.allclose(r6d, expected, atol=1e-6)

    def test_axis_angle_to_6d_shape(self) -> None:
        """axis_angle_to_rotation_6dの出力が (6,) であること。"""
        aa = torch.tensor([0.1, 0.2, 0.0])
        r6d = _axis_angle_to_rotation_6d(aa)
        assert r6d.shape == (6,)

    def test_axis_angle_to_6d_nonzero(self) -> None:
        """非ゼロaxis-angleで非identity 6Dが返ること。"""
        aa = torch.tensor([0.3, 0.0, 0.0])
        r6d = _axis_angle_to_rotation_6d(aa)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        assert not torch.allclose(r6d, identity, atol=1e-3)


# =========================================================================
# blendshapes_to_flame_eye テスト
# =========================================================================


class TestBlendshapesToFlameEye:
    """blendshapes → FLAME eye pose 変換テスト。"""

    def test_neutral_blendshapes_output_shapes(self) -> None:
        """ニュートラルblendshapeでeyes_pose (1,12) とeyelids (1,2) が返ること。"""
        blendshapes: dict[str, float] = {}
        eyes_pose, eyelids = _blendshapes_to_flame_eye(blendshapes)
        assert eyes_pose.shape == (1, 12)
        assert eyelids.shape == (1, 2)

    def test_neutral_blendshapes_near_identity(self) -> None:
        """ニュートラルblendshapeでeyes_poseがidentityに近いこと。"""
        blendshapes: dict[str, float] = {}
        eyes_pose, _ = _blendshapes_to_flame_eye(blendshapes)
        identity = _default_eyes_pose()
        assert torch.allclose(eyes_pose, identity, atol=1e-5)

    def test_neutral_blendshapes_zero_eyelids(self) -> None:
        """ニュートラルblendshapeでeyelidsがゼロであること。"""
        blendshapes: dict[str, float] = {}
        _, eyelids = _blendshapes_to_flame_eye(blendshapes)
        assert torch.allclose(eyelids, torch.zeros(1, 2))

    def test_blink_left(self) -> None:
        """左眼の瞬きでeyelids[0]が正の値を持つこと。"""
        blendshapes = {"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.0}
        _, eyelids = _blendshapes_to_flame_eye(blendshapes)
        assert eyelids[0, 0].item() == pytest.approx(0.8, abs=1e-5)
        assert eyelids[0, 1].item() == pytest.approx(0.0, abs=1e-5)

    def test_blink_both(self) -> None:
        """両眼の瞬きでeyelidsの両方が正の値を持つこと。"""
        blendshapes = {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0}
        _, eyelids = _blendshapes_to_flame_eye(blendshapes)
        assert eyelids[0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert eyelids[0, 1].item() == pytest.approx(1.0, abs=1e-5)

    def test_look_up_changes_eyes_pose(self) -> None:
        """上方向を見るblendshapeでeyes_poseがidentityから変化すること。"""
        blendshapes = {"eyeLookUpLeft": 1.0, "eyeLookUpRight": 1.0}
        eyes_pose, _ = _blendshapes_to_flame_eye(blendshapes)
        identity = _default_eyes_pose()
        assert not torch.allclose(eyes_pose, identity, atol=1e-3)

    def test_eyelids_value_range(self) -> None:
        """eyelidsの値が0-1の範囲内であること。"""
        blendshapes = {"eyeBlinkLeft": 0.5, "eyeBlinkRight": 0.3}
        _, eyelids = _blendshapes_to_flame_eye(blendshapes)
        assert eyelids.min() >= 0.0
        assert eyelids.max() <= 1.0

    def test_eyes_pose_6d_components(self) -> None:
        """eyes_poseが左右各6D（合計12D）で構成されること。"""
        blendshapes = {
            "eyeLookUpLeft": 0.5,
            "eyeLookInRight": 0.3,
        }
        eyes_pose, _ = _blendshapes_to_flame_eye(blendshapes)
        left_6d = eyes_pose[0, :6]
        right_6d = eyes_pose[0, 6:]
        assert left_6d.shape == (6,)
        assert right_6d.shape == (6,)


# =========================================================================
# FaceDetector.detect_eye_pose 統合テスト
# =========================================================================


class TestFaceDetectorDetectEyePose:
    """FaceDetector.detect_eye_pose のフォールバック動作テスト。"""

    def test_detect_eye_pose_returns_tuple(self) -> None:
        """detect_eye_poseがタプルを返すこと（モックで検証）。"""
        from flare.utils.face_detect import FaceDetector

        detector = MagicMock(spec=FaceDetector)
        detector.detect_eye_pose = FaceDetector.detect_eye_pose.__get__(
            detector, FaceDetector
        )
        detector._get_blendshapes = MagicMock(return_value=None)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 300, 300)
        result = detector.detect_eye_pose(frame, bbox)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_detect_eye_pose_fallback_shapes(self) -> None:
        """blendshape取得失敗時にデフォルト値が返ること。"""
        from flare.utils.face_detect import FaceDetector

        detector = MagicMock(spec=FaceDetector)
        detector.detect_eye_pose = FaceDetector.detect_eye_pose.__get__(
            detector, FaceDetector
        )
        detector._get_blendshapes = MagicMock(return_value=None)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 300, 300)
        eyes_pose, eyelids = detector.detect_eye_pose(frame, bbox)

        assert eyes_pose.shape == (1, 12)
        assert eyelids.shape == (1, 2)

    def test_detect_eye_pose_with_blendshapes(self) -> None:
        """blendshapeが利用可能な場合に正しい形状が返ること。"""
        from flare.utils.face_detect import FaceDetector

        detector = MagicMock(spec=FaceDetector)
        detector.detect_eye_pose = FaceDetector.detect_eye_pose.__get__(
            detector, FaceDetector
        )

        mock_blendshapes = {
            "eyeLookUpLeft": 0.3,
            "eyeLookDownLeft": 0.1,
            "eyeLookInLeft": 0.0,
            "eyeLookOutLeft": 0.2,
            "eyeLookUpRight": 0.3,
            "eyeLookDownRight": 0.1,
            "eyeLookInRight": 0.2,
            "eyeLookOutRight": 0.0,
            "eyeBlinkLeft": 0.1,
            "eyeBlinkRight": 0.15,
        }
        detector._get_blendshapes = MagicMock(return_value=mock_blendshapes)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 300, 300)
        eyes_pose, eyelids = detector.detect_eye_pose(frame, bbox)

        assert eyes_pose.shape == (1, 12)
        assert eyelids.shape == (1, 2)
        assert eyelids[0, 0].item() == pytest.approx(0.1, abs=1e-5)
        assert eyelids[0, 1].item() == pytest.approx(0.15, abs=1e-5)


# =========================================================================
# Module-level constants テスト
# =========================================================================


class TestBFMToFlameConstants:
    """BFMToFlame モジュール定数テスト。"""

    def test_bfm_2017_exp_dim(self) -> None:
        """BFM 2017 exp次元数が64であること。"""
        assert _BFM_2017_EXP_DIM == 64

    def test_bfm_cropped_exp_dim(self) -> None:
        """cropped BFM exp次元数が10であること。"""
        assert _BFM_CROPPED_EXP_DIM == 10

    def test_bfm_cropped_shape_dim(self) -> None:
        """cropped BFM shape次元数が40であること。"""
        assert _BFM_CROPPED_SHAPE_DIM == 40

    def test_flame_expr_dim(self) -> None:
        """FLAME expr次元数が100であること。"""
        assert _FLAME_EXPR_DIM == 100

    def test_flame_shape_dim(self) -> None:
        """FLAME shape次元数が300であること。"""
        assert _FLAME_SHAPE_DIM == 300
