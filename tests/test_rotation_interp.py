"""回転補間ユーティリティ ``flare.utils.rotation_interp`` のテスト。

以下の観点でテストする:
    - 軸角 ⇄ 四元数変換の往復
    - SLERP の両端一致・中点性質
    - SLERP と線形補間の小角領域での近似一致
    - SLERP の大角度領域での正しさ（±170° 問題）
    - ギャップ補間の max_gap 制限
    - 先頭・末尾ギャップの非補間
"""

from __future__ import annotations

import numpy as np
import pytest

from flare.utils.rotation_interp import (
    axis_angle_to_quaternion,
    interp_rotation,
    interp_rotation_linear,
    interp_rotation_slerp,
    quaternion_to_axis_angle,
    slerp_axis_angle,
    slerp_quaternion,
)


class TestAxisAngleQuaternionRoundTrip:
    """軸角 ⇄ 四元数変換の往復テスト。"""

    def test_identity(self) -> None:
        aa = np.array([0.0, 0.0, 0.0])
        q = axis_angle_to_quaternion(aa)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        aa2 = quaternion_to_axis_angle(q)
        np.testing.assert_allclose(aa2, aa, atol=1e-6)

    def test_x_axis_90deg(self) -> None:
        aa = np.array([np.pi / 2, 0.0, 0.0])
        q = axis_angle_to_quaternion(aa)
        # w = cos(π/4) ≈ 0.7071, x = sin(π/4) ≈ 0.7071
        assert q[0] == pytest.approx(np.cos(np.pi / 4), abs=1e-6)
        assert q[1] == pytest.approx(np.sin(np.pi / 4), abs=1e-6)
        aa2 = quaternion_to_axis_angle(q)
        np.testing.assert_allclose(aa2, aa, atol=1e-6)

    def test_batched_round_trip(self) -> None:
        rng = np.random.default_rng(42)
        aa = rng.standard_normal((10, 3)).astype(np.float32) * 0.5
        q = axis_angle_to_quaternion(aa)
        aa2 = quaternion_to_axis_angle(q)
        np.testing.assert_allclose(aa2, aa, atol=1e-5)

    def test_quaternion_unit_norm(self) -> None:
        rng = np.random.default_rng(0)
        aa = rng.standard_normal((20, 3)).astype(np.float32)
        q = axis_angle_to_quaternion(aa)
        norms = np.linalg.norm(q, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class TestSlerpQuaternion:
    """SLERP の基本性質のテスト。"""

    def test_endpoints(self) -> None:
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = axis_angle_to_quaternion(np.array([0.0, 1.0, 0.0]))
        t0 = np.array([0.0])
        t1 = np.array([1.0])
        np.testing.assert_allclose(slerp_quaternion(q0, q1, t0), q0, atol=1e-6)
        np.testing.assert_allclose(slerp_quaternion(q0, q1, t1), q1, atol=1e-6)

    def test_midpoint_small_angle(self) -> None:
        # 小角度なら SLERP と線形補間がほぼ一致
        aa0 = np.array([0.0, 0.1, 0.0])
        aa1 = np.array([0.0, 0.2, 0.0])
        mid_slerp = slerp_axis_angle(aa0, aa1, 0.5)
        mid_linear = 0.5 * (aa0 + aa1)
        np.testing.assert_allclose(mid_slerp, mid_linear, atol=1e-4)

    def test_identity_interpolation(self) -> None:
        # 同じ四元数同士の補間は元の値
        q = axis_angle_to_quaternion(np.array([0.3, 0.4, 0.5]))
        t = np.array([0.5])
        result = slerp_quaternion(q, q, t)
        np.testing.assert_allclose(result, q, atol=1e-6)


class TestSlerpLargeAngle:
    """SLERP の大角度領域での正しさを検証する。"""

    def test_short_arc_selection(self) -> None:
        # ±170° yaw → 中点は±180° (真後ろ)、正面 (0°) になってはならない
        aa_pos = np.array([0.0, np.deg2rad(170.0), 0.0])
        aa_neg = np.array([0.0, np.deg2rad(-170.0), 0.0])
        mid = slerp_axis_angle(aa_pos, aa_neg, 0.5)
        # 軸角表現では ±π 近傍の「真後ろ」を表す
        angle = float(np.linalg.norm(mid))
        assert angle > np.deg2rad(170.0)

    def test_diverges_from_linear_large_angle(self) -> None:
        # 大角度 (90° 差) では SLERP と線形補間が明確に違う
        aa0 = np.array([np.deg2rad(60.0), 0.0, 0.0])
        aa1 = np.array([0.0, np.deg2rad(60.0), 0.0])
        mid_slerp = slerp_axis_angle(aa0, aa1, 0.5)
        mid_linear = 0.5 * (aa0 + aa1)
        diff = float(np.linalg.norm(mid_slerp - mid_linear))
        assert diff > 1e-3  # 明確に異なる


class TestInterpRotationSlerp:
    """interp_rotation_slerp の動作テスト。"""

    def test_no_gap(self) -> None:
        rotations = np.array(
            [[0.0, 0.1, 0.0], [0.0, 0.2, 0.0], [0.0, 0.3, 0.0]], dtype=np.float32
        )
        mask = np.array([True, True, True])
        filled, new_mask = interp_rotation_slerp(rotations, mask, max_gap=5)
        np.testing.assert_array_equal(filled, rotations)
        np.testing.assert_array_equal(new_mask, mask)

    def test_single_frame_gap(self) -> None:
        rotations = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32
        )
        mask = np.array([True, False, True])
        filled, new_mask = interp_rotation_slerp(rotations, mask, max_gap=5)
        # 中点は (0, 0.1, 0) 付近
        np.testing.assert_allclose(filled[1], [0.0, 0.1, 0.0], atol=1e-4)
        assert np.all(new_mask)

    def test_long_gap_not_filled(self) -> None:
        rotations = np.zeros((10, 3), dtype=np.float32)
        rotations[0] = [0.0, 0.1, 0.0]
        rotations[9] = [0.0, 0.5, 0.0]
        mask = np.array([True] + [False] * 8 + [True])
        _, new_mask = interp_rotation_slerp(rotations, mask, max_gap=5)
        assert not new_mask[1:9].any()

    def test_leading_gap_not_filled(self) -> None:
        rotations = np.zeros((5, 3), dtype=np.float32)
        rotations[3] = [0.0, 0.3, 0.0]
        rotations[4] = [0.0, 0.4, 0.0]
        mask = np.array([False, False, False, True, True])
        _, new_mask = interp_rotation_slerp(rotations, mask, max_gap=10)
        assert not new_mask[:3].any()

    def test_trailing_gap_not_filled(self) -> None:
        rotations = np.zeros((5, 3), dtype=np.float32)
        rotations[0] = [0.0, 0.1, 0.0]
        rotations[1] = [0.0, 0.2, 0.0]
        mask = np.array([True, True, False, False, False])
        _, new_mask = interp_rotation_slerp(rotations, mask, max_gap=10)
        assert not new_mask[2:].any()

    def test_multi_frame_gap_monotone(self) -> None:
        # 4 フレームのギャップで単調な回転増加
        rotations = np.zeros((6, 3), dtype=np.float32)
        rotations[0] = [0.0, 0.0, 0.0]
        rotations[5] = [0.0, 0.5, 0.0]
        mask = np.array([True, False, False, False, False, True])
        filled, new_mask = interp_rotation_slerp(rotations, mask, max_gap=5)
        assert np.all(new_mask)
        # 補間された yaw が単調増加
        yaws = filled[:, 1]
        assert np.all(np.diff(yaws) > 0)

    def test_invalid_shape_raises(self) -> None:
        rotations = np.zeros((5, 4), dtype=np.float32)
        mask = np.array([True] * 5)
        with pytest.raises(ValueError, match="shape"):
            interp_rotation_slerp(rotations, mask, max_gap=5)


class TestInterpRotationLinear:
    """旧版互換の軸角線形補間のテスト。"""

    def test_matches_linear_small_angle(self) -> None:
        rotations = np.zeros((3, 3), dtype=np.float32)
        rotations[0] = [0.0, 0.0, 0.0]
        rotations[2] = [0.0, 0.2, 0.0]
        mask = np.array([True, False, True])
        filled, _ = interp_rotation_linear(rotations, mask, max_gap=5)
        np.testing.assert_allclose(filled[1], [0.0, 0.1, 0.0], atol=1e-6)

    def test_small_angle_agrees_with_slerp(self) -> None:
        # 小角領域では SLERP とほぼ一致
        rng = np.random.default_rng(7)
        rotations = np.zeros((5, 3), dtype=np.float32)
        rotations[0] = rng.standard_normal(3) * 0.05
        rotations[4] = rng.standard_normal(3) * 0.05
        mask = np.array([True, False, False, False, True])
        filled_lin, _ = interp_rotation_linear(rotations.copy(), mask, max_gap=5)
        filled_slerp, _ = interp_rotation_slerp(rotations.copy(), mask, max_gap=5)
        np.testing.assert_allclose(filled_lin, filled_slerp, atol=1e-3)


class TestInterpRotationDispatch:
    """interp_rotation ディスパッチャのテスト。"""

    def test_dispatch_slerp(self) -> None:
        rotations = np.zeros((3, 3), dtype=np.float32)
        rotations[0] = [0.0, 0.0, 0.0]
        rotations[2] = [0.0, 0.2, 0.0]
        mask = np.array([True, False, True])
        filled, _ = interp_rotation(rotations, mask, max_gap=5, method="slerp")
        assert filled[1, 1] == pytest.approx(0.1, abs=1e-4)

    def test_dispatch_linear(self) -> None:
        rotations = np.zeros((3, 3), dtype=np.float32)
        rotations[0] = [0.0, 0.0, 0.0]
        rotations[2] = [0.0, 0.2, 0.0]
        mask = np.array([True, False, True])
        filled, _ = interp_rotation(rotations, mask, max_gap=5, method="linear")
        assert filled[1, 1] == pytest.approx(0.1, abs=1e-6)

    def test_unknown_method_raises(self) -> None:
        rotations = np.zeros((3, 3), dtype=np.float32)
        mask = np.array([True, False, True])
        with pytest.raises(ValueError, match="Unknown rotation"):
            interp_rotation(rotations, mask, max_gap=5, method="cubic")
