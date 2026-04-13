"""線形空間補間ユーティリティ ``flare.utils.interp`` のテスト。

以下の観点でテストする:
    - 有効区間・ギャップ区間の列挙
    - 線形補間の数学的正しさ（手計算との一致）
    - PCHIP 補間のオーバーシュート抑制
    - max_gap 制限の動作
    - 先頭・末尾ギャップの非補間
    - 統計量算出と正規化の一貫性
"""

from __future__ import annotations

import numpy as np
import pytest

from flare.utils.interp import (
    _HAS_SCIPY,
    compute_stats,
    find_gap_runs,
    find_valid_runs,
    interp_linear,
    normalize,
    split_on_long_gaps,
)


class TestRunDetection:
    """有効区間・ギャップ区間検出のテスト。"""

    def test_all_valid(self) -> None:
        mask = np.array([True, True, True, True])
        assert find_valid_runs(mask) == [(0, 3)]
        assert find_gap_runs(mask) == []

    def test_all_gap(self) -> None:
        mask = np.array([False, False, False])
        assert find_valid_runs(mask) == []
        assert find_gap_runs(mask) == [(0, 2)]

    def test_single_gap_middle(self) -> None:
        mask = np.array([True, True, False, False, True, True])
        assert find_valid_runs(mask) == [(0, 1), (4, 5)]
        assert find_gap_runs(mask) == [(2, 3)]

    def test_leading_gap(self) -> None:
        mask = np.array([False, False, True, True, True])
        assert find_gap_runs(mask) == [(0, 1)]
        assert find_valid_runs(mask) == [(2, 4)]

    def test_trailing_gap(self) -> None:
        mask = np.array([True, True, True, False])
        assert find_gap_runs(mask) == [(3, 3)]
        assert find_valid_runs(mask) == [(0, 2)]

    def test_multiple_gaps(self) -> None:
        mask = np.array([True, False, True, False, False, True, True])
        assert find_gap_runs(mask) == [(1, 1), (3, 4)]
        assert find_valid_runs(mask) == [(0, 0), (2, 2), (5, 6)]

    def test_empty_mask(self) -> None:
        mask = np.array([], dtype=bool)
        assert find_valid_runs(mask) == []
        assert find_gap_runs(mask) == []


class TestInterpLinear:
    """線形補間の数学的正しさと境界条件のテスト。"""

    def test_no_gap_returns_input(self) -> None:
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        mask = np.array([True, True, True])
        filled, new_mask = interp_linear(values, mask, max_gap=5)
        np.testing.assert_array_equal(filled, values)
        np.testing.assert_array_equal(new_mask, mask)

    def test_single_gap_middle(self) -> None:
        # 0 と 10 の中間は 5
        values = np.array([[0.0], [99.0], [10.0]], dtype=np.float32)
        mask = np.array([True, False, True])
        filled, new_mask = interp_linear(values, mask, max_gap=5)
        assert filled[1, 0] == pytest.approx(5.0)
        assert new_mask[1]

    def test_multi_frame_gap(self) -> None:
        # 両端 0 / 12、3 フレームギャップ → [3, 6, 9]
        values = np.zeros((5, 1), dtype=np.float32)
        values[0, 0] = 0.0
        values[4, 0] = 12.0
        mask = np.array([True, False, False, False, True])
        filled, new_mask = interp_linear(values, mask, max_gap=5)
        assert filled[1, 0] == pytest.approx(3.0)
        assert filled[2, 0] == pytest.approx(6.0)
        assert filled[3, 0] == pytest.approx(9.0)
        assert np.all(new_mask)

    def test_long_gap_not_interpolated(self) -> None:
        values = np.zeros((10, 1), dtype=np.float32)
        values[0, 0] = 0.0
        values[9, 0] = 9.0
        mask = np.array([True] + [False] * 8 + [True])
        filled, new_mask = interp_linear(values, mask, max_gap=5)
        # max_gap=5 < 8 なので補間されない
        assert not new_mask[1:9].any()

    def test_leading_gap_not_interpolated(self) -> None:
        values = np.zeros((5, 1), dtype=np.float32)
        values[3, 0] = 3.0
        values[4, 0] = 4.0
        mask = np.array([False, False, False, True, True])
        filled, new_mask = interp_linear(values, mask, max_gap=10)
        assert not new_mask[0:3].any()

    def test_trailing_gap_not_interpolated(self) -> None:
        values = np.zeros((5, 1), dtype=np.float32)
        values[0, 0] = 1.0
        values[1, 0] = 2.0
        mask = np.array([True, True, False, False, False])
        filled, new_mask = interp_linear(values, mask, max_gap=10)
        assert not new_mask[2:5].any()

    def test_multidim_features(self) -> None:
        # 50 次元の PCA 係数を想定
        values = np.zeros((5, 50), dtype=np.float32)
        values[0] = 0.0
        values[4] = 1.0
        mask = np.array([True, False, False, False, True])
        filled, _ = interp_linear(values, mask, max_gap=5)
        assert filled[1] == pytest.approx(np.full(50, 0.25), rel=1e-5)
        assert filled[2] == pytest.approx(np.full(50, 0.5), rel=1e-5)
        assert filled[3] == pytest.approx(np.full(50, 0.75), rel=1e-5)

    def test_1d_input(self) -> None:
        values = np.array([0.0, 99.0, 10.0], dtype=np.float32)
        mask = np.array([True, False, True])
        filled, _ = interp_linear(values, mask, max_gap=5)
        assert filled.ndim == 1
        assert filled[1] == pytest.approx(5.0)

    def test_insufficient_valid_returns_input(self) -> None:
        values = np.array([[1.0], [0.0], [0.0]], dtype=np.float32)
        mask = np.array([True, False, False])
        filled, new_mask = interp_linear(values, mask, max_gap=5)
        # 有効点が 1 つだけなので補間せず返す
        np.testing.assert_array_equal(new_mask, mask)

    def test_length_mismatch_raises(self) -> None:
        values = np.zeros((5, 3), dtype=np.float32)
        mask = np.array([True, False])
        with pytest.raises(ValueError, match="does not match"):
            interp_linear(values, mask, max_gap=5)

    def test_unknown_order_raises(self) -> None:
        values = np.zeros((3, 1), dtype=np.float32)
        mask = np.array([True, False, True])
        with pytest.raises(ValueError, match="Unknown interpolation order"):
            interp_linear(values, mask, max_gap=5, order="quadratic")


@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")
class TestInterpPCHIP:
    """PCHIP 補間のテスト（scipy がある場合のみ）。"""

    def test_monotonic_preserved(self) -> None:
        # 単調増加データで PCHIP が単調性を保つことを確認
        t_valid = np.array([0, 1, 2, 6, 7, 8])
        y_valid = np.array([[0.0], [1.0], [2.0], [6.0], [7.0], [8.0]], dtype=np.float32)
        values = np.zeros((9, 1), dtype=np.float32)
        mask = np.zeros(9, dtype=bool)
        for i, idx in enumerate(t_valid):
            values[idx] = y_valid[i]
            mask[idx] = True
        filled, _ = interp_linear(values, mask, max_gap=10, order="pchip")
        # 補間された区間（3-5）は単調増加
        assert filled[3, 0] < filled[4, 0] < filled[5, 0]
        assert 2.0 < filled[3, 0] < 6.0

    def test_no_overshoot_on_noisy_data(self) -> None:
        # ノイズを含む凹凸データで PCHIP がオーバーシュートしないことを確認
        values = np.array(
            [[1.0], [1.05], [0.0], [0.0], [0.0], [1.05], [1.0]], dtype=np.float32
        )
        mask = np.array([True, True, False, False, False, True, True])
        filled, _ = interp_linear(values, mask, max_gap=5, order="pchip")
        # 補間値は 1.05 を超えてはいけない（単調性保存）
        assert filled[2:5, 0].max() <= 1.05 + 1e-5


class TestSplitOnLongGaps:
    """split_on_long_gaps のテスト。"""

    def test_no_split(self) -> None:
        mask = np.ones(10, dtype=bool)
        assert split_on_long_gaps(mask) == [(0, 9)]

    def test_single_split(self) -> None:
        mask = np.array([True] * 5 + [False] * 3 + [True] * 4)
        assert split_on_long_gaps(mask) == [(0, 4), (8, 11)]

    def test_min_length_filter(self) -> None:
        mask = np.array([True] * 3 + [False] + [True] * 5)
        # 最初の 3 フレームを min_length=5 で破棄
        assert split_on_long_gaps(mask, min_length=5) == [(4, 8)]

    def test_all_filtered_out(self) -> None:
        mask = np.array([True, True, False, True])
        assert split_on_long_gaps(mask, min_length=10) == []


class TestComputeStats:
    """統計量算出のテスト。"""

    def test_basic_stats(self) -> None:
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        mean, std = compute_stats(values)
        np.testing.assert_allclose(mean, [3.0, 4.0])
        np.testing.assert_allclose(std, np.std(values, axis=0), rtol=1e-5)

    def test_with_mask(self) -> None:
        values = np.array(
            [[1.0], [999.0], [3.0], [5.0]], dtype=np.float32
        )
        mask = np.array([True, False, True, True])
        mean, std = compute_stats(values, mask)
        # 有効値 [1, 3, 5] の平均 = 3
        assert mean[0] == pytest.approx(3.0)

    def test_std_clipped_to_min(self) -> None:
        values = np.ones((5, 3), dtype=np.float32)
        mean, std = compute_stats(values)
        assert std.min() >= 1e-6

    def test_empty_returns_zero_one(self) -> None:
        values = np.zeros((3, 2), dtype=np.float32)
        mask = np.array([False, False, False])
        mean, std = compute_stats(values, mask)
        np.testing.assert_array_equal(mean, np.zeros(2))
        np.testing.assert_array_equal(std, np.ones(2))


class TestNormalize:
    """正規化関数のテスト。"""

    def test_round_trip(self) -> None:
        values = np.random.default_rng(42).standard_normal((100, 5)).astype(np.float32)
        mean, std = compute_stats(values)
        normed = normalize(values, mean, std)
        assert normed.mean(axis=0) == pytest.approx(np.zeros(5), abs=1e-5)
        assert normed.std(axis=0) == pytest.approx(np.ones(5), rel=1e-5)

    def test_dtype_preserved(self) -> None:
        values = np.ones((10, 3), dtype=np.float32) * 5
        mean = np.full(3, 5.0, dtype=np.float32)
        std = np.full(3, 1.0, dtype=np.float32)
        normed = normalize(values, mean, std)
        assert normed.dtype == np.float32
