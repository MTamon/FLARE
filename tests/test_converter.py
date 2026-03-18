"""Converter テスト (Section 8.2 BaseAdapter + AdapterRegistry)

DECA → FlashAvatar ゼロパディング変換のテストひな型を含む。
本実装 (converters/deca_to_flame.py) は Phase 2 で追記予定。
ここではダミー Tensor で変換ロジックの正しさを検証する。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flare.converters.base import BaseAdapter
from flare.converters.registry import AdapterRegistry


class TestBaseAdapterABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseAdapter()  # type: ignore[abstract]


class TestDummyAdapter:
    def test_convert_passthrough(self, dummy_adapter):
        params = {"exp": torch.randn(2, 50), "pose": torch.randn(2, 6)}
        result = dummy_adapter.convert(params)
        assert torch.equal(result["exp"], params["exp"])

    def test_source_target_format(self, dummy_adapter):
        assert dummy_adapter.source_format == "dummy_src"
        assert dummy_adapter.target_format == "dummy_tgt"


class TestAdapterRegistry:
    def test_register_and_get(self):
        registry = AdapterRegistry()

        @registry.register("fmt_a", "fmt_b")
        class AToB(BaseAdapter):
            def convert(self, source_params):
                return source_params

            @property
            def source_format(self):
                return "fmt_a"

            @property
            def target_format(self):
                return "fmt_b"

        cls = registry.get("fmt_a", "fmt_b")
        assert cls is AToB

    def test_duplicate_registration_raises(self):
        registry = AdapterRegistry()

        @registry.register("x", "y")
        class X2Y(BaseAdapter):
            def convert(self, s):
                return s
            @property
            def source_format(self):
                return "x"
            @property
            def target_format(self):
                return "y"

        with pytest.raises(ValueError, match="already registered"):
            @registry.register("x", "y")
            class X2Y2(BaseAdapter):
                def convert(self, s):
                    return s
                @property
                def source_format(self):
                    return "x"
                @property
                def target_format(self):
                    return "y"

    def test_missing_adapter_raises(self):
        registry = AdapterRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent", "format")

    def test_identity_same_format(self, dummy_adapter):
        registry = AdapterRegistry()
        registry.set_identity_factory(lambda: dummy_adapter)
        cls = registry.get("anything", "anything")
        instance = cls()
        assert instance.source_format == "dummy_src"

    def test_available(self):
        registry = AdapterRegistry()
        registry.register_class("a", "b", type(DummyAdapterForTest()))
        assert ("a", "b") in registry.available()


# ---------------------------------------------------------------------------
# DECA → FlashAvatar ゼロパディング変換テスト (ひな型)
#
# Section 4.3 / 5.2:
#   DECA exp 50D と FlashAvatar expr 100D は同一 FLAME PCA 空間。
#   ゼロパディングで正確に変換可能。
#   jaw_pose: axis-angle(3D) → rotation_6d(6D)
#   eyes_pose: 単位回転行列の 6D 表現 x2 (12D)
#   eyelids: ゼロ埋め (2D)
#   → condition vector 合計 120D
#
# NOTE: converters/deca_to_flame.py の本実装は Phase 2 で追記する。
#       ここではダミー Tensor で変換ロジックの正しさを直接検証する。
# ---------------------------------------------------------------------------


class TestDECAToFlashAvatarZeroPadding:
    """DECA → FlashAvatar 変換のコアロジックをダミーで検証。"""

    def test_exp_zero_padding_50_to_100(self):
        """exp 50D → expr 100D: 後半 50 次元がゼロであること。"""
        B = 4
        exp_50d = torch.randn(B, 50)
        expr_100d = F.pad(exp_50d, (0, 50), value=0.0)

        assert expr_100d.shape == (B, 100)
        # 前半 50 次元は元の値を保持
        assert torch.allclose(expr_100d[:, :50], exp_50d)
        # 後半 50 次元はゼロ
        assert torch.all(expr_100d[:, 50:] == 0.0)

    def test_jaw_axis_angle_to_rotation_6d(self):
        """jaw_pose: axis-angle(3D) → rotation_6d(6D) 変換。

        pytorch3d が利用可能な場合のみ実行。Phase 2 で本実装と統合予定。
        """
        pytest.importorskip("pytorch3d")
        from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

        B = 4
        jaw_aa = torch.randn(B, 3)
        jaw_mat = axis_angle_to_matrix(jaw_aa)       # (B, 3, 3)
        jaw_6d = matrix_to_rotation_6d(jaw_mat)       # (B, 6)

        assert jaw_6d.shape == (B, 6)

    def test_eyes_pose_identity(self):
        """eyes_pose: 単位回転行列 6D x2 = 12D。"""
        pytest.importorskip("pytorch3d")
        from pytorch3d.transforms import matrix_to_rotation_6d

        B = 4
        I_6d = matrix_to_rotation_6d(torch.eye(3).unsqueeze(0))  # (1, 6)
        eyes_pose = I_6d.repeat(B, 1).repeat(1, 2)               # (B, 12)

        assert eyes_pose.shape == (B, 12)
        # 左右同一 (単位回転)
        assert torch.allclose(eyes_pose[:, :6], eyes_pose[:, 6:])

    def test_condition_vector_120d(self):
        """最終 condition vector が 120D になること。"""
        B = 2
        expr = torch.randn(B, 100)
        jaw_pose = torch.randn(B, 6)
        eyes_pose = torch.randn(B, 12)
        eyelids = torch.zeros(B, 2)

        condition = torch.cat([expr, jaw_pose, eyes_pose, eyelids], dim=1)
        assert condition.shape == (B, 120)


# テスト用ヘルパー
class DummyAdapterForTest(BaseAdapter):
    def convert(self, source_params):
        return source_params

    @property
    def source_format(self):
        return "test_src"

    @property
    def target_format(self):
        return "test_tgt"