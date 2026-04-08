"""DECAToFlameAdapter の変換ロジックテスト。

仕様書§8.2のゼロパディング変換を検証する:
    - exp 50D → expr 100D
    - jaw_pose axis-angle(3D) → rotation_6d(6D)
    - eyes_pose identity_6d × 2 (12D)
    - eyelids ゼロ埋め (2D)
"""

from __future__ import annotations

import torch

from flare.converters.deca_to_flame import DECAToFlameAdapter


class TestDECAToFlameShapes:
    """変換後テンソルの形状テスト。"""

    def test_expr_shape(self, dummy_deca_output: dict[str, torch.Tensor]) -> None:
        """exp 50D → expr 100D のパディング後の形状が (B, 100) であること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert result["expr"].shape == (2, 100)

    def test_jaw_pose_shape(
        self, dummy_deca_output: dict[str, torch.Tensor]
    ) -> None:
        """jaw_pose の rotation_6d 変換後の形状が (B, 6) であること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert result["jaw_pose"].shape == (2, 6)

    def test_eyes_pose_shape(
        self, dummy_deca_output: dict[str, torch.Tensor]
    ) -> None:
        """eyes_pose の形状が (B, 12) であること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert result["eyes_pose"].shape == (2, 12)

    def test_eyelids_shape(
        self, dummy_deca_output: dict[str, torch.Tensor]
    ) -> None:
        """eyelids の形状が (B, 2) であること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert result["eyelids"].shape == (2, 2)


class TestDECAToFlameValues:
    """変換後テンソルの値テスト。"""

    def test_expr_first_50_preserved(
        self, dummy_deca_output: dict[str, torch.Tensor]
    ) -> None:
        """expr の最初の 50D が元の exp と一致すること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert torch.allclose(
            result["expr"][:, :50], dummy_deca_output["exp"]
        )

    def test_expr_last_50_zero(
        self, dummy_deca_output: dict[str, torch.Tensor]
    ) -> None:
        """expr の最後の 50D がゼロであること（ゼロパディング）。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert torch.allclose(
            result["expr"][:, 50:], torch.zeros(2, 50)
        )

    def test_eyelids_zero(
        self, dummy_deca_output: dict[str, torch.Tensor]
    ) -> None:
        """eyelids が全てゼロであること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert torch.allclose(result["eyelids"], torch.zeros(2, 2))

    def test_eyes_pose_is_identity_6d_repeated(self) -> None:
        """eyes_pose が単位回転行列の6D表現 × 2 であること。"""
        adapter = DECAToFlameAdapter()
        source = {
            "exp": torch.zeros(1, 50),
            "pose": torch.zeros(1, 6),
        }
        result = adapter.convert(source)
        eyes = result["eyes_pose"]

        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        expected = identity_6d.repeat(1, 2)
        assert torch.allclose(eyes, expected, atol=1e-6)

    def test_jaw_pose_identity_for_zero_input(self) -> None:
        """jaw_pose がゼロ入力で単位回転行列の6D表現になること。"""
        adapter = DECAToFlameAdapter()
        source = {
            "exp": torch.zeros(1, 50),
            "pose": torch.zeros(1, 6),
        }
        result = adapter.convert(source)
        jaw = result["jaw_pose"]

        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        assert torch.allclose(jaw, identity_6d, atol=1e-6)


class TestDECAToFlameAdapterProperties:
    """Adapter プロパティテスト。"""

    def test_source_format(self) -> None:
        """source_format が 'deca' であること。"""
        adapter = DECAToFlameAdapter()
        assert adapter.source_format == "deca"

    def test_target_format(self) -> None:
        """target_format が 'flash_avatar' であること。"""
        adapter = DECAToFlameAdapter()
        assert adapter.target_format == "flash_avatar"

    def test_missing_key_raises(self) -> None:
        """必要なキーが欠損している場合にKeyErrorが発生すること。"""
        adapter = DECAToFlameAdapter()
        with pytest.raises(KeyError):
            adapter.convert({"exp": torch.zeros(1, 50)})


class TestDECAToFlameBatchSize:
    """異なるバッチサイズでの動作テスト。"""

    def test_batch_size_1(self) -> None:
        """バッチサイズ1で正しく動作すること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert({
            "exp": torch.randn(1, 50),
            "pose": torch.randn(1, 6),
        })
        assert result["expr"].shape == (1, 100)

    def test_batch_size_16(self) -> None:
        """バッチサイズ16で正しく動作すること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert({
            "exp": torch.randn(16, 50),
            "pose": torch.randn(16, 6),
        })
        assert result["expr"].shape == (16, 100)
        assert result["jaw_pose"].shape == (16, 6)
        assert result["eyes_pose"].shape == (16, 12)
        assert result["eyelids"].shape == (16, 2)


import pytest  # noqa: E402 (for TestDECAToFlameAdapterProperties.test_missing_key_raises)
