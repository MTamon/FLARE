"""DECAToFlameAdapter のゼロパディング変換テスト。

仕様書4.3節・8.2節に基づき、DECA exp 50D → FlashAvatar expr 100D の
ゼロパディング変換、jaw_poseのaxis-angle→6D rotation変換、
eyes_poseの単位回転行列6D表現、eyelidsのゼロ埋めを数値検証する。

FlashAvatar condition vector 120D構成:
    expr(100D) + jaw_pose(6D) + eyes_pose(12D) + eyelids(2D) = 120D
"""

from __future__ import annotations

from typing import Dict

import pytest
import torch
import torch.nn.functional as F

from flare.converters.base import BaseAdapter
from flare.converters.registry import AdapterRegistry


class DECAToFlameAdapter(BaseAdapter):
    """DECA (50D exp) → FlashAvatar (100D expr) 変換。

    仕様書8.2節の具体例に基づく。同一FLAME PCA空間のため
    ゼロパディングで正確に変換可能。

    jaw_pose: DECA axis-angle(3D) → rotation_6d(6D) に変換。
    eyes_pose: Phase 1は単位回転行列の6D表現 × 2。
    eyelids: Phase 1はゼロ埋め。
    """

    def convert(
        self, source_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """DECAパラメータをFlashAvatar形式に変換する。

        Args:
            source_params: DECA出力Dict。
                必須キー: ``"exp"`` (B, 50), ``"pose"`` (B, 6)。

        Returns:
            FlashAvatar condition Dict:
                expr (B, 100), jaw_pose (B, 6),
                eyes_pose (B, 12), eyelids (B, 2)。
        """
        exp_50d: torch.Tensor = source_params["exp"]  # (B, 50)
        pose: torch.Tensor = source_params["pose"]  # (B, 6)
        batch_size: int = exp_50d.shape[0]

        # ゼロパディング: 50D → 100D
        expr_100d: torch.Tensor = F.pad(exp_50d, (0, 50), value=0.0)  # (B, 100)

        # jaw_pose: axis-angle(3D) → rotation_6d(6D)
        jaw_aa: torch.Tensor = pose[:, 3:6]  # (B, 3)
        jaw_6d: torch.Tensor = self._axis_angle_to_rotation_6d(jaw_aa)  # (B, 6)

        # eyes_pose: 単位回転行列の6D表現 × 2
        identity_6d: torch.Tensor = self._identity_rotation_6d(batch_size)  # (B, 12)

        # eyelids: ゼロ埋め
        eyelids: torch.Tensor = torch.zeros(
            batch_size, 2, device=exp_50d.device
        )

        return {
            "expr": expr_100d,
            "jaw_pose": jaw_6d,
            "eyes_pose": identity_6d,
            "eyelids": eyelids,
        }

    @property
    def source_format(self) -> str:
        """変換元フォーマット名。

        Returns:
            ``"deca"``。
        """
        return "deca"

    @property
    def target_format(self) -> str:
        """変換先フォーマット名。

        Returns:
            ``"flash_avatar"``。
        """
        return "flash_avatar"

    @staticmethod
    def _axis_angle_to_rotation_6d(aa: torch.Tensor) -> torch.Tensor:
        """axis-angle(3D)をrotation_6d(6D)に変換する。

        Rodrigues' formulaで回転行列を計算し、最初の2列を取得して
        6D表現とする。

        Args:
            aa: axis-angle テンソル (B, 3)。

        Returns:
            6D rotation テンソル (B, 6)。
        """
        theta: torch.Tensor = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        k: torch.Tensor = aa / theta  # (B, 3) normalized axis

        # skew-symmetric matrix K
        kx: torch.Tensor = k[:, 0]
        ky: torch.Tensor = k[:, 1]
        kz: torch.Tensor = k[:, 2]
        zero: torch.Tensor = torch.zeros_like(kx)

        # K = [[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]]
        row0: torch.Tensor = torch.stack([zero, -kz, ky], dim=-1)
        row1: torch.Tensor = torch.stack([kz, zero, -kx], dim=-1)
        row2: torch.Tensor = torch.stack([-ky, kx, zero], dim=-1)
        K: torch.Tensor = torch.stack([row0, row1, row2], dim=-2)  # (B, 3, 3)

        # Rodrigues: R = I + sin(θ)K + (1-cos(θ))K²
        sin_t: torch.Tensor = theta.sin().unsqueeze(-1)  # (B, 1, 1)
        cos_t: torch.Tensor = theta.cos().unsqueeze(-1)  # (B, 1, 1)
        eye: torch.Tensor = torch.eye(3, device=aa.device).unsqueeze(0)  # (1, 3, 3)

        R: torch.Tensor = eye + sin_t * K + (1.0 - cos_t) * (K @ K)  # (B, 3, 3)

        # 6D representation: first two columns
        return R[:, :, :2].reshape(-1, 6)  # (B, 6)

    @staticmethod
    def _identity_rotation_6d(batch_size: int) -> torch.Tensor:
        """単位回転行列の6D表現を左右眼球分(×2)生成する。

        Args:
            batch_size: バッチサイズ。

        Returns:
            shape (B, 12) のテンソル。6D identity × 2。
        """
        # Identity matrix → first 2 columns → flatten
        # [[1,0], [0,1], [0,0]] → [1, 0, 0, 0, 1, 0]
        i6d: torch.Tensor = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32
        )
        single: torch.Tensor = i6d.unsqueeze(0).repeat(batch_size, 1)  # (B, 6)
        return torch.cat([single, single], dim=-1)  # (B, 12)


class TestDECAToFlameAdapter:
    """DECAToFlameAdapterのテストスイート。"""

    @pytest.fixture()
    def adapter(self) -> DECAToFlameAdapter:
        """DECAToFlameAdapterインスタンスを返す。

        Returns:
            DECAToFlameAdapter。
        """
        return DECAToFlameAdapter()

    @pytest.fixture()
    def deca_output(self) -> Dict[str, torch.Tensor]:
        """テスト用DECAパラメータを返す。

        Returns:
            exp (2, 50) と pose (2, 6) を含むDict。
        """
        torch.manual_seed(42)
        return {
            "exp": torch.randn(2, 50),
            "pose": torch.randn(2, 6),
        }

    def test_expr_zero_padding(
        self,
        adapter: DECAToFlameAdapter,
        deca_output: Dict[str, torch.Tensor],
    ) -> None:
        """exp (B=2, 50D) → expr (B=2, 100D) のゼロパディングを検証する。

        前半50次元が元のexp値と完全一致し、後半50次元がすべてゼロであること。

        Args:
            adapter: テスト対象のAdapter。
            deca_output: テスト用入力。
        """
        result: Dict[str, torch.Tensor] = adapter.convert(deca_output)
        expr: torch.Tensor = result["expr"]

        assert expr.shape == (2, 100)
        assert torch.allclose(expr[:, :50], deca_output["exp"])
        assert torch.all(expr[:, 50:] == 0.0)

    def test_jaw_pose_shape(
        self,
        adapter: DECAToFlameAdapter,
        deca_output: Dict[str, torch.Tensor],
    ) -> None:
        """jaw_pose変換後のshapeが (B, 6) であることを確認する。

        Args:
            adapter: テスト対象のAdapter。
            deca_output: テスト用入力。
        """
        result: Dict[str, torch.Tensor] = adapter.convert(deca_output)

        assert result["jaw_pose"].shape == (2, 6)

    def test_eyes_pose_identity(
        self,
        adapter: DECAToFlameAdapter,
        deca_output: Dict[str, torch.Tensor],
    ) -> None:
        """eyes_pose (B, 12) が単位回転行列の6D表現×2であることを確認する。

        単位回転行列の最初の2列: [[1,0],[0,1],[0,0]] → [1,0,0,0,1,0]

        Args:
            adapter: テスト対象のAdapter。
            deca_output: テスト用入力。
        """
        result: Dict[str, torch.Tensor] = adapter.convert(deca_output)
        eyes: torch.Tensor = result["eyes_pose"]

        assert eyes.shape == (2, 12)

        expected_6d: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        for b in range(2):
            assert torch.allclose(eyes[b, :6], expected_6d)
            assert torch.allclose(eyes[b, 6:], expected_6d)

    def test_eyelids_zeros(
        self,
        adapter: DECAToFlameAdapter,
        deca_output: Dict[str, torch.Tensor],
    ) -> None:
        """eyelids (B, 2) がすべてゼロであることを確認する。

        Args:
            adapter: テスト対象のAdapter。
            deca_output: テスト用入力。
        """
        result: Dict[str, torch.Tensor] = adapter.convert(deca_output)

        assert result["eyelids"].shape == (2, 2)
        assert torch.all(result["eyelids"] == 0.0)

    def test_condition_vector_dim_120(
        self,
        adapter: DECAToFlameAdapter,
        deca_output: Dict[str, torch.Tensor],
    ) -> None:
        """convert()出力をcatした場合に合計120次元になることを確認する。

        expr:100 + jaw_pose:6 + eyes_pose:12 + eyelids:2 = 120

        Args:
            adapter: テスト対象のAdapter。
            deca_output: テスト用入力。
        """
        result: Dict[str, torch.Tensor] = adapter.convert(deca_output)
        condition: torch.Tensor = torch.cat(
            [result["expr"], result["jaw_pose"], result["eyes_pose"], result["eyelids"]],
            dim=-1,
        )

        assert condition.shape == (2, 120)

    def test_source_target_format(self, adapter: DECAToFlameAdapter) -> None:
        """source_format=="deca"、target_format=="flash_avatar" であることを確認する。

        Args:
            adapter: テスト対象のAdapter。
        """
        assert adapter.source_format == "deca"
        assert adapter.target_format == "flash_avatar"

    def test_registry_registration(self) -> None:
        """AdapterRegistryにDECAToFlameAdapterが登録できることを確認する。"""
        AdapterRegistry.reset()
        registry: AdapterRegistry = AdapterRegistry.get_instance()
        registry.register(DECAToFlameAdapter)

        adapter: BaseAdapter = registry.get("deca", "flash_avatar")
        assert isinstance(adapter, DECAToFlameAdapter)
        assert adapter.source_format == "deca"

        assert "deca_to_flash_avatar" in registry.list_adapters()

        AdapterRegistry.reset()

    def test_callable_shortcut(
        self,
        adapter: DECAToFlameAdapter,
        deca_output: Dict[str, torch.Tensor],
    ) -> None:
        """__call__()がconvert()と同じ結果を返すことを確認する。

        Args:
            adapter: テスト対象のAdapter。
            deca_output: テスト用入力。
        """
        result_convert: Dict[str, torch.Tensor] = adapter.convert(deca_output)
        result_call: Dict[str, torch.Tensor] = adapter(deca_output)

        assert torch.allclose(result_convert["expr"], result_call["expr"])
        assert torch.allclose(result_convert["jaw_pose"], result_call["jaw_pose"])
