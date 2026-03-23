"""DECA→FlashAvatar パラメータ変換アダプタ。

仕様書4.3節・5.2節・8.2節に基づき、DECA出力（exp 50D）を
FlashAvatar入力（condition 120D）に変換する。

重要な発見（v2.0新規）:
    DECA exp 50DとFlashAvatar expr 100Dは同一のFLAME expression PCA空間に属する。
    両者は同一のgeneric_model.pklからshapedirs[:,:,300:300+n_exp]をスライスしており、
    DECAはn_exp=50（第1-50主成分）、FlashAvatarはn_exp=100（第1-100主成分）を使用する。

変換ロジック:
    1. exp 50D → expr 100D: ゼロパディング（同一PCA空間のため正確）
    2. jaw_pose: axis-angle 3D → rotation_6d 6D（pytorch3d.transforms使用）
    3. eyes_pose: 単位回転行列の6D表現 × 2 = 12D（Phase 1デフォルト）
    4. eyelids: ゼロ埋め 2D（Phase 1デフォルト）

FlashAvatar condition vector 120D:
    expr(100D) + jaw_pose(6D) + eyes_pose(12D) + eyelids(2D) = 120D

Example:
    >>> from flare.converters.deca_to_flame import DECAToFlameAdapter
    >>> adapter = DECAToFlameAdapter()
    >>> flame_params = adapter.convert(deca_output)
    >>> flame_params["expr"].shape  # (B, 100)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from loguru import logger

from flare.converters.base import BaseAdapter
from flare.converters.registry import AdapterRegistry

# pytorch3dのインポート（ソースビルド版 0.7.8）
# axis_angle_to_matrix / matrix_to_rotation_6d は純PyTorchテンソル演算であり、
# C++/CUDA拡張に依存しないため互換性は保証される。
try:
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

    _PYTORCH3D_AVAILABLE: bool = True
except ImportError:
    _PYTORCH3D_AVAILABLE = False
    logger.debug(
        "pytorch3dが利用できません。純PyTorch実装にフォールバックします。"
    )


def _axis_angle_to_matrix_fallback(axis_angle: torch.Tensor) -> torch.Tensor:
    """axis-angle(3D)から回転行列(3x3)への変換（pytorch3dフォールバック）。

    Rodrigues' formulaによる実装。

    Args:
        axis_angle: axis-angleテンソル。shape ``(B, 3)``。

    Returns:
        回転行列テンソル。shape ``(B, 3, 3)``。
    """
    theta: torch.Tensor = axis_angle.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k: torch.Tensor = axis_angle / theta

    kx: torch.Tensor = k[:, 0]
    ky: torch.Tensor = k[:, 1]
    kz: torch.Tensor = k[:, 2]
    zero: torch.Tensor = torch.zeros_like(kx)

    K: torch.Tensor = torch.stack([
        torch.stack([zero, -kz, ky], dim=-1),
        torch.stack([kz, zero, -kx], dim=-1),
        torch.stack([-ky, kx, zero], dim=-1),
    ], dim=-2)

    sin_t: torch.Tensor = theta.sin().unsqueeze(-1)
    cos_t: torch.Tensor = theta.cos().unsqueeze(-1)
    eye: torch.Tensor = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0)

    R: torch.Tensor = eye + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R


def _matrix_to_rotation_6d_fallback(matrix: torch.Tensor) -> torch.Tensor:
    """回転行列(3x3)から6D rotation表現への変換（pytorch3dフォールバック）。

    pytorch3dと同じ規約: 回転行列の最初の2行を取り出してflattenする。
    ``R[..., :2, :].reshape(..., 6)`` → ``[r00, r01, r02, r10, r11, r12]``。

    Args:
        matrix: 回転行列テンソル。shape ``(B, 3, 3)``。

    Returns:
        6D rotationテンソル。shape ``(B, 6)``。
    """
    return matrix[:, :2, :].reshape(-1, 6)


# 使用する関数の選択
if _PYTORCH3D_AVAILABLE:
    _aa_to_mat = axis_angle_to_matrix
    _mat_to_6d = matrix_to_rotation_6d
else:
    _aa_to_mat = _axis_angle_to_matrix_fallback
    _mat_to_6d = _matrix_to_rotation_6d_fallback


# --- AdapterRegistryへの自動登録 ---
_registry: AdapterRegistry = AdapterRegistry.get_instance()


@_registry.register
class DECAToFlameAdapter(BaseAdapter):
    """DECA (50D exp) → FlashAvatar (120D condition) 変換アダプタ。

    同一FLAME PCA空間のゼロパディングにより、事前学習済みモデルや
    最小二乗法の学習データなしに正確な変換を実現する。

    変換内容:
        - exp 50D → expr 100D: ``F.pad(exp, (0, 50), value=0.0)``
        - jaw_pose: DECA pose[:, 3:6] (axis-angle 3D) → rotation_6d (6D)
        - eyes_pose: 単位回転行列の6D表現 × 2 = 12D
        - eyelids: ゼロ埋め 2D

    Example:
        >>> adapter = DECAToFlameAdapter()
        >>> result = adapter.convert({"exp": exp_50d, "pose": pose_6d})
        >>> result["expr"].shape  # (B, 100)
        >>> # condition = cat([expr, jaw_pose, eyes_pose, eyelids]) → (B, 120)
    """

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

    def convert(
        self, source_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """DECAパラメータをFlashAvatar condition形式に変換する。

        仕様書4.3節・8.2節の変換ロジックに厳密に従う。
        全出力テンソルはsource_paramsと同一デバイス・dtypeで生成される。

        Args:
            source_params: DECA出力Dict。
                必須キー: ``"exp"`` (B, 50), ``"pose"`` (B, 6)。
                pose構成: global_rotation(3D) + jaw_pose(3D)。

        Returns:
            FlashAvatar condition Dict:
                - ``"expr"``: (B, 100) — 表情パラメータ（ゼロパディング済み）
                - ``"jaw_pose"``: (B, 6) — 顎回転（6D rotation表現）
                - ``"eyes_pose"``: (B, 12) — 左右眼球回転（単位回転 × 2）
                - ``"eyelids"``: (B, 2) — 瞼パラメータ（ゼロ埋め）
        """
        exp_50d: torch.Tensor = source_params["exp"]  # (B, 50)
        pose: torch.Tensor = source_params["pose"]  # (B, 6)

        device: torch.device = exp_50d.device
        dtype: torch.dtype = exp_50d.dtype
        batch_size: int = exp_50d.shape[0]

        # 1. ゼロパディング: 50D → 100D
        expr_100d: torch.Tensor = F.pad(
            exp_50d, (0, 50), value=0.0
        )  # (B, 100)

        # 2. jaw_pose: axis-angle(3D) → rotation_6d(6D)
        jaw_aa: torch.Tensor = pose[:, 3:6]  # (B, 3)
        jaw_mat: torch.Tensor = _aa_to_mat(jaw_aa)  # (B, 3, 3)
        jaw_6d: torch.Tensor = _mat_to_6d(jaw_mat)  # (B, 6)

        # 3. eyes_pose: 単位回転行列の6D表現 × 2
        identity_mat: torch.Tensor = torch.eye(
            3, device=device, dtype=dtype
        ).unsqueeze(0)  # (1, 3, 3)
        I_6d: torch.Tensor = _mat_to_6d(identity_mat)  # (1, 6)
        eyes_single: torch.Tensor = I_6d.expand(batch_size, -1)  # (B, 6)
        eyes_pose: torch.Tensor = eyes_single.repeat(1, 2)  # (B, 12)

        # 4. eyelids: ゼロ埋め
        eyelids: torch.Tensor = torch.zeros(
            batch_size, 2, device=device, dtype=dtype
        )  # (B, 2)

        return {
            "expr": expr_100d,      # (B, 100)
            "jaw_pose": jaw_6d,     # (B, 6)
            "eyes_pose": eyes_pose, # (B, 12)
            "eyelids": eyelids,     # (B, 2)
        }
