"""FLAME→PIRender パラメータ変換モジュール。

FLAMEパラメータ（expr/jaw_pose/rotation）をPIRenderが受け取るBFM形式
（exp/pose/trans）に変換するアダプタ。Route Bで生成されたFLAMEパラメータを
Route AのPIRenderに入力するクロスルート変換に使用する。

変換ロジック:
    1. expr (B, N) → exp (B, 64):
       FLAME expression空間からBFM expression空間への射影。
       線形変換行列が利用不可の場合は、先頭64Dを切り出す近似変換を行う。
    2. jaw_pose rotation_6d (B, 6) → pose[:, :3] axis-angle (B, 3):
       rotation_6dからaxis-angle表現への逆変換。
    3. rotation (B, 3) → pose[:, 3:6] (B, 3):
       global rotationをそのまま使用。
    4. trans: translationパラメータ。デフォルトはゼロベクトル (B, 3)。

pytorch3dが利用可能な場合はpytorch3d.transformsを使用し、
不可の場合は同等の純PyTorch実装にフォールバックする。

Example:
    FLAME→PIRender変換::

        adapter = FlameToPIRenderAdapter()
        pirender_params = adapter.convert({
            "expr": torch.randn(1, 100),
            "jaw_pose": torch.randn(1, 6),
            "rotation": torch.randn(1, 3),
        })
        # pirender_params["exp"].shape == (1, 64)
        # pirender_params["pose"].shape == (1, 6)
        # pirender_params["trans"].shape == (1, 3)
"""

from __future__ import annotations

import torch

from flare.converters.base import BaseAdapter

try:
    from pytorch3d.transforms import (  # type: ignore[import-untyped]
        matrix_to_axis_angle,
        rotation_6d_to_matrix,
    )

    _HAS_PYTORCH3D = True
except ImportError:
    _HAS_PYTORCH3D = False


def _rotation_6d_to_matrix_fallback(rotation_6d: torch.Tensor) -> torch.Tensor:
    """rotation 6D表現を回転行列に変換する（純PyTorchフォールバック）。

    pytorch3d.transforms.rotation_6d_to_matrix が利用不可の場合に使用する
    同等の純PyTorch実装。Gram-Schmidt直交化プロセスに基づく。

    Args:
        rotation_6d: rotation 6D表現テンソル。形状は ``(B, 6)``。

    Returns:
        回転行列テンソル。形状は ``(B, 3, 3)``。
    """
    a1 = rotation_6d[..., :3]
    a2 = rotation_6d[..., 3:6]

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def _matrix_to_axis_angle_fallback(matrix: torch.Tensor) -> torch.Tensor:
    """回転行列をaxis-angle表現に変換する（純PyTorchフォールバック）。

    pytorch3d.transforms.matrix_to_axis_angle が利用不可の場合に使用する
    同等の純PyTorch実装。

    Args:
        matrix: 回転行列テンソル。形状は ``(B, 3, 3)``。

    Returns:
        axis-angle表現テンソル。形状は ``(B, 3)``。
    """
    batch_shape = matrix.shape[:-2]

    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cos_angle = (trace - 1.0) * 0.5
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)

    axis = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )

    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    safe_norm = torch.clamp(axis_norm, min=1e-8)
    axis = axis / safe_norm

    result = axis * angle.unsqueeze(-1)

    near_zero = (angle.abs() < 1e-6).unsqueeze(-1)
    result = torch.where(
        near_zero,
        torch.zeros(*batch_shape, 3, device=matrix.device, dtype=matrix.dtype),
        result,
    )

    return result


def _6d_to_mat(rotation_6d: torch.Tensor) -> torch.Tensor:
    """rotation 6D → 回転行列変換のディスパッチャ。

    pytorch3dが利用可能ならpytorch3dを、不可なら純PyTorch実装を使用する。

    Args:
        rotation_6d: rotation 6D表現テンソル。形状は ``(B, 6)``。

    Returns:
        回転行列テンソル。形状は ``(B, 3, 3)``。
    """
    if _HAS_PYTORCH3D:
        return rotation_6d_to_matrix(rotation_6d)
    return _rotation_6d_to_matrix_fallback(rotation_6d)


def _mat_to_aa(matrix: torch.Tensor) -> torch.Tensor:
    """回転行列 → axis-angle変換のディスパッチャ。

    pytorch3dが利用可能ならpytorch3dを、不可なら純PyTorch実装を使用する。

    Args:
        matrix: 回転行列テンソル。形状は ``(B, 3, 3)``。

    Returns:
        axis-angle表現テンソル。形状は ``(B, 3)``。
    """
    if _HAS_PYTORCH3D:
        return matrix_to_axis_angle(matrix)
    return _matrix_to_axis_angle_fallback(matrix)


class FlameToPIRenderAdapter(BaseAdapter):
    """FLAME → PIRender (BFM形式) 変換アダプタ。

    FLAMEパラメータをPIRenderが受け取るBFM形式に変換する。

    変換ロジック:
        1. ``expr`` (B, N) → ``exp`` (B, 64):
           先頭64Dを切り出す近似変換。
        2. ``jaw_pose`` rotation_6d (B, 6) → axis-angle (B, 3):
           6D回転表現からaxis-angleへの逆変換。
        3. ``rotation`` (B, 3) → global rotation (B, 3):
           そのまま使用。
        4. ``pose`` = cat(rotation, jaw_aa) → (B, 6)
        5. ``trans``: translation。入力にない場合はゼロベクトル (B, 3)。
    """

    def convert(
        self, source_params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """FLAMEパラメータをPIRender BFM形式に変換する。

        Args:
            source_params: FLAMEパラメータの辞書。必須キー:
                - ``"expr"``: 表情パラメータ (B, N)。N >= 64。
                - ``"jaw_pose"``: 顎回転 rotation_6d (B, 6)。
                - ``"rotation"``: グローバル回転 axis-angle (B, 3)。
                オプションキー:
                - ``"trans"``: 平行移動 (B, 3)。省略時はゼロ。

        Returns:
            PIRender BFM形式のパラメータ辞書:
                - ``"exp"``: (B, 64) BFM表情係数
                - ``"pose"``: (B, 6) 姿勢（rotation 3D + jaw 3D）
                - ``"trans"``: (B, 3) 平行移動

        Raises:
            KeyError: source_paramsに必要なキーが存在しない場合。
        """
        expr = source_params["expr"]
        jaw_pose_6d = source_params["jaw_pose"]
        rotation = source_params["rotation"]

        batch_size = expr.shape[0]
        device = expr.device
        dtype = expr.dtype

        # expr (B, N) → exp (B, 64): 先頭64Dを切り出し
        if expr.shape[-1] >= 64:
            exp_64d = expr[:, :64]
        else:
            exp_64d = torch.nn.functional.pad(
                expr, (0, 64 - expr.shape[-1]), value=0.0
            )

        # jaw_pose rotation_6d (B, 6) → axis-angle (B, 3)
        jaw_mat = _6d_to_mat(jaw_pose_6d)
        jaw_aa = _mat_to_aa(jaw_mat)

        # pose = cat(rotation, jaw_aa) → (B, 6)
        pose = torch.cat([rotation, jaw_aa], dim=-1)

        # trans: 入力にあればそれを使用、なければゼロ
        if "trans" in source_params:
            trans = source_params["trans"]
        else:
            trans = torch.zeros(batch_size, 3, device=device, dtype=dtype)

        return {
            "exp": exp_64d,
            "pose": pose,
            "trans": trans,
        }

    @property
    def source_format(self) -> str:
        """変換元のパラメータ形式名を返す。

        Returns:
            ``"flame"``。
        """
        return "flame"

    @property
    def target_format(self) -> str:
        """変換先のパラメータ形式名を返す。

        Returns:
            ``"pirender"``。
        """
        return "pirender"
