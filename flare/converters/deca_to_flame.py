"""DECA→FlashAvatar パラメータ変換モジュール。

DECA出力（exp 50D + pose 6D）をFlashAvatarのcondition vector（120D）に変換する。
仕様書§8.2のDECAToFlameAdapter仕様に基づき、以下の変換を行う:

    - exp 50D → expr 100D: 同一FLAME PCA空間のためゼロパディング
    - jaw_pose axis-angle(3D) → rotation_6d(6D): pytorch3d変換
    - eyes_pose: 単位回転行列の6D表現 × 2 (12D) （DECA本来の挙動）
    - eyelids: ゼロ埋め (2D) （DECA本来の挙動）

DECA本来の挙動と MediaPipe 補完オプション:
    DECAは眼球ポーズ・瞼パラメータを出力しない。本Adapterのデフォルト挙動は
    DECA本来の仕様に従い、eyes_pose は単位回転、eyelids はゼロを返す。

    `use_mediapipe_supplement=True` を指定すると、呼び出し側が事前に推定した
    `eyes_pose` / `eyelids` (例: MediaPipe Face Landmarker から得られる値)
    を `source_params` 経由で注入できるようになる。注入された値は本Adapter
    のデフォルト値より優先される。フラグが False の場合は、`source_params` に
    `eyes_pose` / `eyelids` が含まれていても無視される (DECA本来挙動を厳守)。

重要な発見（v2.0）:
    DECA exp 50DとFlashAvatar expr 100Dは同一のFLAME expression PCA空間に
    属する。両者は同一のgeneric_model.pklからshapedirs[:,:,300:300+n_exp]を
    スライスしており、DECAはn_exp=50、FlashAvatarはn_exp=100を使用する。

変換にはpytorch3d.transformsのaxis_angle_to_matrix / matrix_to_rotation_6dを
使用する。pytorch3dが利用不可の場合は同等の純PyTorch実装にフォールバックする。

Example:
    DECA本来挙動 (デフォルト)::

        adapter = DECAToFlameAdapter()
        flash_params = adapter.convert({
            "exp": torch.randn(1, 50),
            "pose": torch.randn(1, 6),
        })
        # eyes_pose は identity_6d × 2、 eyelids は ゼロ

    MediaPipe 補完を有効化::

        adapter = DECAToFlameAdapter(use_mediapipe_supplement=True)
        eyes_pose, eyelids = face_detector.detect_eye_pose(frame, bbox)
        flash_params = adapter.convert({
            "exp": exp,
            "pose": pose,
            "eyes_pose": eyes_pose,   # MediaPipe由来の値を注入
            "eyelids": eyelids,
        })
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flare.converters.base import BaseAdapter

try:
    from pytorch3d.transforms import (  # type: ignore[import-untyped]
        axis_angle_to_matrix,
        matrix_to_rotation_6d,
    )

    _HAS_PYTORCH3D = True
except ImportError:
    _HAS_PYTORCH3D = False


def _axis_angle_to_matrix_fallback(axis_angle: torch.Tensor) -> torch.Tensor:
    """axis-angle表現を回転行列に変換する（純PyTorchフォールバック）。

    pytorch3d.transforms.axis_angle_to_matrix が利用不可の場合に使用する
    同等の純PyTorch実装。Rodrigues' rotation formulaに基づく。

    Args:
        axis_angle: axis-angle表現テンソル。形状は ``(B, 3)``。

    Returns:
        回転行列テンソル。形状は ``(B, 3, 3)``。
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True).unsqueeze(-1)
    safe_angle = torch.clamp(angle, min=1e-8)
    axis = axis_angle.unsqueeze(-1) / safe_angle.squeeze(-1).unsqueeze(-1)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    K = torch.zeros(
        *axis_angle.shape[:-1], 3, 3, device=axis_angle.device, dtype=axis_angle.dtype
    )
    x = axis[..., 0, 0]
    y = axis[..., 1, 0]
    z = axis[..., 2, 0]

    K[..., 0, 1] = -z
    K[..., 0, 2] = y
    K[..., 1, 0] = z
    K[..., 1, 2] = -x
    K[..., 2, 0] = -y
    K[..., 2, 1] = x

    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.expand(*axis_angle.shape[:-1], 3, 3)

    R = eye + sin_a * K + (1.0 - cos_a) * (K @ K)

    near_zero = (
        (angle.squeeze(-1).squeeze(-1) < 1e-8).unsqueeze(-1).unsqueeze(-1)
    )
    R = torch.where(near_zero, eye, R)

    return R


def _matrix_to_rotation_6d_fallback(matrix: torch.Tensor) -> torch.Tensor:
    """回転行列をrotation 6D表現に変換する（純PyTorchフォールバック）。

    pytorch3d.transforms.matrix_to_rotation_6d が利用不可の場合に使用する
    同等の純PyTorch実装。回転行列の最初の2列を取り出してフラット化する。

    Args:
        matrix: 回転行列テンソル。形状は ``(B, 3, 3)``。

    Returns:
        rotation 6D表現テンソル。形状は ``(B, 6)``。
    """
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def _aa_to_mat(axis_angle: torch.Tensor) -> torch.Tensor:
    """axis-angle → 回転行列変換のディスパッチャ。

    pytorch3dが利用可能ならpytorch3dを、不可なら純PyTorch実装を使用する。

    Args:
        axis_angle: axis-angle表現テンソル。形状は ``(B, 3)``。

    Returns:
        回転行列テンソル。形状は ``(B, 3, 3)``。
    """
    if _HAS_PYTORCH3D:
        return axis_angle_to_matrix(axis_angle)
    return _axis_angle_to_matrix_fallback(axis_angle)


def _mat_to_6d(matrix: torch.Tensor) -> torch.Tensor:
    """回転行列 → rotation 6D変換のディスパッチャ。

    pytorch3dが利用可能ならpytorch3dを、不可なら純PyTorch実装を使用する。

    Args:
        matrix: 回転行列テンソル。形状は ``(B, 3, 3)``。

    Returns:
        rotation 6D表現テンソル。形状は ``(B, 6)``。
    """
    if _HAS_PYTORCH3D:
        return matrix_to_rotation_6d(matrix)
    return _matrix_to_rotation_6d_fallback(matrix)


class DECAToFlameAdapter(BaseAdapter):
    """DECA (50D exp) → FlashAvatar (100D expr) 変換アダプタ。

    同一FLAME PCA空間に基づくゼロパディング変換を行う。
    線形マッピング行列や事前学習データは不要。

    変換ロジック（仕様書§8.2準拠）:
        1. ``exp`` (B, 50) → ``F.pad(exp, (0, 50), value=0.0)`` → ``expr`` (B, 100)
        2. ``pose[:, 3:6]`` axis-angle (B, 3)
           → ``axis_angle_to_matrix`` → (B, 3, 3)
           → ``matrix_to_rotation_6d`` → ``jaw_pose`` (B, 6)
        3. ``eyes_pose``: デフォルトは単位回転行列の6D表現 × 2 (B, 12)。
           ``use_mediapipe_supplement=True`` のとき、``source_params["eyes_pose"]``
           が指定されていればそちらを優先使用する。
        4. ``eyelids``: デフォルトは ``torch.zeros(B, 2)``。
           ``use_mediapipe_supplement=True`` のとき、``source_params["eyelids"]``
           が指定されていればそちらを優先使用する。

    pytorch3dが利用可能な場合はpytorch3d.transformsを使用し、
    不可の場合は同等の純PyTorch実装にフォールバックする。

    Attributes:
        _use_mediapipe_supplement: MediaPipe由来の eyes_pose / eyelids
            の注入を許可するか。DECA本来は出力しないパラメータのため、
            デフォルトは ``False``。
    """

    def __init__(self, use_mediapipe_supplement: bool = False) -> None:
        """DECAToFlameAdapterを初期化する。

        Args:
            use_mediapipe_supplement: MediaPipe等で外部推定した
                ``eyes_pose`` / ``eyelids`` の注入を許可するかどうか。
                ``False`` (デフォルト): DECA本来挙動 (eyes_pose=identity, eyelids=0)。
                  ``source_params`` に ``eyes_pose`` / ``eyelids`` が含まれていても無視する。
                ``True``: ``source_params["eyes_pose"]`` / ``source_params["eyelids"]``
                  が存在すれば優先使用し、無ければデフォルト値にフォールバックする。
        """
        self._use_mediapipe_supplement = use_mediapipe_supplement

    def convert(
        self, source_params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """DECAパラメータをFlashAvatar形式に変換する。

        Args:
            source_params: DECA出力の辞書。必須キー:
                - ``"exp"``: 表情パラメータ (B, 50)
                - ``"pose"``: 姿勢パラメータ (B, 6)。
                  global_rotation(3D) + jaw_pose(3D)。

                オプションキー (``use_mediapipe_supplement=True`` のとき有効):
                - ``"eyes_pose"``: 外部推定の眼球回転 (B, 12)。
                - ``"eyelids"``: 外部推定の瞼パラメータ (B, 2)。

        Returns:
            FlashAvatar形式のパラメータ辞書:
                - ``"expr"``: (B, 100) FLAME表情パラメータ
                - ``"jaw_pose"``: (B, 6) 顎回転 (rotation_6d)
                - ``"eyes_pose"``: (B, 12) 眼球回転
                - ``"eyelids"``: (B, 2) 瞼パラメータ

        Raises:
            KeyError: source_paramsに必要なキーが存在しない場合。
        """
        exp_50d = source_params["exp"]
        pose = source_params["pose"]
        batch_size = exp_50d.shape[0]
        device = exp_50d.device
        dtype = exp_50d.dtype

        # exp 50D → expr 100D: ゼロパディング
        expr_100d = F.pad(exp_50d, (0, 50), value=0.0)

        # jaw_pose: axis-angle(3D) → rotation_6d(6D)
        jaw_aa = pose[:, 3:6]
        jaw_mat = _aa_to_mat(jaw_aa)
        jaw_6d = _mat_to_6d(jaw_mat)

        eyes_pose = self._resolve_eyes_pose(source_params, batch_size, device, dtype)
        eyelids = self._resolve_eyelids(source_params, batch_size, device, dtype)

        return {
            "expr": expr_100d,
            "jaw_pose": jaw_6d,
            "eyes_pose": eyes_pose,
            "eyelids": eyelids,
        }

    def _resolve_eyes_pose(
        self,
        source_params: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """eyes_pose を解決する。フラグONかつ source 提供あれば採用、それ以外は単位回転×2。"""
        if (
            self._use_mediapipe_supplement
            and "eyes_pose" in source_params
            and source_params["eyes_pose"] is not None
        ):
            return source_params["eyes_pose"].to(device=device, dtype=dtype)

        identity_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        identity_6d = _mat_to_6d(identity_mat)
        return identity_6d.repeat(batch_size, 1).repeat(1, 2)

    def _resolve_eyelids(
        self,
        source_params: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """eyelids を解決する。フラグONかつ source 提供あれば採用、それ以外はゼロ。"""
        if (
            self._use_mediapipe_supplement
            and "eyelids" in source_params
            and source_params["eyelids"] is not None
        ):
            return source_params["eyelids"].to(device=device, dtype=dtype)

        return torch.zeros(batch_size, 2, device=device, dtype=dtype)

    @property
    def use_mediapipe_supplement(self) -> bool:
        """MediaPipe補完が有効かどうかを返す。"""
        return self._use_mediapipe_supplement

    @property
    def source_format(self) -> str:
        """変換元のパラメータ形式名を返す。

        Returns:
            ``"deca"``。
        """
        return "deca"

    @property
    def target_format(self) -> str:
        """変換先のパラメータ形式名を返す。

        Returns:
            ``"flash_avatar"``。
        """
        return "flash_avatar"
