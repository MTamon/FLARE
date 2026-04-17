"""SMIRK→FlashAvatar パラメータ変換モジュール。

SMIRK出力（exp 50D + pose 6D + eyelid 2D）をFlashAvatarの
condition vector（120D）に変換する。

DECAToFlameAdapterとの違い:
    - ``eyelid`` (2D) を直接 ``eyelids`` に渡す。
      DECAは常にゼロ埋めだが、SMIRKは実際の瞼開閉量を推定するため
      目の開閉のダイナミクスが正しく FlashAvatar に伝わる。

変換ロジック（DECAToFlameAdapter と同じ変換規則を共有）:
    1. ``exp`` (B, 50) → ``F.pad(exp, (0, 50))`` → ``expr`` (B, 100)
    2. ``pose[:, 3:6]`` axis-angle → rotation_6d → ``jaw_pose`` (B, 6)
    3. ``eyes_pose``: 単位回転行列6D × 2 → (B, 12)（Phase 1デフォルト）
    4. ``eyelids``: SMIRKの ``eyelid`` をそのまま使用 (B, 2)

注意: eyes_pose は flame-head-tracker 統合後に実際の眼球回転に置き換え可能。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flare.converters.base import BaseAdapter
from flare.converters.deca_to_flame import _aa_to_mat, _mat_to_6d


class SMIRKToFlameAdapter(BaseAdapter):
    """SMIRK (50D exp + eyelid 2D) → FlashAvatar (120D) 変換アダプタ。

    DECAToFlameAdapterと同一のFLAME PCA空間変換を使いつつ、
    SMIRKが推定する eyelid パラメータを活用する。

    変換ロジック:
        1. ``exp`` (B, 50) → ``F.pad(exp, (0, 50), value=0.0)`` → ``expr`` (B, 100)
        2. ``pose[:, 3:6]`` axis-angle (B, 3)
           → ``axis_angle_to_matrix`` → (B, 3, 3)
           → ``matrix_to_rotation_6d`` → ``jaw_pose`` (B, 6)
        3. ``eyes_pose``: 単位回転行列 → 6D表現 → repeat × 2 → (B, 12)
        4. ``eyelids``: SMIRK の ``eyelid`` (B, 2) をそのまま使用
    """

    def convert(
        self, source_params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """SMIRKパラメータをFlashAvatar形式に変換する。

        Args:
            source_params: SMIRK出力の辞書。必須キー:
                - ``"exp"``: 表情パラメータ (B, 50)
                - ``"pose"``: 姿勢パラメータ (B, 6)。
                  global_rotation(3D) + jaw_pose(3D)。
                - ``"eyelid"``: 瞼パラメータ (B, 2)。
                  SMIRKExtractor の ``"eyelid"`` キーと対応する。

        Returns:
            FlashAvatar形式のパラメータ辞書:
                - ``"expr"``: (B, 100) FLAME表情パラメータ
                - ``"jaw_pose"``: (B, 6) 顎回転 (rotation_6d)
                - ``"eyes_pose"``: (B, 12) 眼球回転 (identity_6d × 2)
                - ``"eyelids"``: (B, 2) 瞼パラメータ (SMIRK推定値)

        Raises:
            KeyError: source_paramsに必要なキーが存在しない場合。
        """
        exp_50d = source_params["exp"]
        pose = source_params["pose"]
        eyelid = source_params["eyelid"]
        batch_size = exp_50d.shape[0]
        device = exp_50d.device
        dtype = exp_50d.dtype

        # exp 50D → expr 100D: ゼロパディング
        expr_100d = F.pad(exp_50d, (0, 50), value=0.0)

        # jaw_pose: axis-angle(3D) → rotation_6d(6D)
        jaw_aa = pose[:, 3:6]
        jaw_mat = _aa_to_mat(jaw_aa)
        jaw_6d = _mat_to_6d(jaw_mat)

        # eyes_pose: 単位回転行列の6D表現 × 2
        identity_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        identity_6d = _mat_to_6d(identity_mat)
        eyes_pose = identity_6d.repeat(batch_size, 1).repeat(1, 2)

        # eyelids: SMIRK 推定値をそのまま使用 (DECA と異なりゼロ埋めしない)
        eyelids = eyelid

        return {
            "expr": expr_100d,
            "jaw_pose": jaw_6d,
            "eyes_pose": eyes_pose,
            "eyelids": eyelids,
        }

    @property
    def source_format(self) -> str:
        """変換元のパラメータ形式名を返す。

        Returns:
            ``"smirk"``。
        """
        return "smirk"

    @property
    def target_format(self) -> str:
        """変換先のパラメータ形式名を返す。

        Returns:
            ``"flash_avatar"``。
        """
        return "flash_avatar"
