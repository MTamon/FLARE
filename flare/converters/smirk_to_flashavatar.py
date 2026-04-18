"""SMIRK→FlashAvatar パラメータ変換モジュール。

SMIRK出力（exp 50D + pose 6D + eyelid 2D + cam 3D + shape 300D）を
FlashAvatarのcondition vector（120D）に変換する。

変換ロジック:
    - ``shape`` (B, 300): FlashAvatar は条件ベクトルに使わないので破棄。
    - ``exp`` (B, 50) → ``expr`` (B, 100): 同一FLAME PCA空間のためゼロパディング。
    - ``pose[:, 3:6]`` axis-angle (B, 3) → ``jaw_pose`` (B, 6): rotation_6d 変換。
    - ``eyes_pose`` (B, 12): SMIRK 本来は出力しないため、デフォルトは
      単位回転行列の6D表現 × 2。``use_mediapipe_supplement=True`` のとき
      ``source_params["eyes_pose"]`` が指定されていれば優先採用する。
    - ``eyelid`` (B, 2) → ``eyelids`` (B, 2): SMIRK ネイティブ出力をそのまま
      パススルー (キー名のみ ``eyelid`` → ``eyelids`` にリネーム)。

DECA との違い:
    - ``shape`` キーを受け取るが、FlashAvatar の条件ベクトルには使わないので破棄。
    - ``eyelids`` は SMIRK が ExpressionEncoder の最終層からネイティブ出力する
      (clamp [0, 1])。``use_mediapipe_supplement`` フラグの対象外であり、常に
      SMIRK 本来の値を採用する。MediaPipe 補完が制御するのは ``eyes_pose`` のみ。
    - SMIRK の ``pose`` は cuda128 fork で ``cat([pose_params(3), jaw_params(3)])``
      に組み直し済み。DECA の ``pose`` レイアウト [global_rot(3), jaw(3)] と一致するため、
      ``pose[:, 3:6]`` で jaw axis-angle を取り出す処理は DECA と完全に共通化できる。

MediaPipe 補完オプション:
    SMIRK は eyes_pose を出力しない (jaw, eyelid のみ)。本Adapterのデフォルトは
    SMIRK 本来挙動に従い、eyes_pose は単位回転を返す。
    ``use_mediapipe_supplement=True`` を指定すると、呼び出し側が事前に推定した
    ``eyes_pose`` (例: MediaPipe Face Landmarker から得られる値) を
    ``source_params`` 経由で注入できる。フラグが False の場合は、``source_params``
    に ``eyes_pose`` が含まれていても無視される (SMIRK 本来挙動を厳守)。

Example:
    SMIRK 本来挙動 (デフォルト)::

        adapter = SmirkToFlashAvatarAdapter()
        flash_params = adapter.convert({
            "shape":  torch.randn(1, 300),
            "exp":    torch.randn(1,  50),
            "pose":   torch.randn(1,   6),
            "cam":    torch.randn(1,   3),
            "eyelid": torch.randn(1,   2),
        })
        # eyes_pose は identity_6d × 2、 eyelids は SMIRK ネイティブ値

    MediaPipe 補完で eyes_pose を注入::

        adapter = SmirkToFlashAvatarAdapter(use_mediapipe_supplement=True)
        eyes_pose, _ = face_detector.detect_eye_pose(frame, bbox)
        flash_params = adapter.convert({
            "shape": shape, "exp": exp, "pose": pose,
            "cam": cam, "eyelid": eyelid,
            "eyes_pose": eyes_pose,   # MediaPipe由来の値を注入
        })
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flare.converters.base import BaseAdapter
from flare.converters.deca_to_flame import _aa_to_mat, _mat_to_6d


class SmirkToFlashAvatarAdapter(BaseAdapter):
    """SMIRK → FlashAvatar 変換アダプタ。

    同一FLAME PCA空間に基づくゼロパディング変換 (DECA と同じ) に加え、
    SMIRK ネイティブの ``eyelid`` 出力を ``eyelids`` キーにリネームしてそのまま
    パススルーする。``shape`` キーは FlashAvatar 条件ベクトルでは使わないので破棄。

    変換ロジック:
        1. ``exp`` (B, 50) → ``F.pad(exp, (0, 50), value=0.0)`` → ``expr`` (B, 100)。
        2. ``pose[:, 3:6]`` axis-angle → ``jaw_pose`` (B, 6) (rotation_6d)。
        3. ``eyes_pose`` (B, 12): デフォルトは単位回転行列の6D表現 × 2。
           ``use_mediapipe_supplement=True`` のとき ``source_params["eyes_pose"]``
           があればそちらを優先採用する。
        4. ``eyelids`` (B, 2): ``source_params["eyelid"]`` をそのまま渡す。

    Attributes:
        _use_mediapipe_supplement: MediaPipe 由来の ``eyes_pose`` 注入を許可するか。
            SMIRK は eyes_pose を出力しないため、デフォルトは ``False``。
            なお SMIRK は eyelid をネイティブ出力するため、本フラグの制御対象は
            ``eyes_pose`` のみ (eyelids は常に SMIRK の値を使う)。
    """

    def __init__(self, use_mediapipe_supplement: bool = False) -> None:
        """SmirkToFlashAvatarAdapterを初期化する。

        Args:
            use_mediapipe_supplement: MediaPipe等で外部推定した ``eyes_pose``
                の注入を許可するかどうか。
                ``False`` (デフォルト): SMIRK 本来挙動 (eyes_pose=identity)。
                  ``source_params`` に ``eyes_pose`` が含まれていても無視する。
                ``True``: ``source_params["eyes_pose"]`` が存在すれば優先採用し、
                  無ければデフォルト値 (identity × 2) にフォールバックする。
        """
        self._use_mediapipe_supplement = use_mediapipe_supplement

    def convert(
        self, source_params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """SMIRKパラメータをFlashAvatar形式に変換する。

        Args:
            source_params: SMIRK出力の辞書。必須キー:
                - ``"exp"``: 表情パラメータ (B, 50)
                - ``"pose"``: 姿勢パラメータ (B, 6)。
                  global_rotation(3D) + jaw_pose(3D)。
                - ``"eyelid"``: 瞼パラメータ (B, 2)。SMIRK ネイティブ出力。

                オプションキー (常に許可):
                - ``"shape"``: 形状パラメータ (B, 300)。受け取るが破棄する。
                - ``"cam"``: カメラパラメータ (B, 3)。受け取るが破棄する。

                オプションキー (``use_mediapipe_supplement=True`` のとき有効):
                - ``"eyes_pose"``: 外部推定の眼球回転 (B, 12)。

        Returns:
            FlashAvatar形式のパラメータ辞書:
                - ``"expr"``: (B, 100) FLAME表情パラメータ
                - ``"jaw_pose"``: (B, 6) 顎回転 (rotation_6d)
                - ``"eyes_pose"``: (B, 12) 眼球回転
                - ``"eyelids"``: (B, 2) 瞼パラメータ (SMIRK ネイティブ)

        Raises:
            KeyError: source_params に ``"exp"`` / ``"pose"`` / ``"eyelid"``
                のいずれかが存在しない場合。
        """
        exp_50d = source_params["exp"]
        pose = source_params["pose"]
        eyelid = source_params["eyelid"]
        batch_size = exp_50d.shape[0]
        device = exp_50d.device
        dtype = exp_50d.dtype

        expr_100d = F.pad(exp_50d, (0, 50), value=0.0)

        jaw_aa = pose[:, 3:6]
        jaw_mat = _aa_to_mat(jaw_aa)
        jaw_6d = _mat_to_6d(jaw_mat)

        eyes_pose = self._resolve_eyes_pose(source_params, batch_size, device, dtype)
        eyelids = eyelid.to(device=device, dtype=dtype)

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

    @property
    def use_mediapipe_supplement(self) -> bool:
        """MediaPipe補完が有効かどうかを返す。"""
        return self._use_mediapipe_supplement

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
