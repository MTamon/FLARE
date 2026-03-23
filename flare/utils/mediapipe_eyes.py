"""MediaPipe eyes_pose / eyelids 推定モジュール。

仕様書4.3節「eyes_pose / eyelidsの推定方法」の方法A（MediaPipe推定）を実装する。
MediaPipe Face Landmarkerのアイランドマークから、FLAMEのeye poseおよび
eyelidsパラメータを推定する。

推定方法一覧（仕様書4.3節）:
    ====== ====== ========== ==============================================
    方法    精度    実装コスト  説明
    ====== ====== ========== ==============================================
    A      高      中         MediaPipe Face Landmarker → FLAME eye pose変換
    B      低〜中  低         単位回転埋め（Phase 1デフォルト）
    C      高      高         expression/poseから推定する小規模MLPを学習
    ====== ====== ========== ==============================================

本モジュールは方法Aを実装する。Phase 1のDECAToFlameAdapterでは方法B
（単位回転埋め）がデフォルトであり、本モジュールはオプション機能として提供する。

参照リポジトリ:
    mediapipe-blendshapes-to-flame（MediaPipe BlendShapes → FLAME変換）

Example:
    >>> estimator = MediaPipeEyeEstimator()
    >>> result = estimator.estimate(bgr_frame, deca_pose)
    >>> result["eyes_pose"].shape  # (1, 12)
    >>> result["eyelids"].shape    # (1, 2)
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
import torch
from loguru import logger

# MediaPipeインポート（利用不可の場合はフォールバック）
try:
    import mediapipe as mp

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        _MP_BACKEND: str = "solutions"
    else:
        _MP_BACKEND = "unavailable"
except ImportError:
    _MP_BACKEND = "unavailable"


class MediaPipeEyeEstimator:
    """MediaPipeベースの眼球回転・瞼パラメータ推定器。

    MediaPipe Face Meshのアイランドマークから左右の眼球回転角度を推定し、
    FLAMEのeyes_pose（6D rotation × 2 = 12D）およびeyelids（2D）に変換する。

    MediaPipeが利用不可の場合は自動的にフォールバック（単位回転 + ゼロ瞼）を返す。

    Attributes:
        _face_mesh: MediaPipe FaceMeshインスタンス。
        _device: テンソル出力先デバイス。

    Example:
        >>> estimator = MediaPipeEyeEstimator()
        >>> result = estimator.estimate(frame_bgr, deca_pose_tensor)
        >>> eyes_pose = result["eyes_pose"]   # (1, 12)
        >>> eyelids = result["eyelids"]       # (1, 2)
    """

    #: 左目のアイリス中心ランドマークインデックス（MediaPipe FaceMesh）
    _LEFT_IRIS_CENTER: int = 468

    #: 右目のアイリス中心ランドマークインデックス
    _RIGHT_IRIS_CENTER: int = 473

    #: 左目の上瞼ランドマークインデックス
    _LEFT_UPPER_EYELID: int = 159

    #: 左目の下瞼ランドマークインデックス
    _LEFT_LOWER_EYELID: int = 145

    #: 右目の上瞼ランドマークインデックス
    _RIGHT_UPPER_EYELID: int = 386

    #: 右目の下瞼ランドマークインデックス
    _RIGHT_LOWER_EYELID: int = 374

    #: 左目外側コーナー
    _LEFT_EYE_OUTER: int = 33

    #: 左目内側コーナー
    _LEFT_EYE_INNER: int = 133

    #: 右目外側コーナー
    _RIGHT_EYE_OUTER: int = 362

    #: 右目内側コーナー
    _RIGHT_EYE_INNER: int = 263

    def __init__(
        self,
        device: str = "cpu",
        *,
        min_detection_confidence: float = 0.5,
    ) -> None:
        """MediaPipeEyeEstimatorを初期化する。

        Args:
            device: テンソル出力先デバイス。
            min_detection_confidence: MediaPipe検出の最小信頼度。
        """
        self._device: str = device
        self._face_mesh: object | None = None

        if _MP_BACKEND == "solutions":
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )
            logger.debug(
                "MediaPipeEyeEstimator初期化: backend=solutions | confidence={}",
                min_detection_confidence,
            )
        else:
            logger.info(
                "MediaPipe Face Meshが利用できません。"
                "eyes_pose/eyelidsはフォールバック値を返します。"
            )

    def estimate(
        self,
        image: np.ndarray,
        deca_pose: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """眼球回転および瞼パラメータを推定する。

        MediaPipe Face Meshからアイランドマークを取得し、
        眼球回転をFLAMEのeyes_pose（6D rotation × 2）に変換する。
        瞼の開閉度はeyelids（2D）として返す。

        検出失敗時は単位回転（identity_6d × 2）とゼロ瞼にフォールバックする。

        Args:
            image: 入力画像（BGR形式、shape: (H, W, 3)、dtype: uint8）。
                顔クロップ前の元フレーム画像。
            deca_pose: DECA poseパラメータ（shape: (1, 6)）。
                global_rotation(3D) + jaw_pose(3D)。
                眼球回転の基準座標系の決定に使用する。

        Returns:
            推定結果Dict:
                - ``"eyes_pose"``: (1, 12) 左右眼球回転の6D表現 × 2
                - ``"eyelids"``: (1, 2) 左右瞼の開閉度
        """
        if self._face_mesh is None:
            return self._fallback_output()

        landmarks: Optional[np.ndarray] = self._detect_landmarks(image)

        if landmarks is None:
            return self._fallback_output()

        try:
            eyes_pose: torch.Tensor = self._mediapipe_to_flame_eye_pose(
                landmarks
            )
            eyelids: torch.Tensor = self._estimate_eyelids(landmarks, image.shape)

            return {
                "eyes_pose": eyes_pose,
                "eyelids": eyelids,
            }
        except Exception as exc:
            logger.debug("eyes_pose推定失敗、フォールバック: {}", exc)
            return self._fallback_output()

    def _detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """MediaPipe Face Meshでランドマークを検出する。

        Args:
            image: BGR入力画像。

        Returns:
            全ランドマーク座標（shape: (478, 3)、正規化座標）。
            未検出時はNone。
        """
        rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            results = self._face_mesh.process(rgb)
        except Exception:
            return None

        if (
            results.multi_face_landmarks is None
            or len(results.multi_face_landmarks) == 0
        ):
            return None

        face_lm = results.multi_face_landmarks[0]
        n_landmarks: int = len(face_lm.landmark)

        landmarks: np.ndarray = np.zeros((n_landmarks, 3), dtype=np.float32)
        for i, lm in enumerate(face_lm.landmark):
            landmarks[i, 0] = lm.x
            landmarks[i, 1] = lm.y
            landmarks[i, 2] = lm.z

        return landmarks

    def _mediapipe_to_flame_eye_pose(
        self, landmarks: np.ndarray
    ) -> torch.Tensor:
        """MediaPipeランドマークからFLAME eye pose（6D rotation）を計算する。

        左右の眼球について、アイリス中心と眼窩中心の相対位置から
        視線方向（yaw, pitch）を推定し、回転行列を構成して6D表現に変換する。

        Args:
            landmarks: MediaPipe FaceMeshランドマーク（shape: (N, 3)、正規化座標）。

        Returns:
            eyes_poseテンソル（shape: (1, 12)）。左6D + 右6D。
        """
        n_lm: int = landmarks.shape[0]

        # 左目
        left_6d: np.ndarray = self._compute_eye_rotation(
            landmarks,
            iris_idx=self._LEFT_IRIS_CENTER if n_lm > self._LEFT_IRIS_CENTER else self._LEFT_EYE_INNER,
            outer_idx=self._LEFT_EYE_OUTER,
            inner_idx=self._LEFT_EYE_INNER,
            upper_idx=self._LEFT_UPPER_EYELID,
            lower_idx=self._LEFT_LOWER_EYELID,
        )

        # 右目
        right_6d: np.ndarray = self._compute_eye_rotation(
            landmarks,
            iris_idx=self._RIGHT_IRIS_CENTER if n_lm > self._RIGHT_IRIS_CENTER else self._RIGHT_EYE_INNER,
            outer_idx=self._RIGHT_EYE_OUTER,
            inner_idx=self._RIGHT_EYE_INNER,
            upper_idx=self._RIGHT_UPPER_EYELID,
            lower_idx=self._RIGHT_LOWER_EYELID,
        )

        eyes_12d: np.ndarray = np.concatenate([left_6d, right_6d])  # (12,)
        return torch.tensor(eyes_12d, dtype=torch.float32).unsqueeze(0).to(
            self._device
        )  # (1, 12)

    def _compute_eye_rotation(
        self,
        landmarks: np.ndarray,
        iris_idx: int,
        outer_idx: int,
        inner_idx: int,
        upper_idx: int,
        lower_idx: int,
    ) -> np.ndarray:
        """単眼の眼球回転を6D表現で計算する。

        アイリス中心と眼窩中心の相対位置から視線方向を推定し、
        小角近似で回転行列を構成する。

        Args:
            landmarks: 全ランドマーク（正規化座標）。
            iris_idx: アイリス中心インデックス。
            outer_idx: 眼外側コーナーインデックス。
            inner_idx: 眼内側コーナーインデックス。
            upper_idx: 上瞼インデックス。
            lower_idx: 下瞼インデックス。

        Returns:
            6D rotation表現（shape: (6,)）。
        """
        iris: np.ndarray = landmarks[iris_idx, :2]
        outer: np.ndarray = landmarks[outer_idx, :2]
        inner: np.ndarray = landmarks[inner_idx, :2]
        upper: np.ndarray = landmarks[upper_idx, :2]
        lower: np.ndarray = landmarks[lower_idx, :2]

        # 眼窩中心
        eye_center: np.ndarray = (outer + inner) / 2.0
        eye_v_center: np.ndarray = (upper + lower) / 2.0

        # 眼窩サイズで正規化
        eye_width: float = float(np.linalg.norm(outer - inner))
        eye_height: float = float(np.linalg.norm(upper - lower))

        if eye_width < 1e-6 or eye_height < 1e-6:
            return self._identity_6d()

        # 視線方向（正規化相対位置）→ yaw / pitch角度
        dx: float = float(iris[0] - eye_center[0]) / eye_width
        dy: float = float(iris[1] - eye_v_center[1]) / eye_height

        # 角度推定（経験的スケーリング: MediaPipe正規化座標 → ラジアン）
        yaw: float = dx * 0.7  # 水平方向
        pitch: float = -dy * 0.5  # 垂直方向（画像y軸は下向き）

        # 回転行列構成（小角近似: Ry(yaw) @ Rx(pitch)）
        cy: float = np.cos(yaw)
        sy: float = np.sin(yaw)
        cp: float = np.cos(pitch)
        sp: float = np.sin(pitch)

        # R = Ry @ Rx
        R: np.ndarray = np.array([
            [cy, sy * sp, sy * cp],
            [0.0, cp, -sp],
            [-sy, cy * sp, cy * cp],
        ], dtype=np.float32)

        # 6D representation: first 2 rows
        return R[:2, :].flatten()  # (6,)

    def _estimate_eyelids(
        self,
        landmarks: np.ndarray,
        image_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """瞼の開閉度を推定する。

        上瞼と下瞼のランドマーク距離を眼窩幅で正規化し、
        0（完全閉眼）〜1（完全開眼）のスケールで返す。

        Args:
            landmarks: 全ランドマーク（正規化座標）。
            image_shape: 入力画像のshape（正規化解除用）。

        Returns:
            eyelidsテンソル（shape: (1, 2)）。[左瞼, 右瞼]。
        """
        left_open: float = self._compute_eye_openness(
            landmarks,
            self._LEFT_UPPER_EYELID,
            self._LEFT_LOWER_EYELID,
            self._LEFT_EYE_OUTER,
            self._LEFT_EYE_INNER,
        )

        right_open: float = self._compute_eye_openness(
            landmarks,
            self._RIGHT_UPPER_EYELID,
            self._RIGHT_LOWER_EYELID,
            self._RIGHT_EYE_OUTER,
            self._RIGHT_EYE_INNER,
        )

        eyelids: np.ndarray = np.array(
            [[left_open, right_open]], dtype=np.float32
        )
        return torch.tensor(eyelids, dtype=torch.float32).to(self._device)

    @staticmethod
    def _compute_eye_openness(
        landmarks: np.ndarray,
        upper_idx: int,
        lower_idx: int,
        outer_idx: int,
        inner_idx: int,
    ) -> float:
        """単眼の開閉度を計算する。

        Args:
            landmarks: 全ランドマーク。
            upper_idx: 上瞼インデックス。
            lower_idx: 下瞼インデックス。
            outer_idx: 眼外側コーナーインデックス。
            inner_idx: 眼内側コーナーインデックス。

        Returns:
            開閉度（0.0〜1.0）。
        """
        upper: np.ndarray = landmarks[upper_idx, :2]
        lower: np.ndarray = landmarks[lower_idx, :2]
        outer: np.ndarray = landmarks[outer_idx, :2]
        inner: np.ndarray = landmarks[inner_idx, :2]

        eye_height: float = float(np.linalg.norm(upper - lower))
        eye_width: float = float(np.linalg.norm(outer - inner))

        if eye_width < 1e-6:
            return 0.5

        # EAR (Eye Aspect Ratio) を0-1にスケーリング
        ear: float = eye_height / eye_width
        openness: float = np.clip(ear / 0.4, 0.0, 1.0)
        return float(openness)

    def _fallback_output(self) -> Dict[str, torch.Tensor]:
        """フォールバック出力を返す。

        単位回転行列の6D表現 × 2（eyes_pose）とゼロ瞼（eyelids）。
        方法B（単位回転埋め）と同等の出力。

        Returns:
            フォールバックのeyes_pose(1,12)とeyelids(1,2)。
        """
        identity_6d: np.ndarray = np.array(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32
        )
        eyes_pose: np.ndarray = np.concatenate(
            [identity_6d, identity_6d]
        ).reshape(1, 12)

        eyelids: np.ndarray = np.zeros((1, 2), dtype=np.float32)

        return {
            "eyes_pose": torch.tensor(eyes_pose, dtype=torch.float32).to(
                self._device
            ),
            "eyelids": torch.tensor(eyelids, dtype=torch.float32).to(
                self._device
            ),
        }

    @staticmethod
    def _identity_6d() -> np.ndarray:
        """単位回転行列の6D表現を返す。

        Returns:
            [1, 0, 0, 0, 1, 0] の6要素配列。
        """
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
