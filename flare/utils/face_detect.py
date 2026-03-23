"""顔検出・クロッピングユーティリティ。

仕様書8.3節「face_detect.py: 顔検出の責務分離」に基づき、
顔検出・バウンディングボックス取得・クロッピング・ランドマーク検出を
共有ユーティリティとして提供する。

責務分離の原則:
    ============== =================================== ===========================
    モジュール       責務                                 詳細
    ============== =================================== ===========================
    face_detect.py 顔検出・クロッピング                  入力画像から顔領域を検出
    Extractor      3DMMパラメータ推定                    検出済み顔画像のみ受け取る
    ============== =================================== ===========================

DECA FAN に関する発見（v2.0）:
    DECAのencode()内部にFAN顔検出は含まれない。FAN は前処理でのみ使用されるため、
    本ツールのface_detect.pyとDECAの共存にバイパス改修は不要。

バックエンド選択:
    1. MediaPipe solutions API（0.10.11等）: ``mp.solutions.face_detection``
    2. MediaPipe tasks API（0.10.20+）: ``mp.tasks.vision``
    3. OpenCV Haar Cascade（フォールバック）

Example:
    >>> detector = FaceDetector(min_detection_confidence=0.7)
    >>> bbox = detector.detect(frame)
    >>> cropped = detector.crop_and_align(frame, bbox, size=224)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from flare.utils.errors import FaceNotDetectedError

# --- MediaPipe バックエンド検出 ---
_BACKEND: str = "haar"  # default fallback

try:
    import mediapipe as mp

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        _BACKEND = "mediapipe_solutions"
    elif hasattr(mp, "tasks"):
        _BACKEND = "mediapipe_tasks"
    else:
        _BACKEND = "haar"
except ImportError:
    pass

if _BACKEND == "haar":
    logger.info(
        "MediaPipeが利用できません。OpenCV Haar Cascadeにフォールバックします。"
    )


class FaceDetector:
    """顔検出器。

    MediaPipe Face Detection（solutions API または tasks API）を使用して
    顔のバウンディングボックスを検出する。MediaPipeが利用不可の場合は
    OpenCV Haar Cascadeにフォールバックする。

    Attributes:
        _min_detection_confidence: 検出の最小信頼度閾値。
        _fallback_to_prev: 顔未検出時に前フレームbboxを使用するか。
        _prev_bbox: 前フレームのバウンディングボックス。

    Example:
        >>> detector = FaceDetector(min_detection_confidence=0.5)
        >>> bbox = detector.detect(frame)  # (x1, y1, x2, y2)
        >>> cropped = detector.crop_and_align(frame, bbox, size=224)
    """

    #: MediaPipe FaceMesh 468点からDlib互換68点へのマッピングインデックス。
    _FACE_MESH_TO_68: list[int] = [
        # 顎ライン (0-16): 17点
        162, 234, 93, 132, 58, 172, 136, 150, 149, 176,
        148, 152, 377, 400, 378, 379, 365, 397, 288, 361,
        323, 454, 389,
        # 左眉 (17-21): 5点
        70, 63, 105, 66, 107,
        # 右眉 (22-26): 5点
        336, 296, 334, 293, 300,
        # 鼻梁 (27-30): 4点
        168, 6, 197, 195,
        # 鼻下部 (31-35): 5点
        5, 4, 1, 275, 440,
        # 左目 (36-41): 6点
        33, 160, 158, 133, 153, 144,
        # 右目 (42-47): 6点
        362, 385, 387, 263, 373, 380,
        # 外唇 (48-59): 12点
        61, 39, 37, 0, 267, 269, 291, 405, 314,
        17, 84, 181,
        # 内唇 (60-67): 8点
        78, 82, 13, 312, 308, 317, 14, 87,
    ]

    def __init__(
        self,
        device: str = "cpu",
        *,
        min_detection_confidence: float = 0.5,
        fallback_to_prev: bool = True,
    ) -> None:
        """FaceDetectorを初期化する。

        Args:
            device: 計算デバイス（現在はCPUのみサポート）。
            min_detection_confidence: 検出の最小信頼度。0.0〜1.0。
            fallback_to_prev: 顔未検出時に前フレームのbboxに
                フォールバックするかどうか。デフォルトTrue。
        """
        self._device: str = device
        self._min_detection_confidence: float = min_detection_confidence
        self._fallback_to_prev: bool = fallback_to_prev
        self._prev_bbox: Optional[Tuple[int, int, int, int]] = None
        self._backend: str = _BACKEND

        # バックエンド固有オブジェクト
        self._face_detection: object | None = None
        self._face_mesh: object | None = None
        self._haar_cascade: cv2.CascadeClassifier | None = None

        self._init_backend()

        logger.debug(
            "FaceDetector初期化: backend={} | confidence={} | fallback={}",
            self._backend,
            min_detection_confidence,
            fallback_to_prev,
        )

    def _init_backend(self) -> None:
        """検出バックエンドを初期化する。

        MediaPipe solutions API → tasks API → Haar Cascade の順で試行する。
        """
        if self._backend == "mediapipe_solutions":
            try:
                self._face_detection = (
                    mp.solutions.face_detection.FaceDetection(
                        min_detection_confidence=self._min_detection_confidence,
                        model_selection=1,
                    )
                )
                if hasattr(mp.solutions, "face_mesh"):
                    self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=self._min_detection_confidence,
                    )
            except Exception as exc:
                logger.warning(
                    "MediaPipe solutions初期化失敗、Haarにフォールバック: {}", exc
                )
                self._backend = "haar"
                self._init_haar()
        elif self._backend == "mediapipe_tasks":
            # tasks APIではモデルファイルが必要。
            # 利用不可の場合はHaarにフォールバック。
            logger.info(
                "MediaPipe tasks API検出。Haar Cascadeをフォールバックとして併用。"
            )
            self._backend = "haar"
            self._init_haar()
        else:
            self._init_haar()

    def _init_haar(self) -> None:
        """Haar Cascadeを初期化する。"""
        cascade_path: str = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._haar_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """画像から顔バウンディングボックスを検出する。

        Args:
            image: 入力画像（BGR形式、shape: (H, W, 3)、dtype: uint8）。

        Returns:
            バウンディングボックス ``(x1, y1, x2, y2)``。
            座標はピクセル単位の整数値。

        Raises:
            FaceNotDetectedError: 顔が検出されず、フォールバックも
                利用できない場合。
        """
        bbox: Optional[Tuple[int, int, int, int]] = self._detect_impl(image)

        if bbox is not None:
            self._prev_bbox = bbox
            return bbox

        # フォールバック
        if self._fallback_to_prev and self._prev_bbox is not None:
            logger.debug("顔未検出: 前フレームbboxにフォールバック")
            return self._prev_bbox

        raise FaceNotDetectedError(
            "顔が検出されませんでした。フォールバックbboxも利用できません。"
        )

    def crop_and_align(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        size: int = 224,
        *,
        margin: float = 0.2,
    ) -> np.ndarray:
        """バウンディングボックスで顔をクロップしリサイズする。

        bboxをmargin比率で拡張し、画像境界にクランプした後、
        指定サイズにリサイズして返す。

        Args:
            image: 入力画像（BGR形式、shape: (H, W, 3)）。
            bbox: バウンディングボックス ``(x1, y1, x2, y2)``。
            size: 出力画像の辺の長さ（正方形）。デフォルト224。
            margin: bboxの拡張率。デフォルト0.2（20%拡張）。

        Returns:
            クロップ・リサイズ済み画像（shape: (size, size, 3)、dtype: uint8）。
        """
        h: int = image.shape[0]
        w: int = image.shape[1]
        x1, y1, x2, y2 = bbox

        # マージン拡張
        bw: int = x2 - x1
        bh: int = y2 - y1
        margin_x: int = int(bw * margin)
        margin_y: int = int(bh * margin)

        x1_exp: int = max(0, x1 - margin_x)
        y1_exp: int = max(0, y1 - margin_y)
        x2_exp: int = min(w, x2 + margin_x)
        y2_exp: int = min(h, y2 + margin_y)

        cropped: np.ndarray = image[y1_exp:y2_exp, x1_exp:x2_exp]
        resized: np.ndarray = cv2.resize(
            cropped, (size, size), interpolation=cv2.INTER_LINEAR
        )
        return resized

    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """68点顔ランドマークを検出する。

        MediaPipe FaceMeshの468点からDlib互換68点にマッピングして返す。
        MediaPipe solutions APIが利用不可またはランドマーク検出に失敗した場合は
        Noneを返す。

        Args:
            image: 入力画像（BGR形式、shape: (H, W, 3)、dtype: uint8）。

        Returns:
            ランドマーク座標（shape: (68, 2)、dtype: float32）。
            各行は ``(x, y)`` ピクセル座標。未検出時は ``None``。
        """
        if self._face_mesh is None:
            return None

        h: int = image.shape[0]
        w: int = image.shape[1]

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

        face_landmarks = results.multi_face_landmarks[0]
        all_lm = face_landmarks.landmark

        n_mapped: int = min(len(self._FACE_MESH_TO_68), 68)
        landmarks_68: np.ndarray = np.zeros((68, 2), dtype=np.float32)

        for i in range(n_mapped):
            idx: int = self._FACE_MESH_TO_68[i]
            if idx < len(all_lm):
                landmarks_68[i, 0] = all_lm[idx].x * w
                landmarks_68[i, 1] = all_lm[idx].y * h

        return landmarks_68

    def reset_state(self) -> None:
        """内部状態をリセットする。

        前フレームのバウンディングボックスをクリアする。
        新しい動画やセッションの開始時に呼び出す。
        """
        self._prev_bbox = None
        logger.debug("FaceDetector状態リセット")

    def _detect_impl(
        self, image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """顔検出の内部実装。

        現在のバックエンドに応じた検出メソッドを呼び出す。

        Args:
            image: 入力画像（BGR形式）。

        Returns:
            検出されたバウンディングボックス ``(x1, y1, x2, y2)``。
            未検出時は ``None``。
        """
        if self._backend == "mediapipe_solutions" and self._face_detection is not None:
            return self._detect_mediapipe(image)
        return self._detect_haar(image)

    def _detect_mediapipe(
        self, image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """MediaPipe solutions APIによる顔検出。

        Args:
            image: 入力画像（BGR形式）。

        Returns:
            バウンディングボックスまたはNone。
        """
        h: int = image.shape[0]
        w: int = image.shape[1]

        rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            results = self._face_detection.process(rgb)
        except Exception:
            return None

        if results.detections is None or len(results.detections) == 0:
            return None

        detection = results.detections[0]
        bbox_rel = detection.location_data.relative_bounding_box

        x1: int = max(0, int(bbox_rel.xmin * w))
        y1: int = max(0, int(bbox_rel.ymin * h))
        x2: int = min(w, int((bbox_rel.xmin + bbox_rel.width) * w))
        y2: int = min(h, int((bbox_rel.ymin + bbox_rel.height) * h))

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    def _detect_haar(
        self, image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """OpenCV Haar Cascadeによる顔検出（フォールバック）。

        Args:
            image: 入力画像（BGR形式）。

        Returns:
            バウンディングボックスまたはNone。
        """
        if self._haar_cascade is None:
            return None

        gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if isinstance(faces, tuple) or len(faces) == 0:
            return None

        # 最大面積の検出を使用
        faces_arr: np.ndarray = np.array(faces)
        areas: np.ndarray = faces_arr[:, 2] * faces_arr[:, 3]
        best_idx: int = int(np.argmax(areas))
        x, y, fw, fh = faces_arr[best_idx]

        return (int(x), int(y), int(x + fw), int(y + fh))
