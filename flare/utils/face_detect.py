"""顔検出の共有ユーティリティモジュール。

仕様書§8.3に基づき、顔検出・クロッピングの責務をExtractorから分離した
共有ユーティリティとして実装する。Extractorは本モジュールで切り出された
顔画像のみを受け取り、顔検出は行わない。

バックエンドとしてMediaPipe Face Detectionを使用する。
MediaPipe v0.10.11以前（``mp.solutions`` API）と v0.10.14以降
（``mp.tasks`` API）の両方に対応する。

仕様書§4.3に基づく眼球ポーズ推定（方式A）:
    MediaPipe Face LandmarkerのblendshapeスコアからFLAME eye pose/eyelidsを
    推定する。mediapipe-blendshapes-to-flameの変換マッピングを使用して、
    blendshapeスコアをFLAME互換の6D rotation × 2（eyes_pose 12D）と
    eyelids 2D に変換する。

パイプライン処理フロー::

    # face_detect.py が顔検出・クロッピングを実行
    face_bbox = face_detector.detect(frame)
    cropped = face_detector.crop_and_align(frame, face_bbox, size=224)
    image_tensor = to_tensor(cropped).to(device)

    # Extractor は検出済み画像のみ受け取る（FAN 呼び出しなし）
    codedict = deca_model.encode(image_tensor)

    # 眼球ポーズ推定
    eyes_pose, eyelids = face_detector.detect_eye_pose(frame, face_bbox)

Example:
    FaceDetectorの基本的な使用::

        detector = FaceDetector(min_detection_confidence=0.5)
        bbox = detector.detect(frame)
        if bbox is not None:
            cropped = detector.crop_and_align(frame, bbox, size=224)
            eyes_pose, eyelids = detector.detect_eye_pose(frame, bbox)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

_USE_SOLUTIONS_API: bool = True
"""bool: True なら ``mp.solutions`` API、False なら ``mp.tasks`` API を使用する。"""

try:
    import mediapipe as mp

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        _USE_SOLUTIONS_API = True
    else:
        _USE_SOLUTIONS_API = False
except ImportError:
    _USE_SOLUTIONS_API = False


# =========================================================================
# Blendshape → FLAME eye pose 変換ヘルパー
# =========================================================================

# MediaPipe blendshapeスコアからFLAME eye poseへの変換係数。
# mediapipe-blendshapes-to-flame マッピングに基づく。
# 各blendshapeスコア(0-1)をaxis-angle回転角(ラジアン)にスケールする。
_EYE_PITCH_SCALE: float = 0.35
"""float: 眼球pitch回転のスケール係数（ラジアン/blendshapeスコア）。"""

_EYE_YAW_SCALE: float = 0.45
"""float: 眼球yaw回転のスケール係数（ラジアン/blendshapeスコア）。"""


def _default_eyes_pose() -> torch.Tensor:
    """デフォルトのeyes_pose（identity rotation × 2）を返す。

    Returns:
        identity rotation 6D表現を2つ連結したテンソル (1, 12)。
    """
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return torch.cat([identity_6d, identity_6d]).unsqueeze(0)


def _default_eyelids() -> torch.Tensor:
    """デフォルトのeyelids（ゼロ、開眼状態）を返す。

    Returns:
        ゼロテンソル (1, 2)。
    """
    return torch.zeros(1, 2)


def _axis_angle_to_rotation_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """axis-angle表現を6D rotation表現に変換する。

    Rodrigues' formulaで回転行列を求め、最初の2列をフラット化して6Dにする。

    Args:
        axis_angle: axis-angle表現テンソル (3,)。

    Returns:
        6D rotation表現テンソル (6,)。
    """
    angle = torch.norm(axis_angle)
    if angle < 1e-8:
        return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    axis = axis_angle / angle
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    K = torch.zeros(3, 3)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]

    R = torch.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)
    return R[:, :2].T.reshape(6)


def _blendshapes_to_flame_eye(
    blendshapes: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """MediaPipe blendshapeスコアをFLAME eyes_pose/eyelidsに変換する。

    ARKit互換の52個のblendshapeスコアのうち、眼球と瞼に関連する
    スコアを使用してFLAME互換パラメータに変換する。

    変換マッピング:
        - pitch（上下）: eyeLookUp - eyeLookDown → axis-angle X成分
        - yaw（左右）: eyeLookOut - eyeLookIn → axis-angle Y成分
        - eyelids: eyeBlink スコアを直接使用

    Args:
        blendshapes: blendshape名→スコア（0-1）の辞書。

    Returns:
        2要素のタプル:
            - ``eyes_pose``: (1, 12) 左右眼球の6D rotation。
            - ``eyelids``: (1, 2) 左右瞼の開閉度。
    """
    # 左眼のblendshapeスコア
    left_up = blendshapes.get("eyeLookUpLeft", 0.0)
    left_down = blendshapes.get("eyeLookDownLeft", 0.0)
    left_in = blendshapes.get("eyeLookInLeft", 0.0)
    left_out = blendshapes.get("eyeLookOutLeft", 0.0)

    # 右眼のblendshapeスコア
    right_up = blendshapes.get("eyeLookUpRight", 0.0)
    right_down = blendshapes.get("eyeLookDownRight", 0.0)
    right_in = blendshapes.get("eyeLookInRight", 0.0)
    right_out = blendshapes.get("eyeLookOutRight", 0.0)

    # 瞼スコア
    blink_left = blendshapes.get("eyeBlinkLeft", 0.0)
    blink_right = blendshapes.get("eyeBlinkRight", 0.0)

    # pitch: 上方向が負、下方向が正
    left_pitch = (left_down - left_up) * _EYE_PITCH_SCALE
    right_pitch = (right_down - right_up) * _EYE_PITCH_SCALE

    # yaw: 外側が正（左眼ではout=正、右眼ではout=正の逆向き）
    left_yaw = (left_out - left_in) * _EYE_YAW_SCALE
    right_yaw = (right_in - right_out) * _EYE_YAW_SCALE

    # axis-angle → 6D rotation
    left_aa = torch.tensor([left_pitch, left_yaw, 0.0])
    right_aa = torch.tensor([right_pitch, right_yaw, 0.0])

    left_6d = _axis_angle_to_rotation_6d(left_aa)
    right_6d = _axis_angle_to_rotation_6d(right_aa)

    eyes_pose = torch.cat([left_6d, right_6d]).unsqueeze(0)
    eyelids = torch.tensor([[blink_left, blink_right]])

    return eyes_pose, eyelids


class FaceDetector:
    """MediaPipeベースの顔検出・クロッピングクラス。

    MediaPipe Face Detectionを使用して入力フレームから顔領域を検出し、
    バウンディングボックスを返す。検出された顔領域は指定サイズに
    クロッピング・リサイズできる。

    仕様書§8.3の設計に従い、Extractorから独立したモジュールとして機能する。
    DECA encode()に内部FAN処理がないことが確認されているため、
    本モジュールとDECAの共存にバイパス改修は不要。

    MediaPipe v0.10.11以前（``mp.solutions`` API）と v0.10.14以降
    （``mp.tasks`` API）の両方に自動対応する。

    Attributes:
        _min_detection_confidence: 検出信頼度の下限閾値。
        _model_path: ``mp.tasks`` API使用時のモデルファイルパス。
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ) -> None:
        """FaceDetectorを初期化する。

        Args:
            min_detection_confidence: 顔検出の最小信頼度。0.0〜1.0の範囲。
                この閾値を下回る検出結果は無視される。
            model_path: ``mp.tasks`` API使用時のBlazeFaceモデルファイルパス。
                Noneの場合はMediaPipeバンドルモデルを使用する（``mp.solutions``
                API の場合は無視される）。

        Raises:
            RuntimeError: MediaPipeの初期化に失敗した場合。
        """
        self._min_detection_confidence = min_detection_confidence
        self._model_path = model_path
        self._solutions_detector: object | None = None
        self._tasks_detector: object | None = None

        if _USE_SOLUTIONS_API:
            self._solutions_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=min_detection_confidence,
            )
        else:
            self._init_tasks_api()

    def _init_tasks_api(self) -> None:
        """MediaPipe Tasks APIでFace Detectorを初期化する。

        Raises:
            RuntimeError: モデルパスが未指定で、かつバンドルモデルが
                見つからない場合。
        """
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceDetector as MpFaceDetector,
            FaceDetectorOptions,
        )

        if self._model_path is not None:
            base_options = BaseOptions(model_asset_path=self._model_path)
        else:
            import importlib.resources
            import pathlib

            mp_root = pathlib.Path(mp.__file__).parent
            candidates = [
                mp_root / "modules" / "face_detection" / "face_detection_short_range.tflite",
                mp_root / "modules" / "face_detection" / "face_detection_full_range_sparse.tflite",
            ]
            found_path: str | None = None
            for candidate in candidates:
                if candidate.exists():
                    found_path = str(candidate)
                    break

            if found_path is None:
                raise RuntimeError(
                    "MediaPipe Tasks API requires a BlazeFace model file. "
                    "Please provide model_path argument or install MediaPipe "
                    "v0.10.11 which includes mp.solutions API. "
                    f"Searched: {[str(c) for c in candidates]}"
                )
            base_options = BaseOptions(model_asset_path=found_path)

        options = FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self._min_detection_confidence,
        )
        self._tasks_detector = MpFaceDetector.create_from_options(options)

    def detect(self, frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """フレームから顔のバウンディングボックスを検出する。

        複数の顔が検出された場合、最も信頼度の高い顔のバウンディングボックスを
        返す。

        Args:
            frame: BGR形式の入力画像。形状は ``(H, W, 3)``、dtype は ``uint8``。

        Returns:
            検出された顔のバウンディングボックス ``(x1, y1, x2, y2)``。
            各値はピクセル座標（整数）。顔が検出されなかった場合は ``None``。
        """
        if _USE_SOLUTIONS_API:
            return self._detect_solutions(frame)
        return self._detect_tasks(frame)

    def _detect_solutions(
        self, frame: np.ndarray
    ) -> Optional[tuple[int, int, int, int]]:
        """mp.solutions APIを使用した顔検出。

        Args:
            frame: BGR形式の入力画像。形状は ``(H, W, 3)``。

        Returns:
            バウンディングボックスまたはNone。
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._solutions_detector.process(rgb)

        if results.detections is None or len(results.detections) == 0:
            return None

        best = max(results.detections, key=lambda d: d.score[0])
        bbox_rel = best.location_data.relative_bounding_box

        x1 = max(0, int(bbox_rel.xmin * w))
        y1 = max(0, int(bbox_rel.ymin * h))
        x2 = min(w, int((bbox_rel.xmin + bbox_rel.width) * w))
        y2 = min(h, int((bbox_rel.ymin + bbox_rel.height) * h))

        return (x1, y1, x2, y2)

    def _detect_tasks(
        self, frame: np.ndarray
    ) -> Optional[tuple[int, int, int, int]]:
        """mp.tasks APIを使用した顔検出。

        Args:
            frame: BGR形式の入力画像。形状は ``(H, W, 3)``。

        Returns:
            バウンディングボックスまたはNone。
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._tasks_detector.detect(mp_image)

        if result.detections is None or len(result.detections) == 0:
            return None

        best = max(
            result.detections,
            key=lambda d: d.categories[0].score if d.categories else 0.0,
        )
        bb = best.bounding_box
        x1 = max(0, bb.origin_x)
        y1 = max(0, bb.origin_y)
        x2 = min(w, bb.origin_x + bb.width)
        y2 = min(h, bb.origin_y + bb.height)

        return (x1, y1, x2, y2)

    def crop_and_align(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        size: int = 224,
        margin_scale: float = 1.25,
    ) -> np.ndarray:
        """バウンディングボックスで顔領域を切り出し、指定サイズにリサイズする。

        バウンディングボックスを正方形に拡張した後、フレームからクロッピングし、
        指定サイズにリサイズする。フレーム外にはみ出す場合はゼロパディングを行う。

        Args:
            frame: BGR形式の入力画像。形状は ``(H, W, 3)``、dtype は ``uint8``。
            bbox: 顔のバウンディングボックス ``(x1, y1, x2, y2)``。
                ``detect()`` の戻り値をそのまま渡す。
            size: 出力画像の一辺のサイズ（ピクセル）。正方形画像を生成する。
            margin_scale: bbox に乗じる余白倍率。DECA の訓練分布に合わせ
                デフォルト 1.25（DECA 公式 inference 時の scale margin と同値）。

        Returns:
            クロッピング・リサイズ済みの顔画像。形状は ``(size, size, 3)``、
            dtype は ``uint8``。
        """
        h, w, _ = frame.shape
        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half_side = int(max(x2 - x1, y2 - y1) * margin_scale / 2)

        sq_x1 = cx - half_side
        sq_y1 = cy - half_side
        sq_x2 = cx + half_side
        sq_y2 = cy + half_side

        pad_left = max(0, -sq_x1)
        pad_top = max(0, -sq_y1)
        pad_right = max(0, sq_x2 - w)
        pad_bottom = max(0, sq_y2 - h)

        crop_x1 = max(0, sq_x1)
        crop_y1 = max(0, sq_y1)
        crop_x2 = min(w, sq_x2)
        crop_y2 = min(h, sq_y2)

        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            cropped = cv2.copyMakeBorder(
                cropped,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        resized: np.ndarray = cv2.resize(
            cropped, (size, size), interpolation=cv2.INTER_LINEAR
        )
        return resized

    def detect_eye_pose(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """眼球ポーズと瞼パラメータを推定する。

        仕様書§4.3の方式A: MediaPipe Face LandmarkerのblendshapeスコアからFLAME
        互換のeyes_pose (12D) とeyelids (2D) を推定する。

        MediaPipe Face LandmarkerのblendshapeはARKit互換の52個のスコアを出力する。
        そのうち眼球回転に関連するblendshapeスコアを使用して、
        FLAME eye poseの6D rotation × 2（左右眼球）を推定する。

        blendshape → FLAME eye pose マッピング:
            - ``eyeLookUpLeft/Right``: 上方向回転 → pitch負方向
            - ``eyeLookDownLeft/Right``: 下方向回転 → pitch正方向
            - ``eyeLookInLeft/OutLeft``: 水平回転 → yaw
            - ``eyeLookInRight/OutRight``: 水平回転 → yaw
            - ``eyeBlinkLeft/Right``: 瞼閉じスコア → eyelids

        blendshapeが利用不可の場合（Face Landmarkerなし等）は、
        identity rotation（6D: [1,0,0,0,1,0]）の eyes_pose と
        ゼロの eyelids を返す。

        Args:
            frame: BGR形式の入力画像。形状は ``(H, W, 3)``、dtype は ``uint8``。
            bbox: 顔のバウンディングボックス ``(x1, y1, x2, y2)``。

        Returns:
            2要素のタプル:
                - ``eyes_pose``: 眼球回転 (1, 12)。左右各6D rotation。
                - ``eyelids``: 瞼パラメータ (1, 2)。左右各1D（0=開、1=閉）。
        """
        blendshapes = self._get_blendshapes(frame, bbox)

        if blendshapes is None:
            return _default_eyes_pose(), _default_eyelids()

        return _blendshapes_to_flame_eye(blendshapes)

    def _get_blendshapes(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> Optional[dict[str, float]]:
        """MediaPipe Face LandmarkerからblendshapeスコアをNode取得する。

        MediaPipe Face Landmarkerが利用可能な場合にblendshapeスコアを返す。
        利用不可の場合はNoneを返す。

        Args:
            frame: BGR形式の入力画像。形状は ``(H, W, 3)``。
            bbox: 顔のバウンディングボックス ``(x1, y1, x2, y2)``。

        Returns:
            blendshape名→スコアの辞書、またはNone。
        """
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
            )
            import pathlib

            mp_root = pathlib.Path(mp.__file__).parent
            model_candidates = [
                mp_root / "modules" / "face_landmarker" / "face_landmarker.task",
                mp_root / "modules" / "face_landmarker" / "face_landmarker_v2.task",
            ]
            model_path: Optional[str] = None
            for candidate in model_candidates:
                if candidate.exists():
                    model_path = str(candidate)
                    break

            if model_path is None:
                return None

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                output_face_blendshapes=True,
                num_faces=1,
            )
            landmarker = FaceLandmarker.create_from_options(options)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            landmarker.close()

            if (
                result.face_blendshapes is None
                or len(result.face_blendshapes) == 0
            ):
                return None

            blendshapes: dict[str, float] = {}
            for bs in result.face_blendshapes[0]:
                blendshapes[bs.category_name] = bs.score

            return blendshapes

        except (ImportError, RuntimeError, Exception):
            return None

    def estimate_eyes_pose(self, frame: np.ndarray) -> np.ndarray:
        """眼球ポーズを推定する（後方互換性のために保持）。

        新しい ``detect_eye_pose()`` メソッドの使用を推奨する。

        Args:
            frame: BGR形式の入力画像。形状は ``(H, W, 3)``。

        Returns:
            眼球ポーズパラメータ。形状は ``(12,)``（左右各6D rotation）。
        """
        eyes_pose, _ = self.detect_eye_pose(
            frame, (0, 0, frame.shape[1], frame.shape[0])
        )
        return eyes_pose.squeeze(0).numpy()

    def estimate_eyelids(self, frame: np.ndarray) -> np.ndarray:
        """瞼パラメータを推定する（後方互換性のために保持）。

        新しい ``detect_eye_pose()`` メソッドの使用を推奨する。

        Args:
            frame: BGR形式の入力画像。形状は ``(H, W, 3)``。

        Returns:
            瞼パラメータ。形状は ``(2,)``（左右各1D）。
        """
        _, eyelids = self.detect_eye_pose(
            frame, (0, 0, frame.shape[1], frame.shape[0])
        )
        return eyelids.squeeze(0).numpy()

    def release(self) -> None:
        """MediaPipeリソースを解放する。

        複数回呼び出しても安全。
        """
        if self._solutions_detector is not None:
            self._solutions_detector.close()
            self._solutions_detector = None
        if self._tasks_detector is not None:
            self._tasks_detector.close()
            self._tasks_detector = None
