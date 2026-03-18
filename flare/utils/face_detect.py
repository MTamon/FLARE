"""顔検出の責務分離 (Section 8.3)

設計方針 (Section 8.3):
  face_detect.py は共有ユーティリティとして位置づける。
  Extractor は検出済みの顔画像のみを受け取る設計とする。

  DECA FAN に関する発見 (v2.0):
    DECA の encode() 内部に FAN 顔検出は含まれない。
    FAN は前処理 (TestData の datasets/datasets.py) でのみ使用される。
    したがって、本ツールの face_detect.py と DECA の共存にバイパス改修は不要。

パイプライン処理フロー (Section 8.3):
    face_bbox = face_detector.detect(frame)
    cropped   = face_detector.crop_and_align(frame, face_bbox, size=224)
    tensor    = to_tensor(cropped).to(device)
    codedict  = deca_model.encode(tensor)   # FAN 呼び出しなし
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

from .errors import FaceNotDetectedError


# ---------------------------------------------------------------------------
# 型エイリアス
# ---------------------------------------------------------------------------
BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# 抽象基底クラス
# ---------------------------------------------------------------------------

class BaseFaceDetector(ABC):
    """顔検出器の抽象基底クラス。

    責務:
      - 入力画像から顔領域を検出しバウンディングボックスを返す
      - 顔アラインメント（ランドマーク検出含む）
    Extractor 側の責務ではない点に注意 (Section 8.3)。
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[BBox]:
        """フレームから顔バウンディングボックスを検出する。

        Args:
            frame: BGR 画像 (H, W, 3) uint8。

        Returns:
            検出された顔の BBox リスト。検出なしの場合は空リスト。
        """
        ...

    def detect_largest(self, frame: np.ndarray) -> BBox:
        """最大面積の顔を 1 つ返す。未検出時は FaceNotDetectedError。"""
        bboxes = self.detect(frame)
        if not bboxes:
            raise FaceNotDetectedError("No face detected in the frame.")
        return max(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


# ---------------------------------------------------------------------------
# OpenCV Haar / DNN ベースのデフォルト実装
# ---------------------------------------------------------------------------

class OpenCVFaceDetector(BaseFaceDetector):
    """OpenCV DNN ベースの SSD 顔検出器。

    MediaPipe が利用可能な環境ではそちらに差し替え可能。
    本クラスは opencv-python 同梱の Caffe モデルをフォールバックとして使う。
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self._conf_thresh = confidence_threshold

        if model_path and config_path:
            self._net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        else:
            # Haar Cascade をフォールバック
            self._net = None
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect(self, frame: np.ndarray) -> List[BBox]:
        if self._net is not None:
            return self._detect_dnn(frame)
        return self._detect_haar(frame)

    def _detect_dnn(self, frame: np.ndarray) -> List[BBox]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self._net.setInput(blob)
        detections = self._net.forward()
        bboxes: List[BBox] = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self._conf_thresh:
                continue
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))
            bboxes.append((x1, y1, x2, y2))
        return bboxes

    def _detect_haar(self, frame: np.ndarray) -> List[BBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return [(x, y, x + w, y + h) for (x, y, w, h) in rects]


# ---------------------------------------------------------------------------
# クロップ・アライン
# ---------------------------------------------------------------------------

def crop_and_align(
    frame: np.ndarray,
    bbox: BBox,
    size: int = 224,
    margin: float = 0.15,
) -> np.ndarray:
    """バウンディングボックスに基づいて顔領域をクロップ＆リサイズする。

    Args:
        frame: BGR 画像 (H, W, 3)。
        bbox: (x1, y1, x2, y2)。
        size: 出力正方形のピクセルサイズ (Section 8.2: input_size=224)。
        margin: バウンディングボックスに対するマージン比率。

    Returns:
        (size, size, 3) BGR 画像。
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    side = int(max(bw, bh) * (1.0 + margin))
    half = side // 2

    # パディング付きクロップ
    nx1 = max(0, cx - half)
    ny1 = max(0, cy - half)
    nx2 = min(w, cx + half)
    ny2 = min(h, cy + half)
    crop = frame[ny1:ny2, nx1:nx2]

    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized


def crop_to_tensor(
    frame: np.ndarray,
    bbox: BBox,
    size: int = 224,
    device: str = "cpu",
) -> torch.Tensor:
    """crop_and_align + Tensor 変換のショートカット。

    Returns:
        (1, 3, size, size) float32 Tensor, 値域 [0, 1]。
    """
    crop = crop_and_align(frame, bbox, size=size)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )
    return tensor.to(device)